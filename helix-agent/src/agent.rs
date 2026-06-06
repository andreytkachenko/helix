use std::{collections::HashMap, fmt, pin::pin, sync::Arc};

use futures_util::{Stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    context::Context,
    error::Error,
    message::{Content, Message, Role},
    model::{Model, Usage},
    providers::Provider,
    tools::{ToolRegistry, ToolResult},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::Stop => write!(f, "stop"),
            StopReason::Length => write!(f, "length"),
            StopReason::ToolUse => write!(f, "tool_use"),
            StopReason::Error => write!(f, "error"),
            StopReason::Aborted => write!(f, "aborted"),
        }
    }
}

// ─── Thinking level ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub thinking_level: ThinkingLevel,
}

pub struct Agent {
    pub context: Arc<Mutex<Context>>,
    pub model: Arc<Model>,
    pub provider: Arc<dyn Provider + 'static>,
    pub tools: Arc<dyn ToolRegistry + 'static>,
    pub config: AgentConfig,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        prompt: String,
        model: Arc<Model>,
        history: Vec<Message>,
        provider: Arc<dyn Provider>,
        tools: Arc<dyn ToolRegistry>,
    ) -> Self {
        Self {
            config,
            context: Arc::new(Mutex::new(Context::new(prompt, history))),
            provider,
            tools,
            model,
        }
    }

    pub fn start(&self, prompt: String, ctx: impl RunContext + 'static) -> AgentProcess {
        let provider = self.provider.clone();
        let tools = self.tools.clone();
        let context = self.context.clone();
        let model = self.model.clone();

        AgentProcess {
            handler: tokio::spawn(async move {
                let mut stop_reason = StopReason::Stop;
                let mut total_usage = Usage::default();
                let mut tool_calls: Vec<Content> = Vec::new();
                let mut assistant_text: Option<String> = None;
                let mut assistant_thinking: Option<String> = None;
                let mut pending_messages = vec![Content::Text { text: prompt }];
                let mut pending_toolcalls: HashMap<String, ToolCall> = HashMap::new();
                let mut agent_loop_continue = true;
                let mut turn_number: u64 = 0;

                while agent_loop_continue {
                    // Check for stop request
                    if ctx.is_stopped() {
                        break;
                    }

                    turn_number += 1;

                    // Check for queued steering messages from RunContext
                    while let Some(val) = ctx.next_steering() {
                        pending_messages.push(val);
                    }

                    // ── Turn start ──────────────────────────────────────────────
                    ctx.turn_start(turn_number);

                    // Process pending steering messages into context
                    {
                        let mut lock = context.lock();
                        for msg in pending_messages.drain(..) {
                            ctx.user_message(&msg);

                            lock.add(Message {
                                role: Role::User,
                                content: vec![msg],
                                tool_call_id: None,
                                timestamp: 0,
                            });
                        }
                    }

                    let messages = context.lock().to_messages();
                    let stream = provider.stream(&model, &*tools, messages).await;
                    match &stream {
                        Ok(_) => {}
                        Err(e) => {
                            ctx.agent_error(e.clone());
                            break;
                        }
                    }
                    let mut stream = stream.unwrap();
                    let mut message_id = 0;

                    while let Some(event) = stream.next().await {
                        match event.message {
                            crate::providers::AssistantMessage::Start => (),
                            crate::providers::AssistantMessage::TextStart { .. } => {
                                message_id = ctx.assistant_start(false);
                            }
                            crate::providers::AssistantMessage::TextDelta { delta, .. } => {
                                if ctx.assistant_update(message_id, &delta) {
                                    break;
                                }
                            }
                            crate::providers::AssistantMessage::TextEnd { content, .. } => {
                                ctx.assistant_end(message_id);

                                assistant_text = Some(content.clone());

                                context.lock().add(Message::assistant(
                                    content,
                                    false,
                                    event.timestamp,
                                ));
                            }
                            crate::providers::AssistantMessage::ThinkingStart { .. } => {
                                message_id = ctx.assistant_start(true);
                            }
                            crate::providers::AssistantMessage::ThinkingDelta { delta, .. } => {
                                if ctx.assistant_update(message_id, &delta) {
                                    break;
                                }
                            }
                            crate::providers::AssistantMessage::ThinkingEnd { content, .. } => {
                                ctx.assistant_end(message_id);
                                assistant_thinking = Some(content.clone());

                                context.lock().add(Message::assistant(
                                    content,
                                    true,
                                    event.timestamp,
                                ));
                            }
                            crate::providers::AssistantMessage::ToolCallStart => {}
                            crate::providers::AssistantMessage::ToolCall {
                                index: _,
                                id,
                                tool_name,
                                arguments,
                            } => {
                                ctx.tool_start(&id, &tool_name, arguments.clone());

                                pending_toolcalls.insert(
                                    id.clone(),
                                    ToolCall {
                                        tool_name: tool_name.clone(),
                                        arguments: arguments.clone(),
                                    },
                                );

                                tool_calls.push(Content::ToolCall {
                                    id,
                                    name: tool_name,
                                    arguments,
                                })
                            }
                            crate::providers::AssistantMessage::ToolCallEnd => {}
                            crate::providers::AssistantMessage::Done { reason, usage } => {
                                total_usage.add(usage);
                                stop_reason = reason;
                            }
                            crate::providers::AssistantMessage::Error { error } => {
                                ctx.agent_error(error.clone());
                                context
                                    .lock()
                                    .add(Message::user(format!("Agent system error: {error}")));
                            }
                        }
                    }

                    // Execute tool calls if any
                    if !tool_calls.is_empty() {
                        context.lock().add(Message {
                            role: Role::Assistant,
                            content: std::mem::take(&mut tool_calls),
                            timestamp: 0,
                            tool_call_id: None,
                        });

                        let mut stream = pin!(
                            execute_tool_calls(&*tools, &pending_toolcalls, |id, delta| {
                                let _ = ctx.tool_update(&id, &delta);
                            })
                            .await
                        );

                        while let Some(result) = stream.next().await {
                            ctx.tool_end(&result.id, &result);

                            context.lock().add(Message {
                                role: Role::Tool,
                                content: result.content,
                                timestamp: 0,
                                tool_call_id: Some(result.id),
                            });
                        }
                    }

                    // Get full message list
                    let context_lock = context.lock();

                    // turn_end returns true to STOP, so invert for loop continuation
                    agent_loop_continue = !ctx.turn_end(TurnEndInfo {
                        stop_reason,
                        usage: total_usage,
                        tool_calls: &pending_toolcalls,
                        assistant_text: assistant_text.as_deref(),
                        assistant_thinking: assistant_thinking.as_deref(),
                        messages: context_lock.as_messages(),
                    });

                    // Clear pending toolcalls and tool_calls for next iteration
                    pending_toolcalls.clear();
                    tool_calls.clear();
                }
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub(crate) tool_name: String,
    pub(crate) arguments: serde_json::Value,
}

pub struct AgentProcess {
    handler: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for AgentProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentProcess").finish()
    }
}

impl AgentProcess {
    pub async fn wait(self) {
        self.handler.await.unwrap();
    }

    pub fn abort(&self) {
        self.handler.abort();
    }

    /// Get the inner JoinHandle for type compatibility.
    pub fn into_handle(self) -> tokio::task::JoinHandle<()> {
        self.handler
    }
    
    /// Check if the task is still running.
    pub fn is_started(&self) -> bool {
        !self.handler.is_finished()
    }
}

/// Information passed to `RunContext::turn_end` about what happened during a turn.
#[derive(Debug, Clone)]
pub struct TurnEndInfo<'a> {
    /// The reason the provider stopped generating (stop, length, tool_use, error).
    pub stop_reason: StopReason,
    /// Token usage for this turn.
    pub usage: Usage,
    /// Tool calls that were made in this turn.
    pub tool_calls: &'a HashMap<String, ToolCall>,
    /// The assistant's text response (if any — empty if only tool calls or thinking).
    pub assistant_text: Option<&'a str>,
    /// The assistant's thinking content (if any).
    pub assistant_thinking: Option<&'a str>,
    /// All messages in context at the end of this turn.
    pub messages: &'a [Message],
}

impl<'a> TurnEndInfo<'a> {
    /// Returns true if the last content from the assistant was plain text
    /// (not a tool call and not thinking).
    pub fn last_was_assistant_text(&self) -> bool {
        self.assistant_text.is_some()
            && self.assistant_thinking.is_none()
            && self.tool_calls.is_empty()
    }
}

pub trait RunContext: Send + Sync {
    /// Called at the start of each read turn. Default: no-op.
    fn turn_start(&self, _turn_number: u64) {}

    /// Called at the end of each read turn. Return true to stop the agent loop.
    /// The default logic: if last message was assistant text, stop; otherwise continue.
    fn turn_end(&self, _info: TurnEndInfo<'_>) -> bool;

    /// Check if a stop has been requested. Default: returns false (not stopped).
    fn is_stopped(&self) -> bool {
        false
    }

    /// Provide a steering message for the next iteration. Default: returns None.
    fn next_steering(&self) -> Option<Content> {
        None
    }

    /// Queue a steering message to be picked up by `next_steering`.
    fn queue_steering(&self, _msg: Content) {}

    /// Called when a new user message is about to be sent.
    fn user_message(&self, msg: &Content);

    // ── Assistant message lifecycle ─────────────────────────────────────────

    fn assistant_start(&self, is_thinking: bool) -> u64;
    fn assistant_update(&self, id: u64, delta: &str) -> bool;
    fn assistant_end(&self, id: u64);

    // ── Tool execution lifecycle ────────────────────────────────────────────

    fn tool_start(&self, id: &str, name: &str, args: serde_json::Value);
    fn tool_update(&self, id: &str, delta: &str) -> bool;
    fn tool_end(&self, id: &str, result: &ToolResult);

    // ── Error handling ──────────────────────────────────────────────────────

    fn agent_error(&self, error: String) {
        let _ = error;
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    // Agent lifecycle
    AgentStart,
    AgentEnd,

    // Turn lifecycle
    TurnStart,
    TurnEnd,

    // Message lifecycle
    MessageStart,
    MessageUpdate,
    MessageEnd,

    // Tool execution lifecycle
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: ToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: ToolResult,
        is_error: bool,
    },
}

async fn execute_tool_calls<'a, F: Fn(String, String) + Send + Sync + Clone + 'a>(
    tools: &'a dyn ToolRegistry,
    tool_calls: &'a HashMap<String, ToolCall>,
    cb: F,
) -> impl Stream<Item = ToolResult> + 'a {
    async_stream::stream! {
        for (id, tc) in tool_calls.iter() {
            yield match execute_tool(tools, id.clone(), tc.tool_name.clone(), tc.arguments.clone(), cb.clone()).await {
                Ok(ok) => ok,
                Err(err) => ToolResult::error(id.clone(), tc.tool_name.clone(), err),
            };
        }

    }
}

async fn execute_tool<F: Fn(String, String) + Send + Sync + Clone>(
    tools: &dyn ToolRegistry,
    id: String,
    tool_name: String,
    arguments: serde_json::Value,
    update: F,
) -> Result<ToolResult, Error> {
    let tool = tools
        .get_tool(&tool_name)
        .ok_or_else(|| Error::ToolNotFound(tool_name.clone()))?;

    // Execute
    let result = tool
        .execute(id.clone(), arguments, None, &update)
        .await
        .map_err(|err| Error::ToolError(tool_name, err))?;

    Ok(result)
}
