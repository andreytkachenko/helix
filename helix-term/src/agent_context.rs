use std::path::Path;
use std::sync::Arc;

use helix_agent::agent::{RunContext, StopReason, TurnEndInfo};
use helix_agent::message::{Content, Message, Role};
use helix_agent::tools::ToolResult;
use helix_view::editor::{AgentStatus, DocumentUpdate};

/// Bridge between the agent process and the helix editor.
/// Implements `RunContext` to update the editor's agent session in real-time.
/// Updates are made via Arc<RwLock<>> fields on the AgentSession.
pub struct HelixRunContext {
    /// Reference to the agent session's Arc fields for thread-safe updates.
    assistant_text: Arc<std::sync::RwLock<Option<String>>>,
    assistant_thinking: Arc<std::sync::RwLock<Option<String>>>,
    active_tools: Arc<std::sync::RwLock<Vec<helix_view::editor::ActiveTool>>>,
    status: Arc<std::sync::RwLock<AgentStatus>>,
    /// Conversation messages (synced to session for UI display).
    messages: Arc<std::sync::RwLock<Vec<Message>>>,
    /// Steering queue for user input during agent execution.
    steering_queue: Arc<crossbeam_queue::SegQueue<Content>>,
    /// Stop signal.
    stop_requested: Arc<std::sync::atomic::AtomicBool>,
    /// Pending document updates from tool execution.
    document_updates: Arc<std::sync::RwLock<Vec<DocumentUpdate>>>,
    /// Base working directory for resolving relative file paths.
    cwd: std::path::PathBuf,
}

impl HelixRunContext {
    pub fn new(
        assistant_text: Arc<std::sync::RwLock<Option<String>>>,
        assistant_thinking: Arc<std::sync::RwLock<Option<String>>>,
        active_tools: Arc<std::sync::RwLock<Vec<helix_view::editor::ActiveTool>>>,
        status: Arc<std::sync::RwLock<AgentStatus>>,
        messages: Arc<std::sync::RwLock<Vec<Message>>>,
        steering_queue: Arc<crossbeam_queue::SegQueue<Content>>,
        stop_requested: Arc<std::sync::atomic::AtomicBool>,
        document_updates: Arc<std::sync::RwLock<Vec<DocumentUpdate>>>,
        cwd: std::path::PathBuf,
    ) -> Self {
        Self {
            assistant_text,
            assistant_thinking,
            active_tools,
            status,
            messages,
            steering_queue,
            stop_requested,
            document_updates,
            cwd,
        }
    }

    /// Check if a stop has been requested.
    pub fn is_stopped(&self) -> bool {
        self.stop_requested.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Request a stop.
    pub fn request_stop(&self) {
        self.stop_requested
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the steering queue.
    pub fn steering_queue(&self) -> &Arc<crossbeam_queue::SegQueue<Content>> {
        &self.steering_queue
    }

    /// Resolve a file path: if relative, join with cwd, then canonicalize if possible.
    fn resolve_path(&self, path_str: &str) -> std::path::PathBuf {
        let path = if Path::new(path_str).is_absolute() {
            std::path::PathBuf::from(path_str)
        } else {
            self.cwd.join(path_str)
        };
        // Try to canonicalize to get the real absolute path
        std::fs::canonicalize(&path).unwrap_or_else(|_| path)
    }
}

impl RunContext for HelixRunContext {
    fn turn_start(&self, _turn_number: u64) {
        *self.status.write().unwrap() = AgentStatus::Thinking;
        helix_event::request_redraw();
    }

    fn turn_end(&self, info: TurnEndInfo<'_>) -> bool {
        let should_stop = match info.stop_reason {
            StopReason::Stop => {
                *self.status.write().unwrap() = AgentStatus::Idle;
                true // Stop the agent loop
            }
            StopReason::ToolUse => {
                *self.status.write().unwrap() = AgentStatus::Working;
                false // Continue the loop
            }
            StopReason::Error => {
                *self.status.write().unwrap() = AgentStatus::Idle;
                true
            }
            StopReason::Aborted => {
                *self.status.write().unwrap() = AgentStatus::Idle;
                true
            }
            StopReason::Length => {
                *self.status.write().unwrap() = AgentStatus::Idle;
                true
            }
        };
        helix_event::request_redraw();
        should_stop
    }

    fn next_steering(&self) -> Option<Content> {
        self.steering_queue.pop()
    }

    fn queue_steering(&self, msg: Content) {
        self.steering_queue.push(msg);
    }

    fn user_message(&self, msg: &Content) {
        // Add user message to session for UI display
        self.messages.write().unwrap().push(Message {
            role: Role::User,
            content: vec![msg.clone()],
            tool_call_id: None,
            timestamp: 0,
        });
        helix_event::request_redraw();
    }

    // ── Assistant message lifecycle ─────────────────────────────────────────

    fn assistant_start(&self, is_thinking: bool) -> u64 {
        if is_thinking {
            *self.assistant_thinking.write().unwrap() = Some(String::new());
            *self.status.write().unwrap() = AgentStatus::Thinking;
        } else {
            *self.assistant_text.write().unwrap() = Some(String::new());
            *self.status.write().unwrap() = AgentStatus::Streaming;
        }
        helix_event::request_redraw();
        0 // Simple ID for single content
    }

    fn assistant_update(&self, _id: u64, delta: &str) -> bool {
        // Append delta to the currently active field (text or thinking)
        if let Some(ref mut text) = *self.assistant_text.write().unwrap() {
            text.push_str(delta);
        }
        if let Some(ref mut thinking) = *self.assistant_thinking.write().unwrap() {
            thinking.push_str(delta);
        }
        // Trigger UI redraw so streaming text appears in real-time
        helix_event::request_redraw();
        false // Don't stop
    }

    fn assistant_end(&self, _id: u64) {
        // Save the finished assistant message to session.messages for UI history
        let text = self.assistant_text.read().unwrap().clone();
        let thinking = self.assistant_thinking.read().unwrap().clone();

        if let Some(ref t) = text {
            if !t.is_empty() {
                self.messages.write().unwrap().push(Message {
                    role: Role::Assistant,
                    content: vec![Content::Text { text: t.clone() }],
                    tool_call_id: None,
                    timestamp: 0,
                });
            }
        }
        if let Some(ref t) = thinking {
            if !t.is_empty() {
                self.messages.write().unwrap().push(Message {
                    role: Role::Assistant,
                    content: vec![Content::Thinking { thinking: t.clone() }],
                    tool_call_id: None,
                    timestamp: 0,
                });
            }
        }

        // Clear the streaming buffers
        *self.assistant_text.write().unwrap() = None;
        *self.assistant_thinking.write().unwrap() = None;

        // Don't set Idle here - status is managed by turn_start/turn_end
        // If tools will execute, turn_end will set Working
        // If done, turn_end will set Idle
    }

    // ── Tool execution lifecycle ────────────────────────────────────────────

    fn tool_start(&self, id: &str, name: &str, args: serde_json::Value) {
        self.active_tools
            .write()
            .unwrap()
            .push(helix_view::editor::ActiveTool {
                id: id.to_string(),
                name: name.to_string(),
                args: args.clone(),
            });

        // Add tool call to session messages for UI
        self.messages.write().unwrap().push(Message {
            role: Role::Assistant,
            content: vec![Content::ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: args,
            }],
            tool_call_id: None,
            timestamp: 0,
        });

        *self.status.write().unwrap() = AgentStatus::Working;
        helix_event::request_redraw();
    }

    fn tool_update(&self, _id: &str, _delta: &str) -> bool {
        false
    }

    fn tool_end(&self, id: &str, result: &ToolResult) {
        self.active_tools
            .write()
            .unwrap()
            .retain(|t| t.id != id);

        // Add tool result to session messages for UI
        self.messages.write().unwrap().push(Message {
            role: Role::Tool,
            content: result.content.clone(),
            tool_call_id: Some(id.to_string()),
            timestamp: 0,
        });

        // Check if the tool result contains file path info for document updates
        if let Some(path_str) = result.details.get("path").and_then(|v| v.as_str()) {
            // Resolve the path (canonicalize if possible)
            let resolved_path = self.resolve_path(path_str);

            // Read the file content to update the document
            if let Ok(content) = std::fs::read_to_string(&resolved_path) {
                self.document_updates
                    .write()
                    .unwrap()
                    .push(DocumentUpdate {
                        path: resolved_path,
                        content,
                    });
            }
        }

        // Keep Working status - turn_end will set the final status
        *self.status.write().unwrap() = AgentStatus::Working;
        helix_event::request_redraw();
    }

    fn agent_error(&self, error: String) {
        self.messages.write().unwrap().push(Message {
            role: Role::Assistant,
            content: vec![Content::Text {
                text: format!("Error: {}", error),
            }],
            tool_call_id: None,
            timestamp: 0,
        });
        *self.status.write().unwrap() = AgentStatus::Idle;
        helix_event::request_redraw();
    }
}
