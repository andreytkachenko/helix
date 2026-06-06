use std::pin::Pin;

use async_stream::stream;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;

use crate::{
    agent::StopReason,
    message::{Content, Message, Role},
    model::{Model, Usage},
    providers::{AssistantMessageEvent, Provider, ToolRegistry},
};

/// SSE JSON chunk from the OpenAI-compatible API.
#[derive(Debug, Deserialize)]
struct ChatChunk {
    choices: Vec<ChoiceDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChoiceDelta {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallChunk>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallChunk {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    _type: Option<String>,
    function: Option<FunctionChunk>,
}

#[derive(Debug, Deserialize)]
struct FunctionChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u32>,
}

/// Response from the /v1/models endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    base_url: String,
    api_key: String,
}

impl OpenAIProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self { base_url, api_key }
    }
}

/// Convert an internal Message to OpenAI-compatible format.
/// - Simple text → "content": "string"
/// - Thinking → included as text (OpenAI doesn't have a thinking type in requests)
/// - Tool calls → "tool_calls" array
/// - Tool results → "role": "tool" with "tool_call_id"
/// TODO need to use structs intead raw json value and get rid of unnecessery allocations
fn message_to_openai(msg: &Message) -> serde_json::Value {
    match msg.role {
        Role::Tool => {
            // Tool result message
            let text: String = msg.content.iter().map(|c| c.to_string()).collect();
            serde_json::json!({
                "role": "tool",
                "content": text,
                "tool_call_id": msg.tool_call_id,
            })
        }
        Role::Assistant => {
            let mut has_text = false;
            let mut has_thinking = false;
            let mut has_tool_calls = false;
            for c in &msg.content {
                match c {
                    Content::Text { .. } => has_text = true,
                    Content::Thinking { .. } => has_thinking = true,
                    Content::ToolCall { .. } => has_tool_calls = true,
                    Content::Image { .. } => has_text = true,
                }
            }

            let mut json = serde_json::Map::new();
            json.insert("role".to_string(), serde_json::json!("assistant"));

            // Build content string from text parts
            let mut text_parts: Vec<String> = Vec::new();
            for c in &msg.content {
                match c {
                    Content::Text { text } => text_parts.push(text.clone()),
                    Content::Thinking { thinking } => {
                        // Include thinking as text for OpenAI (some servers support it in context)
                        text_parts.push(thinking.clone());
                    }
                    Content::Image { image_url, format: _format } => {
                        // OpenAI image format
                        let img_val = if let Some(fmt) = _format {
                            serde_json::json!({ "url": format!("data:image/{};base64,{}", fmt, image_url) })
                        } else {
                            serde_json::json!({ "url": image_url })
                        };
                        json.insert("content".to_string(),
                            serde_json::json!([{ "type": "image_url", "image_url": img_val }]));
                        has_text = false; // handled above
                    }
                    Content::ToolCall { .. } => {}
                }
            }

            if has_text || has_thinking {
                json.insert("content".to_string(), serde_json::json!(text_parts.join("\n")));
            } else if !has_tool_calls {
                json.insert("content".to_string(), serde_json::json!(null));
            }

            // Build tool_calls array
            if has_tool_calls {
                let calls: Vec<serde_json::Value> = msg.content.iter().filter_map(|c| {
                    if let Content::ToolCall { id, name, arguments } = c {
                        Some(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments.to_string(),
                            }
                        }))
                    } else {
                        None
                    }
                }).collect();
                json.insert("tool_calls".to_string(), serde_json::json!(calls));
            }

            serde_json::Value::Object(json)
        }
        _ => {
            // System, User — simple text content as string
            let text: String = msg.content.iter().map(|c| c.to_string()).collect();
            serde_json::json!({
                "role": msg.role.as_str(),
                "content": text,
            })
        }
    }
}

#[async_trait::async_trait]
impl Provider for OpenAIProvider {
    async fn models(&self) -> Result<Vec<Model>, String> {
        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", self.base_url);
        let resp = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| format!("Failed to fetch models: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("Models request failed with status {}", resp.status()));
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse models response: {}", e))?;

        Ok(body
            .data
            .into_iter()
            .map(|m| Model {
                id: m.id.clone(),
                name: m.id,
                api: "openai".to_string(),
                provider: "openai".to_string(),
                base_url: Some(self.base_url.clone()),
                reasoning: false,
                input: vec![],
                cost: Default::default(),
                context_window: 128000,
                max_tokens: 8192,
            })
            .collect())
    }

    async fn stream(
        &self,
        model: &Model,
        tools: &dyn ToolRegistry,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + '_>>, String> {
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model_id = model.id.clone();
        let tool_defs: Vec<serde_json::Value> = {
            let mut defs = Vec::new();
            tools.for_each(&mut |t| {
                defs.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.input_schema(),
                    }
                }));
            });
            defs
        };

        // Convert internal messages to OpenAI-compatible format
        let openai_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| message_to_openai(&msg))
            .collect();

        Ok(Box::pin(stream! {
            yield AssistantMessageEvent::start();

            let client = reqwest::Client::new();
            let url = format!("{}/v1/chat/completions", base_url);

            let body = serde_json::json!({
                "model": model_id,
                "messages": openai_messages,
                "stream": true,
                "stream_options": { "include_usage": true },
                "tools": if tool_defs.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(tool_defs)
                },
            });

            let req = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body);

            let res = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield AssistantMessageEvent::error(format!("request failed: {}", e));
                    yield AssistantMessageEvent::done(StopReason::Error, Usage::default());
                    return;
                }
            };

            if !res.status().is_success() {
                let status = res.status();
                let body = match res.text().await { Ok(b) => b, Err(e) => e.to_string() };
                yield AssistantMessageEvent::error(format!("HTTP {}: {}", status, body));
                yield AssistantMessageEvent::done(StopReason::Error, Usage::default());
                return;
            }

            // Streaming state
            let mut content_index: usize = 0;
            let mut thinking_index: Option<usize> = None;
            let mut did_start_text = false;
            let mut has_finish_reason = false;
            let mut stop_reason = StopReason::Stop;

            // Accumulators
            let mut assistant_content = String::new();
            let mut assistant_thinking = String::new();
            let mut tool_calls: Vec<(usize, String, String, String)> = Vec::new();
            let mut prompt_tokens = 0u64;
            let mut completion_tokens = 0u64;

            // Process SSE stream line by line
            let mut lines = res.bytes_stream();
            let mut line_buffer = String::new();

            while let Some(chunk_result) = lines.next().await {
                let chunk = match chunk_result {
                    Ok(bytes) => match std::str::from_utf8(&bytes) {
                        Ok(s) => s.to_string(),
                        Err(_) => continue,
                    },
                    Err(_) => break,
                };

                line_buffer.push_str(&chunk);

                // Process complete lines
                while let Some(pos) = line_buffer.find('\n') {
                    let line = line_buffer[..pos].trim().to_string();
                    line_buffer = line_buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    let data = line.trim_start_matches("data: ").trim();
                    if data == "[DONE]" {
                        continue;
                    }

                    let chunk: ChatChunk = match serde_json::from_str(data) {
                        Ok(c) => c,
                        Err(err) => {
                            yield AssistantMessageEvent::error(format!("Parse error: {err}"));
                            continue;
                        }
                    };

                    if let Some(usage) = chunk.usage {
                        prompt_tokens += usage.prompt_tokens.unwrap_or(0) as u64;
                        completion_tokens += usage.completion_tokens.unwrap_or(0) as u64;
                    }

                    let Some(choice) = chunk.choices.into_iter().next() else {
                        continue;
                    };

                    let delta = choice.delta;

                    // Track finish reason
                    if let Some(ref reason) = choice.finish_reason {
                        if !has_finish_reason {
                            stop_reason = match reason.as_str() {
                                "stop" => StopReason::Stop,
                                "length" => StopReason::Length,
                                "tool_calls" => StopReason::ToolUse,
                                "error" => StopReason::Error,
                                _ => StopReason::Stop,
                            };
                            has_finish_reason = true;
                        }
                    }

                    // Handle reasoning/thinking content
                    // Supports both "reasoning" (OpenAI) and "reasoning_content" (llama-server/Qwen)
                    let reasoning = delta
                        .reasoning
                        .as_deref()
                        .or(delta.reasoning_content.as_deref());

                    if let Some(reasoning_text) = reasoning {
                        if !reasoning_text.is_empty() {
                            // If we were outputting text, finish it first
                            if !assistant_content.is_empty() {
                                yield AssistantMessageEvent::text_end(0, std::mem::take(&mut assistant_content));
                            }

                            // Start thinking if not already started
                            if assistant_thinking.is_empty() {
                                thinking_index = Some(content_index);
                                yield AssistantMessageEvent::thinking_start(content_index);
                            }

                            assistant_thinking.push_str(reasoning_text);
                            yield AssistantMessageEvent::thinking_delta(
                                thinking_index.unwrap(),
                                reasoning_text.to_string(),
                            );
                        }
                    }

                    // Handle regular content
                    if let Some(ref text) = delta.content {
                        if !text.is_empty() {
                            // If we were thinking, finish thinking first
                            if !assistant_thinking.is_empty() {
                                yield AssistantMessageEvent::thinking_end(0, std::mem::take(&mut assistant_thinking));
                            }

                            // Start text if not already started
                            if !did_start_text {
                                did_start_text = true;
                                content_index += 1;
                                yield AssistantMessageEvent::text_start(content_index - 1);
                            }

                            assistant_content.push_str(text);
                            yield AssistantMessageEvent::text_delta(content_index - 1, text.clone());
                        }
                    }

                    // Handle tool calls
                    if let Some(ref calls) = delta.tool_calls {
                        // Finish any pending content before tool calls
                        if !assistant_thinking.is_empty() {
                            yield AssistantMessageEvent::thinking_end(0, std::mem::take(&mut assistant_thinking));
                        }
                        if !assistant_content.is_empty() {
                            yield AssistantMessageEvent::text_end(0, std::mem::take(&mut assistant_content));
                        }

                        for tc in calls {
                            let idx = tc.index;
                            let Some(ref f) = tc.function else {
                                continue;
                            };

                            // Get id (only present in first chunk)
                            let id = match &tc.id {
                                Some(id) => id.clone(),
                                None => {
                                    // Subsequent chunk - find existing tool call by index
                                    if let Some(last) = tool_calls.iter_mut().find(|(i, _, _, _)| *i == idx) {
                                        last.3.push_str(f.arguments.as_deref().unwrap_or(""));
                                        continue;
                                    } else {
                                        continue;
                                    }
                                }
                            };

                            let fname = f.name.as_deref().unwrap_or("");
                            let args = f.arguments.as_deref().unwrap_or("");

                            // Accumulate arguments for this tool call
                            if let Some(last) = tool_calls.iter_mut().find(|(i, _, _, _)| *i == idx) {
                                last.3.push_str(args);
                            } else {
                                tool_calls.push((idx, id, fname.to_string(), args.to_string()));
                            }
                        }
                    }
                }
            }

            // Finalize text
            if did_start_text {
                let ci = content_index - 1;
                yield AssistantMessageEvent::text_end(ci, assistant_content.clone());
            }

            // Emit tool calls
            if !tool_calls.is_empty() {
                yield AssistantMessageEvent::tool_call_start();
                for (idx, id, fname, fargs) in tool_calls {
                    let parsed_args = match serde_json::from_str(&fargs) {
                        Ok(args) => args,
                        Err(err) => {
                            yield AssistantMessageEvent::error(format!("toolcall argument parse error: {err}"));
                            continue;
                        }
                    };
                    yield AssistantMessageEvent::tool_call(idx, id, fname, parsed_args);
                }
                yield AssistantMessageEvent::tool_call_end();
            }

            yield AssistantMessageEvent::done(stop_reason, Usage {
                input: prompt_tokens,
                output: completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            });
        }))
    }
}
