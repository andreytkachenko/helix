use std::pin::Pin;

use futures_util::Stream;

use crate::{
    agent::StopReason,
    message::Message,
    model::{Model, Usage},
    tools::ToolRegistry,
};

pub mod openai;

#[derive(Debug, Clone)]
pub struct AssistantMessageEvent {
    pub timestamp: u64,
    pub message: AssistantMessage,
}

impl AssistantMessageEvent {
    pub fn error(error: String) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::Error { error },
        }
    }

    pub fn done(reason: StopReason, usage: Usage) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::Done { reason, usage },
        }
    }

    pub fn start() -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::Start,
        }
    }

    pub fn text_start(content_index: usize) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::TextStart { content_index },
        }
    }

    pub fn text_delta(content_index: usize, delta: String) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::TextDelta {
                content_index,
                delta,
            },
        }
    }

    pub fn text_end(content_index: usize, content: String) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::TextEnd {
                content_index,
                content,
            },
        }
    }

    pub fn thinking_start(content_index: usize) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::ThinkingStart { content_index },
        }
    }

    pub fn thinking_delta(content_index: usize, delta: String) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::ThinkingDelta {
                content_index,
                delta,
            },
        }
    }

    pub fn thinking_end(content_index: usize, content: String) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::ThinkingEnd {
                content_index,
                content,
            },
        }
    }

    pub fn tool_call_start() -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::ToolCallStart,
        }
    }

    pub fn tool_call(
        index: usize,
        id: String,
        tool_name: String,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::ToolCall {
                index,
                id,
                tool_name,
                arguments,
            },
        }
    }

    pub fn tool_call_end() -> Self {
        Self {
            timestamp: now_timestamp(),
            message: AssistantMessage::ToolCallEnd,
        }
    }
}

pub fn now_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[derive(Debug, Clone)]
pub enum AssistantMessage {
    Start,
    TextStart {
        content_index: usize,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    TextEnd {
        content_index: usize,
        content: String,
    },
    ThinkingStart {
        content_index: usize,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
    },
    ToolCallStart,
    ToolCall {
        index: usize,
        id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolCallEnd,
    Done {
        reason: StopReason,
        usage: Usage,
    },
    Error {
        error: String,
    },
}

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    async fn models(&self) -> Result<Vec<Model>, String>;

    async fn stream(
        &self,
        model: &Model,
        tools: &dyn ToolRegistry,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + '_>>, String>;
}
