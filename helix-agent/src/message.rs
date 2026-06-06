use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },

    Thinking {
        thinking: String,
    },

    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },

    Image {
        image_url: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
}

impl fmt::Display for Content {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Content::Text { text } => write!(f, "{text}"),
            Content::Thinking { thinking } => write!(f, "<think>{thinking}</think>"),
            Content::ToolCall {
                id,
                name,
                arguments,
            } => write!(
                f,
                "<tool-call id=\"{id}\" name=\"{name}\">{arguments}</tool-call>"
            ),
            Content::Image { .. } => write!(f, "[image]"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Standard LLM-compatible message roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "is_zero")]
    pub timestamp: u64,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl Message {
    pub fn system<T: ToString>(text: T) -> Message {
        Self {
            role: Role::System,
            content: vec![Content::Text {
                text: text.to_string(),
            }],
            timestamp: 0,
            tool_call_id: None,
        }
    }

    pub fn user<T: ToString>(text: T) -> Message {
        Self {
            role: Role::User,
            content: vec![Content::Text {
                text: text.to_string(),
            }],
            timestamp: 0,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: String, think: bool, timestamp: u64) -> Message {
        Self {
            role: Role::Assistant,
            content: vec![if think {
                Content::Thinking { thinking: content }
            } else {
                Content::Text { text: content }
            }],
            timestamp,
            tool_call_id: None,
        }
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;
    
    #[test]
    fn test_message_serialization() {
        let msg = Message::user("hello");
        let json = serde_json::to_string_pretty(&msg).unwrap();
        eprintln!("Message JSON:\n{}", json);
        assert!(json.contains("\"type\":\"text\""));
    }
    
    #[test]
    fn test_messages_in_body() {
        let msg = Message::user("hello");
        let body = serde_json::json!({
            "model": "test",
            "messages": vec![msg],
            "stream": true,
        });
        let json = serde_json::to_string_pretty(&body).unwrap();
        eprintln!("Body JSON:\n{}", json);
        assert!(json.contains("\"type\":\"text\""));
    }
}
