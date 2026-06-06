use std::path::PathBuf;

use crate::{
    abort::AbortSignal,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolResult, canonicalize_path},
};

#[derive(Debug)]
pub struct LsTool {
    cwd: PathBuf,
    max_lines: usize,
}

impl LsTool {
    pub fn new(cwd: PathBuf, max_lines: u64) -> Self {
        Self {
            cwd,
            max_lines: max_lines as _,
        }
    }
}

#[async_trait::async_trait]
impl AgentTool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }
    fn label(&self) -> &str {
        "List Directory"
    }
    fn description(&self) -> &str {
        "List files and directories in the given path"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The directory path to list" }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        id: String,
        params: serde_json::Value,
        _signal: Option<AbortSignal>,
        _update: &(dyn Fn(String, String) + Send + Sync),
    ) -> Result<ToolResult, ToolError> {
        let path_str = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let path = canonicalize_path(&self.cwd, path_str);
        let mut read_dir = tokio::fs::read_dir(&path).await?;
        let mut list = Vec::new();

        while let Some(entry) = read_dir.next_entry().await? {
            list.push(if entry.file_type().await?.is_dir() {
                format!("{}/", entry.file_name().display())
            } else {
                format!("{}", entry.file_name().display())
            })
        }

        list.sort_by_key(|a| a.to_lowercase());

        let truncated = list.len() > self.max_lines;
        let take = if truncated {
            self.max_lines
        } else {
            list.len()
        };

        let count_str = if truncated {
            format!("{} ({} total)", take, list.len())
        } else {
            format!("{}", list.len())
        };

        list.truncate(self.max_lines);

        Ok(ToolResult {
            id,
            content: vec![Content::Text {
                text: list.join("\n"),
            }],
            details: serde_json::json!({"count":count_str,"path":path.display().to_string(),"truncated":truncated,}),
            terminate: false,
        })
    }
}
