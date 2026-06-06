use std::path::PathBuf;

use crate::{
    abort::AbortSignal,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolExecutionMode, ToolResult, canonicalize_path},
};

#[derive(Debug)]
pub struct ReadFileTool {
    cwd: PathBuf,
    max_lines: u64,
}
impl ReadFileTool {
    pub fn new(cwd: PathBuf, max_lines: u64) -> Self {
        Self { cwd, max_lines }
    }
}

#[async_trait::async_trait]
impl AgentTool for ReadFileTool {
    fn name(&self) -> &str {
        "read"
    }
    fn label(&self) -> &str {
        "Read File"
    }
    fn description(&self) -> &str {
        "Read the complete content of a file"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        })
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }

    async fn execute(
        &self,
        id: String,
        params: serde_json::Value,
        _signal: Option<AbortSignal>,
        _update: &(dyn Fn(String, String) + Send + Sync),
    ) -> Result<ToolResult, ToolError> {
        let path_str = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or(ToolError::MissingParameter("path"))?;

        let path = canonicalize_path(&self.cwd, path_str);

        let metadata = tokio::fs::metadata(&path).await?;
        if metadata.is_dir() {
            return Ok(ToolResult {
                id,
                content: vec![Content::Text {
                    text: format!("Error: {} is a directory, not a file", path.display()),
                }],
                details: serde_json::json!({}),
                terminate: false,
            });
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let mut lines = content.lines();
        let mut first_line = String::new();
        let mut text = String::new();

        for (idx, line) in (&mut lines).take(self.max_lines as _).enumerate() {
            if idx == 0 {
                first_line.push_str(line);
            }

            text.push_str(&line.replace('\r', ""));
            text.push('\n');
        }

        let truncated = lines.next().is_some();

        Ok(ToolResult {
            id,
            content: vec![Content::Text { text }],
            details: serde_json::json!({
                "path": path.display().to_string(),
                "first_line": first_line,
                "truncated": truncated,
            }),
            terminate: false,
        })
    }
}
