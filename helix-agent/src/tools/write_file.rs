use std::path::PathBuf;

use crate::{
    abort::AbortSignal,
    diff::create_patch,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolExecutionMode, ToolResult, canonicalize_path},
};

#[derive(Debug)]
pub struct WriteFileTool {
    cwd: PathBuf,
}

impl WriteFileTool {
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }
}

#[async_trait::async_trait]
impl AgentTool for WriteFileTool {
    fn name(&self) -> &str {
        "write"
    }

    fn label(&self) -> &str {
        "Write File"
    }

    fn description(&self) -> &str {
        "Write/create a new file with the specified content. Creates parent directories automatically."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The file path to write" },
                "content": { "type": "string", "description": "The content to write" }
            },
            "required": ["path", "content"]
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

        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or(ToolError::MissingParameter("content"))?;

        let path = canonicalize_path(&self.cwd, path_str);

        let old_content = tokio::fs::read_to_string(&path).await.ok();

        let is_new_file = old_content.is_none();

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?
        }

        tokio::fs::write(&path, content).await?;

        let new_preview = content.lines().take(20).collect::<Vec<_>>().join("\n");

        // Compute diff if file existed before (rewrite scenario)
        let diff_text = old_content.map(|old| {
            let patch = create_patch(&old, content);
            patch
        });

        Ok(ToolResult {
            id,
            content: vec![Content::Text { text: new_preview }],
            details: serde_json::json!({
                "path": path.display().to_string(),
                "created": is_new_file,
                "diff": diff_text,
            }),
            terminate: false,
        })
    }
}
