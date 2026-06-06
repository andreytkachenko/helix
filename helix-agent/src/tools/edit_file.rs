use std::path::PathBuf;

use crate::{
    abort::AbortSignal,
    diff::create_patch,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolExecutionMode, ToolResult, canonicalize_path},
};

#[derive(Debug)]
pub struct EditFileTool {
    cwd: PathBuf,
}

impl EditFileTool {
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }
}

#[async_trait::async_trait]
impl AgentTool for EditFileTool {
    fn name(&self) -> &str {
        "edit"
    }
    fn label(&self) -> &str {
        "Edit File"
    }
    fn description(&self) -> &str {
        "Make targeted edits to a file. Each edit replaces exact `old_text` with `new_text`."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The file path to edit" },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": { "type": "string", "description": "Text to find and replace" },
                            "new_text": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["old_text", "new_text"]
                    },
                    "description": "List of edits to apply"
                }
            },
            "required": ["path", "edits"]
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

        let edits: Vec<Edit> = serde_json::from_value(params["edits"].clone())
            .map_err(|e| ToolError::ParameterWrongFormat("edits", e))?;

        let path = canonicalize_path(&self.cwd, path_str);

        let content = tokio::fs::read_to_string(&path).await?;

        let mut updated = String::new();
        let mut applied = vec![];

        let mut remaining: &str = &content;
        for edit in &edits {
            if let Some(pos) = remaining.find(&edit.old_text) {
                updated.push_str(&remaining[..pos]);
                updated.push_str(&edit.new_text);
                remaining = &remaining[pos + edit.old_text.len()..];
                applied.push(pos);
            } else {
                return Ok(ToolResult {
                    id,
                    content: vec![Content::Text {
                        text: format!("Edit failed: '{}' not found", edit.old_text),
                    }],
                    details: serde_json::json!({"edits":applied}),
                    terminate: false,
                });
            }
        }

        let new_content = format!("{}{}", updated, remaining);
        tokio::fs::write(&path, &new_content).await?;

        let preview = new_content.lines().take(50).collect::<Vec<_>>().join("\n");

        // Build diffs for each applied edit
        let diffs: Vec<String> = edits
            .iter()
            .filter_map(|edit| {
                // Only include if there's actually a diff
                if edit.old_text == edit.new_text {
                    return None;
                }
                let patch = create_patch(&edit.old_text, &edit.new_text);
                if patch.is_empty() {
                    return None;
                }
                Some(patch)
            })
            .collect();

        Ok(ToolResult {
            id,
            content: vec![Content::Text { text: preview }],
            details: serde_json::json!({
                "path": path.display().to_string(),
                "applied": applied.len(),
                "edits": applied,
                "diffs": diffs,
            }),
            terminate: false,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct Edit {
    old_text: String,
    new_text: String,
}
