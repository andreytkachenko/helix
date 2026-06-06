use std::path::PathBuf;

use tokio::process::Command;

use crate::{
    abort::AbortSignal,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolResult, truncate_line},
};

#[derive(Debug)]
pub struct GrepTool {
    cwd: PathBuf,
    max_lines: u64,
    rg_path: Option<PathBuf>,
}

impl GrepTool {
    pub fn new(cwd: PathBuf, max_lines: u64, rg_path: Option<PathBuf>) -> Self {
        Self {
            cwd,
            max_lines,
            rg_path,
        }
    }
}

#[async_trait::async_trait]
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }
    fn label(&self) -> &str {
        "Grep"
    }
    fn description(&self) -> &str {
        "Search for a pattern in a file or directory using ripgrep. Respects .gitignore."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The file or directory to search in" },
                "pattern": { "type": "string", "description": "The file pattern to search" }
            },
            "required": ["path", "pattern"]
        })
    }

    async fn execute(
        &self,
        id: String,
        params: serde_json::Value,
        _signal: Option<AbortSignal>,
        _update: &(dyn Fn(String, String) + Send + Sync),
    ) -> Result<ToolResult, ToolError> {
        let pattern = params
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or(ToolError::MissingParameter("pattern"))?;

        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| self.cwd.clone());

        let rg = self
            .rg_path
            .as_ref()
            .cloned()
            .or_else(|| find_in_path("rg"))
            .ok_or(ToolError::CommandNotFound("rg"))?;

        let output = Command::new(rg)
            .arg("--line-number")
            .arg("--column")
            .arg("--context")
            .arg("1")
            .arg("--smart-case")
            .arg("-n")
            .arg("--json")
            .arg(pattern)
            .arg(path.as_path())
            .output()
            .await?;

        let text_output = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<String> = text_output.lines().map(|l| truncate_line(l, self.max_lines as usize)).collect();

        Ok(ToolResult {
            id,
            content: vec![Content::Text {
                text: if lines.is_empty() {
                    "No matches found".to_string()
                } else {
                    lines.join("\n")
                },
            }],
            details: serde_json::json!({
                "matches": lines.len(),
                "path": path.display().to_string(),
            }),
            terminate: false,
        })
    }
}

/// Find a command in PATH without using the `which` crate.
fn find_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path).find_map(|dir| {
            let candidate = dir.join(name);
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}
