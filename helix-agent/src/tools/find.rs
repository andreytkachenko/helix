use std::path::PathBuf;

use tokio::process::Command;

use crate::{
    abort::AbortSignal,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolResult, truncate_line},
};

#[derive(Debug)]
pub struct FindTool {
    cwd: PathBuf,
    max_lines: u64,
    fd_path: Option<PathBuf>,
}

impl FindTool {
    pub fn new(cwd: PathBuf, max_lines: u64, fd_path: Option<PathBuf>) -> Self {
        Self {
            cwd,
            max_lines,
            fd_path,
        }
    }
}

#[async_trait::async_trait]
impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }
    fn label(&self) -> &str {
        "Find"
    }
    fn description(&self) -> &str {
        "Find files matching a glob pattern. Respects .gitignore and hidden files."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "The glob pattern to match files" },
                "path": { "type": "string", "description": "The directory to search in" }
            },
            "required": ["pattern"]
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

        let fd = self
            .fd_path
            .as_ref()
            .cloned()
            .or_else(|| find_in_path("fd"))
            .ok_or(ToolError::CommandNotFound("fd"))?;

        let output = Command::new(fd)
            .arg("--no-ignore")
            .arg("--hidden")
            .arg("--type")
            .arg("f")
            .arg(pattern)
            .arg("--max-results")
            .arg(format!("{}", self.max_lines))
            .arg(path.as_path())
            .output()
            .await?;

        let paths: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|l| truncate_line(l, 500))
            .collect();

        let count = paths.len();

        Ok(ToolResult {
            id,
            content: vec![Content::Text {
                text: if paths.is_empty() {
                    "No files found".to_string()
                } else {
                    paths.join("\n")
                },
            }],
            details: serde_json::json!({ "count": count }),
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
