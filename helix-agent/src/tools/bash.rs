use std::{path::PathBuf, process::Stdio};

use futures_util::future::try_join;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{ChildStderr, ChildStdout, Command},
};

use crate::{
    abort::AbortSignal,
    error::ToolError,
    message::Content,
    tools::{AgentTool, ToolExecutionMode, ToolResult},
};

#[derive(Debug)]
pub struct BashTool {
    cwd: PathBuf,
    bash_path: Option<PathBuf>,
}

impl BashTool {
    pub fn new(cwd: PathBuf, bash_path: Option<PathBuf>) -> Self {
        Self { cwd, bash_path }
    }
}

#[async_trait::async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }
    fn label(&self) -> &str {
        "Bash"
    }
    fn description(&self) -> &str {
        "Execute a shell command in the specified working directory."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command to execute" },
                "timeout": { "type": "integer", "description": "Timeout in seconds" }
            },
            "required": ["command"]
        })
    }
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }

    async fn execute(
        &self,
        id: String,
        params: serde_json::Value,
        signal: Option<AbortSignal>,
        update: &(dyn Fn(String, String) + Send + Sync),
    ) -> Result<ToolResult, ToolError> {
        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or(ToolError::MissingParameter("command"))?;

        let timeout_secs = params
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(3600);

        let shell = self
            .bash_path
            .as_ref()
            .map(|b| b.to_string_lossy().to_string())
            .unwrap_or_else(|| "/bin/sh".to_string());

        let mut child = Command::new(&shell)
            .arg("-c")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&self.cwd)
            .spawn()?;

        let mut terminated = false;
        let mut timeout = false;

        let read_fut = read_output(
            child.stdout.take().unwrap(),
            child.stderr.take().unwrap(),
            update,
            id.clone(),
        );
        let run_fut = async {
            let status = if let Some(sig) = signal {
                tokio::select! {
                    _ = sig.wait() => {
                        terminated = true;
                        None
                    },
                    _ = tokio::time::sleep(std::time::Duration::from_secs(timeout_secs)) => {
                        timeout = true;
                        None
                    },
                    res = child.wait() => {
                        Some(res?)
                    },
                }
            } else {
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(timeout_secs)) => {
                        timeout = true;
                        None
                    },
                    res = child.wait() => {
                        Some(res?)
                    },
                }
            };

            Ok(if let Some(status) = status {
                status
            } else {
                child.start_kill()?;
                child.wait().await?
            })
        };

        let (status, output) = try_join(run_fut, read_fut).await?;
        let exit_code = status.code();
        let truncated = output.lines().count() > 2000 || output.len() > 50_000;
        let stdout_truncated = if truncated {
            let tail = output
                .lines()
                .skip(output.lines().count().saturating_sub(2000))
                .collect::<Vec<_>>()
                .join("\n");
            let head = output.lines().take(1000).collect::<Vec<_>>().join("\n");
            format!("{}...\n{}...", head, tail)
        } else {
            output
        };

        let full_output = format!(
            "{}\n\n{}{}",
            stdout_truncated,
            status,
            if timeout {
                "(killed by timeout)"
            } else if terminated {
                "(terminated by user)"
            } else {
                ""
            }
        );

        Ok(ToolResult {
            id,
            content: vec![Content::Text { text: full_output }],
            details: serde_json::json!({
                "exit_code": exit_code,
                "command": command,
                "timeout_seconds": timeout_secs,
            }),
            terminate: false,
        })
    }
}

pub async fn read_output<F: Fn(String, String) + Send + Sync + Clone>(
    mut stdout: ChildStdout,
    mut stderr: ChildStderr,
    update: F,
    id: String,
) -> std::io::Result<String> {
    let mut buffer = String::new();
    let mut stdout_lines = BufReader::new(&mut stdout).lines();
    let mut stderr_lines = BufReader::new(&mut stderr).lines();

    loop {
        tokio::select! {
            maybe = stdout_lines.next_line() => {
                match maybe {
                    Ok(Some(line)) => {
                        buffer.push_str(&line);
                        buffer.push('\n');
                        (update)(id.clone(), line);
                    }
                    Ok(None) => {
                        // stdout closed; drain remaining stderr then break
                        while let Some(x) = stderr_lines.next_line().await.transpose() {
                            let line = x?;
                            buffer.push_str(&line);
                            buffer.push('\n');
                            (update)(id.clone(), line);
                        }
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            maybe = stderr_lines.next_line() => {
                match maybe {
                    Ok(Some(line)) => {
                        buffer.push_str(&line);
                        buffer.push('\n');
                        (update)(id.clone(), line);
                    }
                    Ok(None) => {
                        // stderr closed; drain remaining stdout then break
                        while let Some(x) = stdout_lines.next_line().await.transpose() {
                            let line = x?;
                            buffer.push_str(&line);
                            buffer.push('\n');
                            (update)(id.clone(), line);
                        }
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    Ok(buffer)
}
