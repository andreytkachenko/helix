#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("parameter `{0}` is required, but not provided")]
    MissingParameter(&'static str),

    #[error("parameter `{0}` is not match schema: {1}")]
    ParameterWrongFormat(&'static str, serde_json::Error),

    #[error("command `{0}` is not presend in the system")]
    CommandNotFound(&'static str),

    #[error("tool `{0}` is not exists")]
    ToolNotFound(String),

    #[error("tool `{0}` execution error: {1}")]
    ToolError(String, ToolError),
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parameter `{0}` is required, but not provided")]
    MissingParameter(&'static str),

    #[error("parameter `{0}` is not match schema: {1}")]
    ParameterWrongFormat(&'static str, serde_json::Error),

    #[error("command `{0}` is not presend in the system")]
    CommandNotFound(&'static str),
}
