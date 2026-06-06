use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,

    pub name: String,

    pub api: String,

    pub provider: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    #[serde(default)]
    pub reasoning: bool,

    #[serde(default)]
    pub input: Vec<String>,

    #[serde(default)]
    pub cost: Cost,

    #[serde(default)]
    pub context_window: u64,

    #[serde(default)]
    pub max_tokens: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub total_tokens: u64,
}
impl Usage {
    pub(crate) fn add(&mut self, usage: Usage) {
        self.input += usage.input;
        self.output += usage.output;
        self.total_tokens = self.input + self.output;
    }
}
