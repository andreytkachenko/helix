use std::{collections::HashSet, path::PathBuf, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    abort::AbortSignal,
    error::{Error, ToolError},
    message::Content,
};

pub const TOOL_READ: &str = "read";
pub const TOOL_GREP: &str = "grep";
pub const TOOL_WRITE: &str = "write";
pub const TOOL_EDIT: &str = "edit";
pub const TOOL_BASH: &str = "bash";
pub const TOOL_FIND: &str = "find";
pub const TOOL_LS: &str = "ls";

pub(crate) fn canonicalize_path<P: AsRef<std::path::Path>>(
    cwd: &std::path::Path,
    path: P,
) -> std::path::PathBuf {
    if path.as_ref().is_absolute() {
        std::path::PathBuf::from(path.as_ref())
    } else {
        cwd.join(path)
    }
}

pub(crate) fn truncate_line(line: &str, max_bytes: usize) -> String {
    if line.len() > max_bytes {
        let truncated: String = line.chars().take(max_bytes / 2).collect();
        format!("{}/{}", truncated, max_bytes)
    } else {
        line.to_string()
    }
}

/// Result produced by a tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// ToolCall Identifier
    pub id: String,

    /// Text or image content returned to the model.
    pub content: Vec<Content>,

    /// Arbitrary structured details for logs or UI rendering.
    pub details: serde_json::Value,

    /// Hint that the agent should stop after the current tool batch.
    pub terminate: bool,
}

impl ToolResult {
    pub fn error(id: String, name: String, error: Error) -> Self {
        let error = format!("{error}");

        ToolResult {
            id,
            content: vec![Content::Text {
                text: format!("tool call `{name}` failed with error: {error}"),
            }],
            details: serde_json::json!({ "success": false, "error": error }),
            terminate: false,
        }
    }
}

/// Tool execution mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

/// Trait that represents an executable tool.
#[async_trait::async_trait]
pub trait AgentTool: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::default()
    }

    /// Execute the tool. Return `Err` on failure instead of encoding errors in content.
    async fn execute(
        &self,
        id: String,
        params: serde_json::Value,
        signal: Option<AbortSignal>,
        update: &(dyn Fn(String, String) + Send + Sync),
    ) -> Result<ToolResult, ToolError>;
}

/// Trait that represents a registry of tools.
pub trait ToolRegistry: std::fmt::Debug + Send + Sync {
    fn len(&self) -> usize;
    fn get_tool(&self, name: &str) -> Option<&dyn AgentTool>;
    fn for_each(&self, cb: &mut dyn FnMut(&dyn AgentTool));
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ToolRegistry> ToolRegistry for Arc<T> {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn get_tool(&self, name: &str) -> Option<&dyn AgentTool> {
        (**self).get_tool(name)
    }

    fn for_each(&self, cb: &mut dyn FnMut(&dyn AgentTool)) {
        (**self).for_each(cb)
    }
}

mod bash;
mod edit_file;
mod find;
mod grep;
mod ls;
mod read_file;
mod write_file;

pub struct IntegratedToolConfig {
    cwd: PathBuf,
    read_max_lines: u64,
    ls_max_lines: u64,
    find_max_lines: u64,
    grep_max_lines: u64,
    fd_path: Option<PathBuf>,
    rg_path: Option<PathBuf>,
    bash_path: Option<PathBuf>,
}

impl Default for IntegratedToolConfig {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            read_max_lines: 2000,
            ls_max_lines: 500,
            find_max_lines: 1000,
            grep_max_lines: 1000,
            fd_path: None,
            rg_path: None,
            bash_path: None,
        }
    }
}

pub struct IntegratedTools {
    read: read_file::ReadFileTool,
    write: write_file::WriteFileTool,
    edit: edit_file::EditFileTool,
    grep: grep::GrepTool,
    find: find::FindTool,
    ls: ls::LsTool,
    bash: bash::BashTool,
}

impl IntegratedTools {
    pub fn new(config: IntegratedToolConfig) -> Self {
        Self {
            read: read_file::ReadFileTool::new(config.cwd.clone(), config.read_max_lines),
            write: write_file::WriteFileTool::new(config.cwd.clone()),
            edit: edit_file::EditFileTool::new(config.cwd.clone()),
            grep: grep::GrepTool::new(config.cwd.clone(), config.grep_max_lines, config.rg_path),
            find: find::FindTool::new(config.cwd.clone(), config.find_max_lines, config.fd_path),
            ls: ls::LsTool::new(config.cwd.clone(), config.ls_max_lines),
            bash: bash::BashTool::new(config.cwd, config.bash_path),
        }
    }
}

impl std::fmt::Debug for IntegratedTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IntegratedTools({})", self.len())
    }
}

impl ToolRegistry for IntegratedTools {
    fn len(&self) -> usize {
        7
    }

    fn get_tool(&self, name: &str) -> Option<&dyn AgentTool> {
        match name {
            TOOL_READ => Some(&self.read as _),
            TOOL_WRITE => Some(&self.write as _),
            TOOL_GREP => Some(&self.grep as _),
            TOOL_EDIT => Some(&self.edit as _),
            TOOL_BASH => Some(&self.bash as _),
            TOOL_FIND => Some(&self.find as _),
            TOOL_LS => Some(&self.ls as _),
            _ => None,
        }
    }

    fn for_each(&self, cb: &mut dyn FnMut(&dyn AgentTool)) {
        (cb)(&self.read);
        (cb)(&self.grep);
        (cb)(&self.write);
        (cb)(&self.edit);
        (cb)(&self.bash);
        (cb)(&self.find);
        (cb)(&self.ls);
    }
}

#[derive(Debug)]
pub struct Subset<R: ToolRegistry> {
    registry: R,
    subset: HashSet<String>,
}

impl<R: ToolRegistry> ToolRegistry for Subset<R> {
    fn len(&self) -> usize {
        self.subset.len()
    }

    fn get_tool(&self, name: &str) -> Option<&dyn AgentTool> {
        if self.subset.contains(name) {
            self.registry.get_tool(name)
        } else {
            None
        }
    }

    fn for_each(&self, cb: &mut dyn FnMut(&dyn AgentTool)) {
        for name in self.subset.iter() {
            if let Some(tool) = self.registry.get_tool(name) {
                (cb)(tool)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cwd() -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join("helix-agent-test-xxx");
        let _ = std::fs::create_dir_all(&tmp);
        tmp
    }

    #[allow(dead_code)]
    async fn execute_tool<T: AgentTool + Send + Sync>(
        tool: &T,
        id: &str,
        args: serde_json::Value,
    ) -> ToolResult {
        tool
            .execute(id.to_string(), args, None, &|_, _| {})
            .await
            .unwrap()
    }

    #[test]
    fn test_canonicalize_path_absolute() {
        let cwd = temp_cwd();
        let path = "/absolute/path";
        assert_eq!(
            canonicalize_path(&cwd, path).as_path(),
            std::path::Path::new("/absolute/path")
        );
    }

    #[test]
    fn test_canonicalize_path_relative() {
        let cwd = temp_cwd();
        let path = "relative/path";
        let result = canonicalize_path(&cwd, path);
        assert!(result.starts_with(&cwd));
        assert!(result.components().count() > cwd.components().count());
    }

    #[test]
    fn test_truncate_line_short() {
        let line = "hello";
        assert_eq!(truncate_line(line, 100), "hello");
    }

    #[test]
    fn test_truncate_line_exactly_max() {
        let line = "hello";
        assert_eq!(truncate_line(line, 5), "hello");
    }

    #[test]
    fn test_truncate_line_long() {
        let line = "a".repeat(200);
        let result = truncate_line(&line, 100);
        assert!(result.len() < 200);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error(
            "id1".to_string(),
            "bash".to_string(),
            Error::ToolNotFound("foo".to_string()),
        );
        assert_eq!(result.id, "id1");
        assert_eq!(result.terminate, false);
        assert_eq!(result.details["success"], false);
        assert_eq!(result.details["error"], "tool `foo` is not exists");
    }

    #[test]
    fn test_subset_for_each_only_subset_tools() {
        let config = IntegratedToolConfig::default();
        let registry = IntegratedTools::new(config);

        let mut subset_tools = HashSet::new();
        subset_tools.insert(TOOL_READ.to_string());
        subset_tools.insert(TOOL_WRITE.to_string());

        let subset = Subset {
            registry,
            subset: subset_tools.clone(),
        };

        let mut names = Vec::new();
        subset.for_each(&mut |tool| {
            names.push(tool.name().to_string());
        });

        assert_eq!(names.len(), 2);
        assert!(names.contains(&TOOL_READ.to_string()));
        assert!(names.contains(&TOOL_WRITE.to_string()));
    }

    #[test]
    fn test_subset_filters_tools() {
        let config = IntegratedToolConfig::default();
        let registry = Arc::new(IntegratedTools::new(config));

        let mut subset_tools = HashSet::new();
        subset_tools.insert(TOOL_READ.to_string());
        let subset = Subset {
            registry: registry.clone(),
            subset: subset_tools.clone(),
        };

        assert_eq!(subset.len(), 1);
        assert!(subset.get_tool(TOOL_READ).is_some());
        assert!(subset.get_tool(TOOL_WRITE).is_none());

        // Verify subset respects changes to subset_tools
        let mut subset2 = Subset {
            registry,
            subset: HashSet::new(),
        };
        assert_eq!(subset2.len(), 0);

        subset2.subset.insert(TOOL_BASH.to_string());
        assert_eq!(subset2.len(), 1);
        assert!(subset2.get_tool(TOOL_BASH).is_some());
    }

    #[test]
    fn test_integrated_tools_registry() {
        let config = IntegratedToolConfig::default();
        let tools = IntegratedTools::new(config);

        assert_eq!(tools.len(), 7);
        assert!(!tools.is_empty());

        // All known tools should be present
        assert!(tools.get_tool(TOOL_READ).is_some());
        assert!(tools.get_tool(TOOL_WRITE).is_some());
        assert!(tools.get_tool(TOOL_EDIT).is_some());
        assert!(tools.get_tool(TOOL_GREP).is_some());
        assert!(tools.get_tool(TOOL_BASH).is_some());
        assert!(tools.get_tool(TOOL_FIND).is_some());
        assert!(tools.get_tool(TOOL_LS).is_some());

        assert!(tools.get_tool("unknown").is_none());
    }

    #[test]
    fn test_integrated_tools_for_each_traverses_all() {
        let config = IntegratedToolConfig::default();
        let tools = IntegratedTools::new(config);

        let mut found_names = Vec::new();
        tools.for_each(&mut |tool| {
            found_names.push(tool.name().to_string());
        });

        assert_eq!(found_names.len(), 7);
        for name in [
            TOOL_READ, TOOL_WRITE, TOOL_EDIT, TOOL_GREP, TOOL_BASH, TOOL_FIND, TOOL_LS,
        ] {
            assert!(found_names.contains(&name.to_string()));
        }
    }

    #[test]
    fn test_tool_names_and_labels() {
        let config = IntegratedToolConfig::default();
        let tools = IntegratedTools::new(config);

        let expected: [(&str, &str); 7] = [
            (TOOL_READ, "Read File"),
            (TOOL_WRITE, "Write File"),
            (TOOL_EDIT, "Edit File"),
            (TOOL_GREP, "Grep"),
            (TOOL_BASH, "Bash"),
            (TOOL_FIND, "Find"),
            (TOOL_LS, "List Directory"),
        ];

        for (name, label) in expected {
            let tool = tools
                .get_tool(name)
                .expect(&format!("tool {} should exist", name));
            assert_eq!(tool.name(), name);
            assert_eq!(tool.label(), label);
            assert!(!tool.description().is_empty());
        }
    }

    #[test]
    fn test_input_schema_has_required_fields() {
        let config = IntegratedToolConfig::default();
        let tools = IntegratedTools::new(config);

        // read / write / grep / edit / ls require "path"
        for tool_name in [TOOL_READ, TOOL_WRITE, TOOL_GREP, TOOL_EDIT, TOOL_LS] {
            let tool = tools.get_tool(tool_name).unwrap();
            let schema = tool.input_schema();
            let props = schema["properties"].as_object().unwrap();
            assert!(props.contains_key("path"));
            assert!(
                schema["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|r| r == "path")
            );
        }

        // find requires "pattern", not "path" in required
        let find_tool = tools.get_tool(TOOL_FIND).unwrap();
        let schema = find_tool.input_schema();
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r == "pattern")
        );

        // bash requires "command", not "path"
        let bash = tools.get_tool(TOOL_BASH).unwrap();
        let schema = bash.input_schema();
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("command"));
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r == "command")
        );
        assert!(
            !schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r == "path")
        );
    }

    #[test]
    fn test_tool_execution_mode_default_is_parallel() {
        assert_eq!(ToolExecutionMode::default(), ToolExecutionMode::Parallel);
        let mode: ToolExecutionMode = Default::default();
        assert_eq!(mode, ToolExecutionMode::Parallel);
    }
}
