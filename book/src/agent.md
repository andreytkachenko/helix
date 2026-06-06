# Agent

Helix includes an integrated AI coding agent that can read, write, edit files, run bash commands, and search your codebase — all from within the editor.

## Quick Start

1. Configure your LLM provider (see [Configuration](#configuration) below).
2. Press `A-c` (Ctrl+A) in normal mode to open the agent panel.
3. Type your request and press `Enter` to submit.
4. The agent responds with streaming text and can use tools to modify your code.

## Usage

### Opening and Closing

| Action | Keybinding |
|--------|-----------|
| Toggle agent panel | `A-c` (Ctrl+A) |
| Close agent panel | `Escape` or `A-c` |

### Agent Mode

When the agent panel is open, Helix enters **Agent mode** (displayed as `AGE` in the status bar). Agent mode behaves like normal mode but with agent-specific capabilities:

| Action | Keybinding |
|--------|-----------|
| Enter insert mode (edit prompt) | `i` |
| Exit insert mode (back to AGE) | `Escape` |
| Submit prompt | `Enter` |
| New line in prompt | `A-Enter` (Shift+Enter) or `C-Enter` (Ctrl+Enter) |
| Close agent panel | `Escape` (when not in insert) |
| Run command | `:` |

### Mode Transitions

Agent mode integrates with Helix's mode system:

```
Normal (NOR) ──A-c──→ Agent (AGE) ──i──→ Insert (INS)
                                    ↑         │
                                    └──Escape─┘
Agent (AGE) ──Escape──→ Normal (NOR)
```

- Entering insert mode from agent mode and pressing `Escape` returns to **agent mode** (not normal mode).
- Closing the agent panel from agent mode returns to the **previous mode** (normally normal mode).

### Status Indicator

The status bar shows the agent's current activity next to the mode indicator:

| Status | Symbol | Meaning |
|--------|--------|---------|
| Idle | `○` | Agent is waiting (minimal display) |
| Thinking | `◐` | LLM is reasoning |
| Streaming | `◑` | LLM is generating text |
| Working | `●` | Agent is executing tools |

### Slash Commands

Type these in the agent prompt:

| Command | Description |
|---------|-------------|
| `/clear` | Clear the conversation history |
| `/stop` | Stop the agent's current task |
| `/model <name>` | Switch to a different model |

## Configuration

Add the following to your `config.toml`:

```toml
[agent]
# LLM model to use
model = "claude-sonnet-4-20250514"

# Provider type (currently: "openai")
provider = "openai"

# API base URL
base_url = "https://api.openai.com/v1"

# API key (can also use environment variable)
api_key = ""

# Thinking/reasoning level: off, minimal, low, medium, high, xhigh
thinking_level = "off"

# System prompt
system_prompt = "You are a helpful coding assistant. You can read, write, edit files, run bash commands, search files, and list directories."

# Tool-specific settings
[agent.tools]
# Working directory (default: current directory)
cwd = ""
# Maximum lines to read in a single file read
read_max_lines = 2000
# Maximum lines for ls output
ls_max_lines = 500
# Maximum lines for find output
find_max_lines = 1000
# Maximum lines for grep output
grep_max_lines = 1000
```

### Environment Variables

Instead of hardcoding API keys in config, use environment variables:

```bash
export HELIX_AGENT_API_KEY="your-api-key"
export HELIX_AGENT_BASE_URL="https://your-api-base-url"
```

## Available Tools

The agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `read` | Read file contents (with line offset/limit support) |
| `write` | Create or overwrite a file |
| `edit` | Apply targeted edits to a file (search and replace) |
| `grep` | Search files using ripgrep patterns |
| `find` | Find files by name pattern |
| `ls` | List directory contents |
| `bash` | Execute a bash command |

## Theme Customization

Customize the agent panel colors in your theme:

```toml
[ui.agent]
user = "ui.text"
assistant = "ui.text"
thinking = "ui.text.inactive"
tool = "ui.tooltip"
tool-running = "ui.highlight"
system = "ui.info"
border = "ui.popup"
prompt = "ui.text"

# Status bar mode color
[ui.statusline]
agent = "ui.special"
```

## Keybinding Customization

Override default agent keybindings in `config.toml`:

```toml
[keys.normal]
"A-c" = "agent-toggle"

[keys.agent]
"escape" = "agent-close"
"i" = "agent-insert-mode"
"A-Enter" = "agent-insert-newline"
"C-Enter" = "agent-insert-newline"
"Enter" = "agent-submit"
```

## Tips

- **Be specific** in your prompts. Instead of "fix the bug", say "fix the off-by-one error in the loop on line 42".
- **Use `/stop`** if the agent is doing something unexpected — it will halt at the end of the current tool execution.
- **The agent sees open files** — if a file is already open in Helix, the agent reads from memory instead of disk for faster access.
- **Changes are tracked** — files modified by the agent are marked as modified in the status bar.
