use crate::compositor::{Component, Context, Event, EventResult};
use helix_view::{
    graphics::Rect,
    input::{MouseEvent, MouseEventKind},
    keyboard::{KeyCode, KeyModifiers},
    Editor,
};

use tui::buffer::Buffer as Surface;
use tui::layout::Alignment;
use tui::text::{Span, Spans, Text};
use tui::widgets::{Paragraph, Widget, Wrap};

/// Maximum width for tool result preview lines before truncation.
const TOOL_RESULT_MAX_LINE_WIDTH: usize = 120;
/// Maximum lines for tool result preview.
const TOOL_RESULT_MAX_LINES: usize = 5;

pub struct AgentOutput {
    /// Scroll offset for the message history.
    scroll: usize,
    /// Whether we're tracking the bottom (auto-scroll).
    at_bottom: bool,
}

impl AgentOutput {
    pub fn new() -> Self {
        Self {
            scroll: 0,
            at_bottom: true,
        }
    }

    /// Build the full text content for the output area.
    fn build_text(&self, editor: &Editor) -> Text<'_> {
        let session = &editor.agent_session;
        let messages = session.messages.read().unwrap();
        let theme = &editor.theme;
        let mut lines: Vec<Spans<'_>> = Vec::new();

        // Show placeholder if no messages
        if messages.is_empty() {
            let hint = Span::styled("Agent — type a request and press Enter", theme.get("ui.text.inactive"));
            lines.push(Spans::from(hint));
        }

        for msg in messages.iter() {
            match msg.role {
                helix_agent::message::Role::System => {
                    // Skip system messages in UI (too verbose)
                }
                helix_agent::message::Role::User => {
                    for content in &msg.content {
                        let text = content.to_string();
                        for line in text.lines() {
                            lines.push(Spans::from(Span::styled(
                                format!("▸ {}", line),
                                theme.get("ui.agent.user"),
                            )));
                        }
                    }
                }
                helix_agent::message::Role::Assistant => {
                    for content in &msg.content {
                        match content {
                            helix_agent::message::Content::Text { text } => {
                                for line in text.lines() {
                                    if line.is_empty() {
                                        lines.push(Spans::from(Span::raw("")));
                                    } else {
                                        lines.push(Spans::from(Span::styled(
                                            line.to_string(),
                                            theme.get("ui.agent.assistant"),
                                        )));
                                    }
                                }
                            }
                            helix_agent::message::Content::Thinking { thinking } => {
                                for (idx, line) in thinking.lines().enumerate() {
                                    let display =  if idx == 0 {
                                        format!("… {}", line)                                        
                                    } else {
                                        line.to_string()
                                    };

                                    lines.push(Spans::from(Span::styled(
                                        display,
                                        theme.get("ui.agent.thinking"),
                                    )));
                                }
                            }
                            helix_agent::message::Content::ToolCall {
                                id: _id,
                                name,
                                arguments,
                            } => {
                                // Compact tool call format: "Read /path" or "$ command"
                                let call_text = Self::format_tool_call(&name[..], arguments);
                                lines.push(Spans::from(Span::styled(
                                    call_text,
                                    theme.get("ui.agent.tool"),
                                )));
                            }
                            helix_agent::message::Content::Image { .. } => {
                                lines.push(Spans::from(Span::styled(
                                    "[Image]",
                                    theme.get("ui.text"),
                                )));
                            }
                        }
                    }
                }
                helix_agent::message::Role::Tool => {
                    for content in &msg.content {
                        let text = content.to_string();
                        // Compact tool result: show only first few lines
                        let preview = Self::truncate_tool_result(&text);
                        for line in preview.lines() {
                            lines.push(Spans::from(Span::styled(
                                format!("  {}", line),
                                theme.get("ui.text.inactive"),
                            )));
                        }
                    }
                }
            }
        }

        // ── Streaming text (live) ──────────────────────────────────────────
        if let Some(ref text) = *session.assistant_text.read().unwrap() {
            if !text.is_empty() {
                let assistant_style = theme.get("ui.agent.assistant");
                let cursor = Span::styled("▌", theme.get("ui.visual"));
                let text_lines: Vec<&str> = text.lines().collect();
                let total = text_lines.len();
                for (i, line) in text_lines.iter().enumerate() {
                    let is_last = i == total - 1 && !text.ends_with('\n');
                    if line.is_empty() {
                        lines.push(Spans::from(Span::raw("")));
                    } else if is_last {
                        lines.push(Spans::from(vec![
                            Span::styled(line.to_string(), assistant_style),
                            cursor.clone(),
                        ]));
                    } else {
                        lines.push(Spans::from(Span::styled(
                            line.to_string(),
                            assistant_style,
                        )));
                    }
                }
            }
        }

        // ── Streaming thinking (live) ──────────────────────────────────────
        if let Some(ref thinking) = *session.assistant_thinking.read().unwrap() {
            if !thinking.is_empty() {
                let thinking_style = theme.get("ui.agent.thinking");
                let cursor = Span::styled("▌", theme.get("ui.visual"));
                let thinking_lines: Vec<&str> = thinking.lines().collect();
                let total = thinking_lines.len();
                for (i, line) in thinking_lines.iter().enumerate() {
                    let is_last = i == total - 1 && !thinking.ends_with('\n');
                    let display = if i == 0 {
                        format!("… {}", line)
                    } else {
                        line.to_string()
                    };
                    if is_last {
                        lines.push(Spans::from(vec![
                            Span::styled(display, thinking_style),
                            cursor.clone(),
                        ]));
                    } else {
                        lines.push(Spans::from(Span::styled(display, thinking_style)));
                    }
                }
            }
        }

        // ── Active tool indicators ─────────────────────────────────────────
        for tool in session.active_tools.read().unwrap().iter() {
            let call_text = Self::format_tool_call(&tool.name[..], &tool.args);
            lines.push(Spans::from(Span::styled(
                format!("◐ {}", call_text),
                theme.get("ui.agent.tool_running"),
            )));
        }

        // Ensure at least one trailing empty line
        if lines.is_empty() || !lines.last().map(|s| s.0.is_empty()).unwrap_or(true) {
            lines.push(Spans::from(Span::raw("")));
        }

        Text::from(lines)
    }

    /// Format tool call into compact log format.
    /// Examples: "Read /path/to/file", "Write /path", "Edit /path", "$ ls -la"
    fn format_tool_call(name: &str, args: &serde_json::Value) -> String {
        match name {
            "read" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("Read {}", path)
                } else {
                    "Read".to_string()
                }
            }
            "write" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("Write {}", path)
                } else {
                    "Write".to_string()
                }
            }
            "edit" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("Edit {}", path)
                } else {
                    "Edit".to_string()
                }
            }
            "grep" => {
                let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                format!("Grep {} in {}", pattern, path)
            }
            "find" => {
                if let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) {
                    format!("Find {}", pattern)
                } else {
                    "Find".to_string()
                }
            }
            "ls" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("Ls {}", path)
                } else {
                    "Ls".to_string()
                }
            }
            "bash" => {
                if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                    format!("$ {}", cmd)
                } else {
                    "$ bash".to_string()
                }
            }
            _ => format!("{} {}", name, Self::short_json(args)),
        }
    }

    /// Truncate a tool result to a compact preview.
    fn truncate_tool_result(text: &str) -> String {
        let mut result = String::new();
        let mut line_count = 0;
        for line in text.lines() {
            if line_count >= TOOL_RESULT_MAX_LINES {
                if line_count == TOOL_RESULT_MAX_LINES {
                    result.push_str("  … (truncated)");
                }
                break;
            }
            let truncated = if line.len() > TOOL_RESULT_MAX_LINE_WIDTH {
                &line[..TOOL_RESULT_MAX_LINE_WIDTH]
            } else {
                line
            };
            result.push_str(truncated);
            result.push('\n');
            line_count += 1;
        }
        result.trim_end().to_string()
    }

    /// Short JSON representation for tool args display.
    fn short_json(value: &serde_json::Value) -> String {
        let json = serde_json::to_string(value).unwrap_or_default();
        if json.len() > 60 {
            format!("{}…", &json[..57])
        } else {
            json
        }
    }
}

impl Component for AgentOutput {
    fn handle_event(&mut self, event: &Event, _ctx: &mut Context) -> EventResult {
        match event {
            Event::Key(key_event) => {
                let scroll = &mut self.scroll;
                let at_bottom = &mut self.at_bottom;
                match key_event.code {
                    KeyCode::Esc => EventResult::Ignored(None),
                    KeyCode::Up | KeyCode::Char('k') => {
                        *scroll = scroll.saturating_add(1);
                        *at_bottom = false;
                        EventResult::Consumed(None)
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if *scroll > 0 {
                            *scroll = scroll.saturating_sub(1);
                        }
                        *at_bottom = *scroll == 0;
                        EventResult::Consumed(None)
                    }
                    KeyCode::PageUp => {
                        *scroll = scroll.saturating_add(10).min(99999);
                        *at_bottom = false;
                        EventResult::Consumed(None)
                    }
                    KeyCode::PageDown => {
                        if *scroll > 0 {
                            *scroll = scroll.saturating_sub(10);
                        }
                        *at_bottom = *scroll == 0;
                        EventResult::Consumed(None)
                    }
                    KeyCode::Char('g') if key_event.modifiers == KeyModifiers::NONE => {
                        // Jump to top (max scroll = show first lines)
                        *scroll = usize::MAX;
                        *at_bottom = false;
                        EventResult::Consumed(None)
                    }
                    KeyCode::Char('G') => {
                        // Jump to bottom (scroll = 0, auto-scroll)
                        *scroll = 0;
                        *at_bottom = true;
                        EventResult::Consumed(None)
                    }
                    _ => EventResult::Ignored(None),
                }
            }
            Event::Mouse(mouse_event) => {
                let scroll = &mut self.scroll;
                let at_bottom = &mut self.at_bottom;
                match mouse_event {
                    MouseEvent {
                        kind: MouseEventKind::ScrollUp,
                        ..
                    } => {
                        *scroll = scroll.saturating_add(1);
                        *at_bottom = false;
                        EventResult::Consumed(None)
                    }
                    MouseEvent {
                        kind: MouseEventKind::ScrollDown,
                        ..
                    } => {
                        if *scroll > 0 {
                            *scroll = scroll.saturating_sub(1);
                        }
                        *at_bottom = *scroll == 0;
                        EventResult::Consumed(None)
                    }
                    _ => EventResult::Ignored(None),
                }
            }
            Event::Paste(_) => EventResult::Ignored(None),
            _ => EventResult::Ignored(None),
        }
    }

    fn render(&mut self, area: Rect, frame: &mut Surface, ctx: &mut Context) {
        let theme = &ctx.editor.theme;

        // Clear the entire output area with background
        frame.clear_with(area, theme.get("ui.popup"));

        let text = self.build_text(ctx.editor);

        let visible_lines = area.height as usize;
        let total_lines = text.lines.len();

        // Compute effective scroll: at_bottom = scroll to bottom, otherwise use stored offset
        // Max valid scroll = total_lines - visible_lines (shows first lines at top)
        let max_scroll = total_lines.saturating_sub(visible_lines);
        let scroll = if self.at_bottom {
            max_scroll
        } else {
            self.scroll.min(max_scroll)
        };

        let paragraph = Paragraph::new(&text)
            .scroll((scroll as u16, 0))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });

        paragraph.render(area, frame);
    }

    fn cursor(&self, _area: Rect, _ctx: &Editor) -> (Option<helix_core::Position>, helix_view::graphics::CursorKind) {
        (None, helix_view::graphics::CursorKind::Hidden)
    }

    fn required_size(&mut self, _viewport: (u16, u16)) -> Option<(u16, u16)> {
        None
    }
}
