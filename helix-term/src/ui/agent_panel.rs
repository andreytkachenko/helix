use crate::compositor::{Component, Context, Event, EventResult};
use helix_view::{
    graphics::{CursorKind, Rect},
    input::KeyCode,
    Editor,
};
use helix_core::Position;
use tui::buffer::Buffer as Surface;

use super::agent_output::AgentOutput;
use super::prompt::Prompt;

/// Component ID for finding the agent panel in the compositor.
pub const AGENT_PANEL_ID: &str = "agent_panel";

/// Maximum visible rows for the multiline prompt.
const MAX_PROMPT_ROWS: usize = 5;

/// A composite component: AgentOutput (top) + Prompt (bottom, replacing statusline).
pub struct AgentPanel {
    output: AgentOutput,
    prompt: Prompt,
}

impl AgentPanel {
    pub fn new() -> Self {
        // Create a Prompt with multiline support, history, and statusline background.
        // Register 'a' is used for agent prompt history.
        let prompt = Prompt::new(
            "> ".into(),
            Some('a'), // agent prompt history register
            |_editor, _line| Vec::new(), // no completions
            |_cx, _line, _event| {},     // no-op callback (we handle submit ourselves)
        )
        .with_multiline(MAX_PROMPT_ROWS)
        .with_background("ui.statusline");

        Self {
            output: AgentOutput::new(),
            prompt,
        }
    }

    /// Get the current prompt input text.
    pub fn prompt_input(&self) -> &str {
        self.prompt.line()
    }

    /// Take the prompt input and clear it.
    pub fn take_prompt_input(&mut self, ctx: &mut Context) -> String {
        let input = self.prompt.line().clone();
        self.prompt.clear(ctx.editor);
        input
    }
}

impl Component for AgentPanel {
    fn handle_event(&mut self, event: &Event, ctx: &mut Context) -> EventResult {
        match event {
            Event::Key(key) => {
                // Let plain Enter pass through to command handler (agent_submit)
                if key.code == KeyCode::Enter
                    && !key.modifiers.contains(helix_view::input::KeyModifiers::SHIFT)
                    && !key.modifiers.contains(helix_view::input::KeyModifiers::CONTROL)
                {
                    return EventResult::Ignored(None);
                }

                // Let Escape pass through to command handler (agent_hide)
                if key.code == KeyCode::Esc {
                    return EventResult::Ignored(None);
                }

                // Let Ctrl+C pass through to command handler (agent_stop)
                if key.code == KeyCode::Char('c')
                    && key.modifiers == helix_view::input::KeyModifiers::CONTROL
                {
                    return EventResult::Ignored(None);
                }

                // PgUp/PgDn/Ctrl+Y/Ctrl+E/g/G scroll the output area (not the prompt)
                if key.code == KeyCode::PageUp
                    || key.code == KeyCode::PageDown
                    || (key.code == KeyCode::Char('y') && key.modifiers == helix_view::input::KeyModifiers::CONTROL)
                    || (key.code == KeyCode::Char('d') && key.modifiers == helix_view::input::KeyModifiers::CONTROL)
                    || key.code == KeyCode::Char('g')
                    || key.code == KeyCode::Char('G')
                {
                    // Route scroll keys to output
                    return self.output.handle_event(event, ctx);
                }

                // All other keys go to the prompt
                self.prompt.handle_event(event, ctx)
            }
            Event::Mouse(_) => {
                // Mouse events go to output for scroll wheel
                self.output.handle_event(event, ctx)
            }
            Event::Paste(_) => {
                self.prompt.handle_event(event, ctx)
            }
            _ => EventResult::Ignored(None),
        }
    }

    fn render(&mut self, area: Rect, frame: &mut Surface, ctx: &mut Context) {
        // Check if prompt should be cleared (after submit)
        if ctx.editor
            .agent_session
            .prompt_clear_requested
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.prompt.clear(ctx.editor);
        }

        // Sync prompt <-> session bidirectionally.
        // - prompt -> session: keeps session mirror up to date for agent_submit
        // - session -> prompt: picks up pre-filled text from agent_ask_at_cursor
        {
            let prompt_input = self.prompt.line().clone();
            let mut session = ctx.editor.agent_session.prompt_input.write().unwrap();
            if *session != prompt_input {
                // If session has content but prompt is empty, session was pre-filled externally
                // (e.g., by agent_ask_at_cursor). Copy session -> prompt.
                if !session.is_empty() && prompt_input.is_empty() {
                    self.prompt.set_line(session.clone(), ctx.editor);
                } else {
                    // Otherwise prompt changed (user typing). Copy prompt -> session.
                    *session = prompt_input;
                }
            }
        }

        // Calculate prompt height based on visible content
        let prompt_height = if self.prompt.line().is_empty() {
            1
        } else {
            let total_lines = self.prompt.line().matches('\n').count() + 1;
            (total_lines.min(MAX_PROMPT_ROWS)).max(1) as u16
        };

        // Split: output takes top, prompt sits at bottom (replacing statusline)
        let output_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: area.height.saturating_sub(prompt_height),
        };
        let prompt_area = Rect {
            x: area.x,
            y: area.y + area.height.saturating_sub(prompt_height),
            width: area.width,
            height: prompt_height,
        };

        // Render output behind
        self.output.render(output_area, frame, ctx);

        // Render prompt on top (opaque, hides statusline)
        self.prompt.render(prompt_area, frame, ctx);
    }

    fn required_size(&mut self, viewport: (u16, u16)) -> Option<(u16, u16)> {
        Some(viewport)
    }

    fn cursor(&self, area: Rect, ctx: &Editor) -> (Option<Position>, CursorKind) {
        let prompt_height = if self.prompt.line().is_empty() {
            1
        } else {
            let total_lines = self.prompt.line().matches('\n').count() + 1;
            (total_lines.min(MAX_PROMPT_ROWS)).max(1) as u16
        };

        let prompt_area = Rect {
            x: area.x,
            y: area.y + area.height.saturating_sub(prompt_height),
            width: area.width,
            height: prompt_height,
        };
        self.prompt.cursor(prompt_area, ctx)
    }

    fn id(&self) -> Option<&'static str> {
        Some(AGENT_PANEL_ID)
    }
}

/// Create an AgentPanel wrapped in an Overlay.
/// Panel is full width, 1/3 of screen height, positioned at the bottom.
pub fn agent_panel() -> super::overlay::Overlay<AgentPanel> {
    super::overlay::Overlay {
        content: AgentPanel::new(),
        calc_child_size: Box::new(|rect: Rect| {
            // Full width, 1/3 height at bottom (above commandline)
            let height = (rect.height as f64 * 1.0 / 3.0).ceil() as u16;
            let y = rect.y + rect.height.saturating_sub(height); // +1 for commandline
            Rect {
                x: rect.x,
                y,
                width: rect.width,
                height,
            }
        }),
    }
}

impl From<AgentPanel> for Box<dyn Component> {
    fn from(panel: AgentPanel) -> Self {
        Box::new(panel)
    }
}
