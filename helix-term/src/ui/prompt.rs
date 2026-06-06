use crate::compositor::{Component, Compositor, Context, Event, EventResult};
use crate::{alt, ctrl, key, shift, ui};
use arc_swap::ArcSwap;
use helix_core::syntax;
use helix_view::document::Mode;
use helix_view::input::KeyEvent;
use helix_view::keyboard::{KeyCode, KeyModifiers};
use std::sync::Arc;
use std::{borrow::Cow, ops::RangeFrom};
use tui::buffer::Buffer as Surface;
use tui::text::Span;
use tui::widgets::{Block, Widget};

use helix_core::{
    unicode::segmentation::{GraphemeCursor, UnicodeSegmentation},
    unicode::width::{UnicodeWidthStr, UnicodeWidthChar},
    Position,
};
use helix_view::{
    graphics::{CursorKind, Margin, Rect},
    Editor,
};

type PromptCharHandler = Box<dyn Fn(&mut Prompt, char, &Context)>;

pub type Completion = (RangeFrom<usize>, Span<'static>);
type CompletionFn = Box<dyn FnMut(&Editor, &str) -> Vec<Completion>>;
type CallbackFn = Box<dyn FnMut(&mut Context, &str, PromptEvent)>;
pub type DocFn = Box<dyn Fn(&str) -> Option<Cow<str>>>;

pub struct Prompt {
    prompt: Cow<'static, str>,
    line: String,
    cursor: usize,
    // Fields used for Component callbacks and rendering:
    line_area: Rect,
    anchor: usize,
    truncate_start: bool,
    truncate_end: bool,
    // Multiline support
    /// When true, allows newlines in the input and renders multiple rows.
    multiline: bool,
    /// Maximum number of visible rows for multiline input. Input can grow beyond this;
    /// only the last `max_rows` lines are shown (scroll-to-bottom).
    max_rows: usize,
    /// Optional custom background theme key for the prompt area.
    /// When set, the prompt uses this theme key instead of "ui.background".
    background_key: Option<&'static str>,
    // ---
    completion: Vec<Completion>,
    selection: Option<usize>,
    history_register: Option<char>,
    history_pos: Option<usize>,
    completion_fn: CompletionFn,
    callback_fn: CallbackFn,
    pub doc_fn: DocFn,
    next_char_handler: Option<PromptCharHandler>,
    language: Option<(&'static str, Arc<ArcSwap<syntax::Loader>>)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PromptEvent {
    /// The prompt input has been updated.
    Update,
    /// Validate and finalize the change.
    Validate,
    /// Abort the change, reverting to the initial state.
    Abort,
}

pub enum CompletionDirection {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy)]
pub enum Movement {
    BackwardChar(usize),
    BackwardWord(usize),
    ForwardChar(usize),
    ForwardWord(usize),
    /// Start of the current line (in multiline) or start of text (single line).
    StartOfLine,
    /// End of the current line (in multiline) or end of text (single line).
    EndOfLine,
    /// Start of entire text (multiline only).
    Start,
    /// End of entire text (multiline only).
    End,
    /// Move up one line at same column (multiline only).
    Up,
    /// Move down one line at same column (multiline only).
    Down,
    None,
}

fn is_word_sep(c: char) -> bool {
    c == std::path::MAIN_SEPARATOR || c.is_whitespace()
}

impl Prompt {
    pub fn new(
        prompt: Cow<'static, str>,
        history_register: Option<char>,
        completion_fn: impl FnMut(&Editor, &str) -> Vec<Completion> + 'static,
        callback_fn: impl FnMut(&mut Context, &str, PromptEvent) + 'static,
    ) -> Self {
        Self {
            prompt,
            line: String::new(),
            cursor: 0,
            line_area: Rect::default(),
            anchor: 0,
            truncate_start: false,
            truncate_end: false,
            multiline: false,
            max_rows: 1,
            background_key: None,
            completion: Vec::new(),
            selection: None,
            history_register,
            history_pos: None,
            completion_fn: Box::new(completion_fn),
            callback_fn: Box::new(callback_fn),
            doc_fn: Box::new(|_| None),
            next_char_handler: None,
            language: None,
        }
    }

    /// Enable multiline input with a maximum number of visible rows.
    /// Input can grow beyond max_rows; only the last max_rows lines are visible.
    pub fn with_multiline(mut self, max_rows: usize) -> Self {
        self.multiline = true;
        self.max_rows = max_rows.max(1);
        self
    }

    /// Set a custom background theme key for the prompt area.
    /// When set, this overrides the default "ui.background".
    /// Common keys: "ui.statusline", "ui.popup"
    pub fn with_background(mut self, key: &'static str) -> Self {
        self.background_key = Some(key);
        self
    }

    /// Gets the byte index in the input representing the current cursor location.
    #[inline]
    pub(crate) fn position(&self) -> usize {
        self.cursor
    }

    pub fn with_line(mut self, line: String, editor: &Editor) -> Self {
        self.set_line(line, editor);
        self
    }

    pub fn set_line(&mut self, line: String, editor: &Editor) {
        let cursor = line.len();
        self.line = line;
        self.cursor = cursor;
        self.recalculate_completion(editor);
    }

    pub fn with_language(
        mut self,
        language: &'static str,
        loader: Arc<ArcSwap<syntax::Loader>>,
    ) -> Self {
        self.language = Some((language, loader));
        self
    }

    pub fn line(&self) -> &String {
        &self.line
    }

    pub fn with_history_register(&mut self, history_register: Option<char>) -> &mut Self {
        self.history_register = history_register;
        self
    }

    pub(crate) fn history_register(&self) -> Option<char> {
        self.history_register
    }

    pub(crate) fn first_history_completion<'a>(
        &'a self,
        editor: &'a Editor,
    ) -> Option<Cow<'a, str>> {
        self.history_register
            .and_then(|reg| editor.registers.first(reg, editor))
    }

    pub fn recalculate_completion(&mut self, editor: &Editor) {
        self.exit_selection();
        self.completion = (self.completion_fn)(editor, &self.line);
    }

    /// Compute the cursor position after applying movement
    /// Taken from: <https://github.com/wez/wezterm/blob/e0b62d07ca9bf8ce69a61e30a3c20e7abc48ce7e/termwiz/src/lineedit/mod.rs#L516-L611>
    fn eval_movement(&self, movement: Movement) -> usize {
        match movement {
            Movement::BackwardChar(rep) => {
                let mut position = self.cursor;
                for _ in 0..rep {
                    let mut cursor = GraphemeCursor::new(position, self.line.len(), false);
                    if let Ok(Some(pos)) = cursor.prev_boundary(&self.line, 0) {
                        position = pos;
                    } else {
                        break;
                    }
                }
                position
            }
            Movement::BackwardWord(rep) => {
                let char_indices: Vec<(usize, char)> = self.line.char_indices().collect();
                if char_indices.is_empty() {
                    return self.cursor;
                }
                let mut char_position = char_indices
                    .iter()
                    .position(|(idx, _)| *idx == self.cursor)
                    .unwrap_or(char_indices.len() - 1);

                for _ in 0..rep {
                    if char_position == 0 {
                        break;
                    }

                    let mut found = None;
                    for prev in (0..char_position - 1).rev() {
                        if is_word_sep(char_indices[prev].1) {
                            found = Some(prev + 1);
                            break;
                        }
                    }

                    char_position = found.unwrap_or(0);
                }
                char_indices[char_position].0
            }
            Movement::ForwardWord(rep) => {
                let char_indices: Vec<(usize, char)> = self.line.char_indices().collect();
                if char_indices.is_empty() {
                    return self.cursor;
                }
                let mut char_position = char_indices
                    .iter()
                    .position(|(idx, _)| *idx == self.cursor)
                    .unwrap_or(char_indices.len());

                for _ in 0..rep {
                    // Skip any non-whitespace characters
                    while char_position < char_indices.len()
                        && !is_word_sep(char_indices[char_position].1)
                    {
                        char_position += 1;
                    }

                    // Skip any whitespace characters
                    while char_position < char_indices.len()
                        && is_word_sep(char_indices[char_position].1)
                    {
                        char_position += 1;
                    }

                    // We are now on the start of the next word
                }
                char_indices
                    .get(char_position)
                    .map(|(i, _)| *i)
                    .unwrap_or_else(|| self.line.len())
            }
            Movement::ForwardChar(rep) => {
                let mut position = self.cursor;
                for _ in 0..rep {
                    let mut cursor = GraphemeCursor::new(position, self.line.len(), false);
                    if let Ok(Some(pos)) = cursor.next_boundary(&self.line, 0) {
                        position = pos;
                    } else {
                        break;
                    }
                }
                position
            }
            Movement::StartOfLine => {
                // Start of current line
                let before = &self.line[..self.cursor.min(self.line.len())];
                if let Some(last_nl) = before.rfind('\n') {
                    last_nl + 1
                } else {
                    0
                }
            }
            Movement::EndOfLine => {
                // End of current line
                let after = &self.line[self.cursor.min(self.line.len())..];
                if let Some(next_nl) = after.find('\n') {
                    self.cursor + next_nl
                } else {
                    self.line.len()
                }
            }
            Movement::Start => 0,
            Movement::End => self.line.len(),
            Movement::Up => {
                // Move to previous line at same column
                let before = &self.line[..self.cursor.min(self.line.len())];
                if let Some(last_nl) = before.rfind('\n') {
                    // Column on current line
                    let col = self.cursor - last_nl - 1;
                    // Start of line above
                    let above_start = if last_nl > 0 {
                        if let Some(pprev_nl) = before[..last_nl].rfind('\n') {
                            pprev_nl + 1
                        } else {
                            0
                        }
                    } else {
                        return 0;
                    };
                    let above_line_len = last_nl - above_start;
                    above_start + col.min(above_line_len)
                } else {
                    // Already on first line
                    0
                }
            }
            Movement::Down => {
                // Move to next line at same column
                let after = &self.line[self.cursor.min(self.line.len())..];
                if let Some(next_nl) = after.find('\n') {
                    // Get column on current line
                    let current_line_start = if let Some(last_nl) = self.line[..self.cursor].rfind('\n') {
                        last_nl + 1
                    } else {
                        0
                    };
                    let col = self.cursor - current_line_start;
                    // Find the line below
                    let below_start = self.cursor + next_nl + 1;
                    let below_end = if let Some(next_next_nl) = after[next_nl + 1..].find('\n') {
                        self.cursor + next_nl + 1 + next_next_nl
                    } else {
                        self.line.len()
                    };
                    let below_line_len = below_end - below_start;
                    below_start + col.min(below_line_len)
                } else {
                    // Already on last line
                    self.cursor
                }
            }
            Movement::None => self.cursor,
        }
    }

    pub fn insert_char(&mut self, c: char, cx: &Context) {
        if let Some(handler) = &self.next_char_handler.take() {
            handler(self, c, cx);

            self.next_char_handler = None;
            return;
        }

        self.line.insert(self.cursor, c);
        let mut cursor = GraphemeCursor::new(self.cursor, self.line.len(), false);
        if let Ok(Some(pos)) = cursor.next_boundary(&self.line, 0) {
            self.cursor = pos;
        }
        self.recalculate_completion(cx.editor);
    }

    pub fn insert_str(&mut self, s: &str, editor: &Editor) {
        self.line.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.recalculate_completion(editor);
    }

    pub fn move_cursor(&mut self, movement: Movement) {
        let pos = self.eval_movement(movement);
        self.cursor = pos
    }

    pub fn move_start(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.line.len();
    }

    pub fn delete_char_backwards(&mut self, editor: &Editor) {
        let pos = self.eval_movement(Movement::BackwardChar(1));
        self.line.replace_range(pos..self.cursor, "");
        self.cursor = pos;

        self.recalculate_completion(editor);
    }

    pub fn delete_char_forwards(&mut self, editor: &Editor) {
        let pos = self.eval_movement(Movement::ForwardChar(1));
        self.line.replace_range(self.cursor..pos, "");

        self.recalculate_completion(editor);
    }

    pub fn delete_word_backwards(&mut self, editor: &Editor) {
        let pos = self.eval_movement(Movement::BackwardWord(1));
        self.line.replace_range(pos..self.cursor, "");
        self.cursor = pos;

        self.recalculate_completion(editor);
    }

    pub fn delete_word_forwards(&mut self, editor: &Editor) {
        let pos = self.eval_movement(Movement::ForwardWord(1));
        self.line.replace_range(self.cursor..pos, "");

        self.recalculate_completion(editor);
    }

    pub fn kill_to_start_of_line(&mut self, editor: &Editor) {
        let pos = self.eval_movement(Movement::StartOfLine);
        self.line.replace_range(pos..self.cursor, "");
        self.cursor = pos;

        self.recalculate_completion(editor);
    }

    pub fn kill_to_end_of_line(&mut self, editor: &Editor) {
        let pos = self.eval_movement(Movement::EndOfLine);
        self.line.replace_range(self.cursor..pos, "");

        self.recalculate_completion(editor);
    }

    pub fn clear(&mut self, editor: &Editor) {
        self.line.clear();
        self.cursor = 0;
        self.recalculate_completion(editor);
    }

    pub fn change_history(
        &mut self,
        cx: &mut Context,
        register: char,
        direction: CompletionDirection,
    ) {
        (self.callback_fn)(cx, &self.line, PromptEvent::Abort);
        let mut values = match cx.editor.registers.read(register, cx.editor) {
            Some(values) if values.len() > 0 => values.rev(),
            _ => return,
        };

        let end = values.len().saturating_sub(1);

        let index = match direction {
            CompletionDirection::Forward => self.history_pos.map_or(0, |i| i + 1),
            CompletionDirection::Backward => self
                .history_pos
                .unwrap_or_else(|| values.len())
                .saturating_sub(1),
        }
        .min(end);

        self.line = values.nth(index).unwrap().to_string();
        // Appease the borrow checker.
        drop(values);

        self.history_pos = Some(index);

        self.move_end();
        (self.callback_fn)(cx, &self.line, PromptEvent::Update);
        self.recalculate_completion(cx.editor);
    }

    pub fn change_completion_selection(&mut self, direction: CompletionDirection) {
        if self.completion.is_empty() {
            return;
        }

        let index = match direction {
            CompletionDirection::Forward => self.selection.map_or(0, |i| i + 1),
            CompletionDirection::Backward => {
                self.selection.unwrap_or(0) + self.completion.len() - 1
            }
        } % self.completion.len();

        self.selection = Some(index);

        let (range, item) = &self.completion[index];

        self.line.replace_range(range.clone(), &item.content);

        self.move_end();
    }

    pub fn exit_selection(&mut self) {
        self.selection = None;
    }

    /// Calculate the visual (row, col) position of the cursor from the byte index.
    /// Returns (row_number, column_width) relative to the start of the text.
    pub fn cursor_visual_pos(&self) -> (usize, usize) {
        let mut row = 0;
        let mut col = 0;
        for ch in self.line[..self.cursor.min(self.line.len())].chars() {
            if ch == '\n' {
                row += 1;
                col = 0;
            } else {
                col += ch.width().unwrap_or(1);
            }
        }
        (row, col)
    }

    /// Total number of lines in the input.
    pub fn line_count(&self) -> usize {
        if self.line.is_empty() {
            1
        } else {
            self.line.matches('\n').count() + 1
        }
    }

    /// Find the byte offset of the first visible line when rendering with max_rows.
    /// Returns the starting byte index of the line that should be rendered first.
    /// Uses scroll-to-bottom: shows the last `max_rows` lines.
    pub fn visible_line_start(&self) -> usize {
        if !self.multiline || self.max_rows == 0 {
            return 0;
        }
        let total_lines = self.line_count();
        if total_lines <= self.max_rows {
            return 0;
        }
        // Find the start of line (total_lines - max_rows), 0-indexed
        let first_visible_line = total_lines - self.max_rows;
        let mut current_line = 0;
        for (byte_idx, ch) in self.line.char_indices() {
            if current_line == first_visible_line {
                return byte_idx;
            }
            if ch == '\n' {
                current_line += 1;
            }
        }
        0
    }
}

const BASE_WIDTH: u16 = 30;

impl Prompt {
    pub fn render_prompt(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let theme = &cx.editor.theme;
        let prompt_color = theme.get("ui.text");
        let completion_color = theme.get("ui.menu");
        let selected_color = theme.get("ui.menu.selected");
        let suggestion_color = theme.get("ui.text.inactive");
        // Use custom background key if set, otherwise default to ui.background
        let background = match self.background_key {
            Some(key) => theme.try_get(key).unwrap_or_else(|| theme.get("ui.background")),
            None => theme.get("ui.background"),
        };

        // ── Multiline rendering ──────────────────────────────────────────
        if self.multiline {
            self.render_multiline(area, surface, cx, prompt_color, suggestion_color, background);
            return;
        }

        // ── Single-line rendering (original) ─────────────────────────────
        // completion
        let max_len = self
            .completion
            .iter()
            .map(|(_, completion)| completion.content.len() as u16)
            .max()
            .unwrap_or(BASE_WIDTH)
            .max(BASE_WIDTH);

        let cols = std::cmp::max(1, area.width / max_len);
        let col_width = (area.width.saturating_sub(cols)) / cols;

        let height = (self.completion.len() as u16)
            .div_ceil(cols)
            .min(10) // at most 10 rows (or less)
            .min(area.height.saturating_sub(1));

        let completion_area = Rect::new(
            area.x,
            (area.height - height).saturating_sub(1),
            area.width,
            height,
        );

        if completion_area.height > 0 && !self.completion.is_empty() {
            let area = completion_area;
            let background = theme.get("ui.menu");

            let items = height as usize * cols as usize;

            let offset = self
                .selection
                .map(|selection| selection / items * items)
                .unwrap_or_default();

            surface.clear_with(area, background);

            let mut row = 0;
            let mut col = 0;

            for (i, (_range, completion)) in
                self.completion.iter().enumerate().skip(offset).take(items)
            {
                let is_selected = Some(i) == self.selection;

                let completion_item_style = if is_selected {
                    selected_color
                } else {
                    completion_color.patch(completion.style)
                };

                surface.set_stringn(
                    area.x + col * (1 + col_width),
                    area.y + row,
                    &completion.content,
                    col_width.saturating_sub(1) as usize,
                    completion_item_style,
                );

                row += 1;
                if row > area.height - 1 {
                    row = 0;
                    col += 1;
                }
            }
        }

        if let Some(doc) = (self.doc_fn)(&self.line) {
            let mut text = ui::Text::new(doc.to_string());

            let max_width = BASE_WIDTH * 3;
            let horizontal_padding = 2; // border + margin
            let vertical_padding = 1; // border only
            let text_width = max_width - horizontal_padding * 2;

            let viewport = area;

            let (_width, height) = ui::text::required_size(&text.contents, text_width);

            let area = viewport.intersection(Rect::new(
                completion_area.x,
                completion_area
                    .y
                    .saturating_sub(height + vertical_padding * 2),
                max_width,
                height + vertical_padding * 2,
            ));

            let background = theme.get("ui.help");
            surface.clear_with(area, background);

            let block = Block::bordered()
                // .title(self.title.as_str())
                .border_style(background);

            let inner = block.inner(area).inner(Margin::horizontal(1));

            block.render(area, surface);
            text.render(inner, surface, cx);
        }

        let line = area.height - 1;
        surface.clear_with(area.clip_top(line), background);
        // render buffer text
        surface.set_string(area.x, area.y + line, &self.prompt, prompt_color);

        self.line_area = area
            .clip_left(self.prompt.len() as u16)
            .clip_top(line)
            .clip_right(2);

        if self.line.is_empty() {
            self.anchor = 0;
            // Show the most recently entered value as a suggestion.
            if let Some(suggestion) = self.first_history_completion(cx.editor) {
                surface.set_string(
                    self.line_area.x,
                    self.line_area.y,
                    &suggestion,
                    suggestion_color,
                );
            }
        } else if let Some((language, loader)) = self.language.as_ref() {
            let mut text: ui::text::Text = crate::ui::markdown::highlighted_code_block(
                &self.line,
                language,
                Some(&cx.editor.theme),
                &loader.load(),
                None,
            )
            .into();
            text.render(self.line_area, surface, cx);
        } else {
            let line_width = self.line_area.width as usize;

            if self.line.width() < line_width {
                self.anchor = 0;
            } else if self.cursor <= self.anchor {
                // Ensure the grapheme under the cursor is in view.
                self.anchor = self.line[..self.cursor]
                    .grapheme_indices(true)
                    .next_back()
                    .map(|(i, _)| i)
                    .unwrap_or_default();
            } else if self.line[self.anchor..self.cursor].width() > line_width {
                // Set the anchor to the last grapheme cluster before the width is exceeded.
                let mut width = 0;
                self.anchor = self.line[..self.cursor]
                    .grapheme_indices(true)
                    .rev()
                    .find_map(|(idx, g)| {
                        width += g.width();
                        if width > line_width {
                            Some(idx + g.len())
                        } else {
                            None
                        }
                    })
                    .unwrap();
            }

            self.truncate_start = self.anchor > 0;
            self.truncate_end = self.line[self.anchor..].width() > line_width;

            // if we keep inserting characters just before the end elipsis, we move the anchor
            // so that those new characters are displayed
            if self.truncate_end && self.line[self.anchor..self.cursor].width() >= line_width {
                // Move the anchor forward by one non-zero-width grapheme.
                self.anchor += self.line[self.anchor..]
                    .grapheme_indices(true)
                    .find_map(|(idx, g)| {
                        if g.width() > 0 {
                            Some(idx + g.len())
                        } else {
                            None
                        }
                    })
                    .unwrap();
            }

            surface.set_string_anchored(
                self.line_area.x,
                self.line_area.y,
                self.truncate_start,
                self.truncate_end,
                &self.line.as_str()[self.anchor..],
                line_width,
                |_| prompt_color,
            );
        }
    }

    /// Render multiline input. Shows last `max_rows` lines (scroll-to-bottom).
    /// Scrolls to keep the cursor row visible when it moves outside the viewport.
    fn render_multiline(
        &mut self,
        area: Rect,
        surface: &mut Surface,
        _cx: &mut Context,
        prompt_color: helix_view::theme::Style,
        suggestion_color: helix_view::theme::Style,
        background: helix_view::theme::Style,
    ) {
        let prefix_width = self.prompt.width();
        let line_width = (area.width - prefix_width as u16).max(1) as usize;

        // Determine visible lines - scroll to keep cursor in view
        let total_lines = self.line_count();
        let visible_count = self.max_rows.min(area.height as usize).max(1);
        let cursor_row = self.cursor_visual_pos().0;

        let first_visible = if total_lines > visible_count {
            // Start with scroll-to-bottom
            let mut first = total_lines - visible_count;
            // If cursor is above the visible area, scroll up
            if cursor_row < first {
                first = cursor_row;
            }
            // If cursor is below the visible area, scroll down
            let last_visible = first + visible_count;
            if cursor_row >= last_visible {
                first = cursor_row - visible_count + 1;
            }
            first
        } else {
            0
        };

        // Clear the area
        surface.clear_with(area, background);

        if self.line.is_empty() {
            // Show placeholder
            let placeholder = format!("{}[type your request]", self.prompt);
            let placeholder_row = area.y + area.height.saturating_sub(1);
            surface.set_string(area.x, placeholder_row, &placeholder, suggestion_color);
            self.line_area = area
                .clip_left(prefix_width as u16)
                .clip_top(area.height.saturating_sub(1))
                .clip_right(2);
            self.anchor = 0;
            self.truncate_start = false;
            self.truncate_end = false;
            return;
        }

        // Split into lines and render visible ones
        let lines: Vec<&str> = self.line.lines().collect();
        let start_idx = first_visible.min(lines.len());
        let end_idx = (start_idx + visible_count).min(lines.len());

        for (i, line_idx) in (start_idx..end_idx).enumerate() {
            let row = area.y + i as u16;
            let line_text = lines[line_idx];
            let is_first = i == 0 && start_idx == 0;

            // Draw prefix
            let prefix = if is_first { &self.prompt } else { " " };
            surface.set_string(area.x, row, prefix, prompt_color);

            // Draw line content with anchoring
            let text_x = area.x + prefix_width as u16;

            if line_text.width() <= line_width {
                surface.set_string(text_x, row, line_text, prompt_color);
            } else {
                // Find which line the cursor is on for anchoring
                let (cursor_row, _) = self.cursor_visual_pos();
                let cursor_line_idx = cursor_row;

                if cursor_line_idx == line_idx {
                    // Cursor is on this line - calculate anchor for this line
                    let line_start = lines[..line_idx].iter().map(|l| l.len() + 1).sum::<usize>();
                    let (_, cursor_col) = self.cursor_visual_pos();

                    let local_cursor = cursor_col;
                    let local_line = line_text;

                    if local_line.width() <= line_width {
                        self.anchor = line_start;
                        self.truncate_start = false;
                        self.truncate_end = false;
                        surface.set_string(text_x, row, local_line, prompt_color);
                    } else {
                        // Ensure anchor is at least line_start to avoid underflow
                        let anchor_offset = self.anchor.max(line_start) - line_start;
                        if local_cursor <= anchor_offset {
                            self.anchor = line_start;
                            self.truncate_start = false;
                            let safe_offset = self.anchor.saturating_sub(line_start);
                            self.truncate_end = local_line[safe_offset..].width() > line_width;
                            surface.set_string_anchored(
                                text_x, row, self.truncate_start, self.truncate_end,
                                &local_line[safe_offset..],
                                line_width, |_| prompt_color,
                            );
                        } else {
                            self.anchor = line_start;
                            self.truncate_start = false;
                            self.truncate_end = true;
                            surface.set_string_anchored(
                                text_x, row, self.truncate_start, self.truncate_end,
                                local_line, line_width, |_| prompt_color,
                            );
                        }
                    }
                } else {
                    // Not cursor line - just truncate
                    surface.set_string_anchored(
                        text_x, row, false, line_text.width() > line_width,
                        line_text, line_width, |_| prompt_color,
                    );
                }
            }
        }

        // Set line_area for cursor calculation (use cursor's row)
        let (cursor_row, _) = self.cursor_visual_pos();
        let cursor_visible_row = cursor_row.saturating_sub(first_visible).min(visible_count.saturating_sub(1));
        self.line_area = Rect {
            x: area.x + prefix_width as u16,
            y: area.y + cursor_visible_row as u16,
            width: area.width - prefix_width as u16,
            height: 1,
        };
    }
}

impl Component for Prompt {
    fn handle_event(&mut self, event: &Event, cx: &mut Context) -> EventResult {
        let event = match event {
            Event::Paste(data) => {
                self.insert_str(data, cx.editor);
                self.recalculate_completion(cx.editor);
                return EventResult::Consumed(None);
            }
            Event::Key(event) => *event,
            Event::Resize(..) => return EventResult::Consumed(None),
            // Prompt is a modal and should consume mouse events so clicks don't fall
            // through to the editor underneath
            Event::Mouse(_) => return EventResult::Consumed(None),
            _ => return EventResult::Ignored(None),
        };

        let close_fn = EventResult::Consumed(Some(Box::new(|compositor: &mut Compositor, _| {
            // remove the layer
            compositor.pop();
        })));

        match event {
            ctrl!('c') | key!(Esc) => {
                (self.callback_fn)(cx, &self.line, PromptEvent::Abort);
                return close_fn;
            }
            alt!('b') | ctrl!(Left) => self.move_cursor(Movement::BackwardWord(1)),
            alt!('f') | ctrl!(Right) => self.move_cursor(Movement::ForwardWord(1)),
            ctrl!('b') | key!(Left) => self.move_cursor(Movement::BackwardChar(1)),
            ctrl!('f') | key!(Right) => self.move_cursor(Movement::ForwardChar(1)),
            ctrl!('e') | key!(End) => {
                if self.multiline {
                    self.move_cursor(Movement::EndOfLine)
                } else {
                    self.move_end()
                }
            }
            ctrl!('a') | key!(Home) => {
                if self.multiline {
                    self.move_cursor(Movement::StartOfLine)
                } else {
                    self.move_start()
                }
            }
            ctrl!('w') | alt!(Backspace) | ctrl!(Backspace) => {
                self.delete_word_backwards(cx.editor);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            alt!('d') | alt!(Delete) | ctrl!(Delete) => {
                self.delete_word_forwards(cx.editor);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            ctrl!('k') => {
                self.kill_to_end_of_line(cx.editor);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            ctrl!('u') => {
                self.kill_to_start_of_line(cx.editor);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            ctrl!('h') | key!(Backspace) | shift!(Backspace) => {
                self.delete_char_backwards(cx.editor);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            ctrl!('d') | key!(Delete) => {
                self.delete_char_forwards(cx.editor);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            ctrl!('s') => {
                let (view, doc) = current!(cx.editor);
                let text = doc.text().slice(..);

                use helix_core::textobject;
                let range = textobject::textobject_word(
                    text,
                    doc.selection(view.id).primary(),
                    textobject::TextObject::Inside,
                    1,
                    false,
                );
                let line = text.slice(range.from()..range.to()).to_string();
                if !line.is_empty() {
                    self.insert_str(line.as_str(), cx.editor);
                    (self.callback_fn)(cx, &self.line, PromptEvent::Update);
                }
            }
            // Multiline: Shift+Enter / Ctrl+Enter / Ctrl+J inserts newline
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
            } if self.multiline && (modifiers.contains(KeyModifiers::SHIFT) || modifiers.contains(KeyModifiers::CONTROL)) => {
                self.line.insert(self.cursor, '\n');
                let mut cursor = GraphemeCursor::new(self.cursor, self.line.len(), false);
                if let Ok(Some(pos)) = cursor.next_boundary(&self.line, 0) {
                    self.cursor = pos;
                }
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            KeyEvent {
                code: KeyCode::Char('j'),
                modifiers,
            } if self.multiline && modifiers.contains(KeyModifiers::CONTROL) => {
                self.line.insert(self.cursor, '\n');
                let mut cursor = GraphemeCursor::new(self.cursor, self.line.len(), false);
                if let Ok(Some(pos)) = cursor.next_boundary(&self.line, 0) {
                    self.cursor = pos;
                }
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            key!(Enter) => {
                if self.selection.is_some() && self.line.ends_with(std::path::MAIN_SEPARATOR) {
                    self.recalculate_completion(cx.editor);
                } else if self.multiline {
                    // In multiline mode, plain Enter also submits (callback handles it)
                    let input = if self.line.is_empty() {
                        ""
                    } else {
                        &self.line
                    };
                    (self.callback_fn)(cx, input, PromptEvent::Validate);
                    return close_fn;
                } else {
                    let last_item = self
                        .first_history_completion(cx.editor)
                        .map(|entry| entry.to_string())
                        .unwrap_or_else(|| String::from(""));

                    // handle executing with last command in history if nothing entered
                    let input = if self.line.is_empty() {
                        &last_item
                    } else {
                        if last_item != self.line {
                            // store in history
                            if let Some(register) = self.history_register {
                                if let Err(err) =
                                    cx.editor.registers.push(register, self.line.clone())
                                {
                                    cx.editor.set_error(err.to_string());
                                }
                            };
                        }

                        &self.line
                    };

                    (self.callback_fn)(cx, input, PromptEvent::Validate);

                    return close_fn;
                }
            }
            ctrl!('p') | key!(Up) => {
                if self.multiline {
                    self.move_cursor(Movement::Up);
                    (self.callback_fn)(cx, &self.line, PromptEvent::Update);
                } else if let Some(register) = self.history_register {
                    self.change_history(cx, register, CompletionDirection::Backward);
                }
            }
            ctrl!('n') | key!(Down) => {
                if self.multiline {
                    self.move_cursor(Movement::Down);
                    (self.callback_fn)(cx, &self.line, PromptEvent::Update);
                } else if let Some(register) = self.history_register {
                    self.change_history(cx, register, CompletionDirection::Forward);
                }
            }
            key!(Tab) => {
                self.change_completion_selection(CompletionDirection::Forward);
                // if single completion candidate is a directory list content in completion
                if self.completion.len() == 1 && self.line.ends_with(std::path::MAIN_SEPARATOR) {
                    self.recalculate_completion(cx.editor);
                }
                (self.callback_fn)(cx, &self.line, PromptEvent::Update)
            }
            shift!(Tab) => {
                self.change_completion_selection(CompletionDirection::Backward);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update)
            }
            ctrl!('q') => self.exit_selection(),
            ctrl!('r') => {
                self.completion = cx
                    .editor
                    .registers
                    .iter_preview()
                    .map(|(ch, preview)| (0.., format!("{} {}", ch, &preview).into()))
                    .collect();
                self.next_char_handler = Some(Box::new(|prompt, c, context| {
                    prompt.insert_str(
                        &context
                            .editor
                            .registers
                            .first(c, context.editor)
                            .unwrap_or_default(),
                        context.editor,
                    );
                }));
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
                return EventResult::Consumed(None);
            }
            // any char event that's not mapped to any other combo
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers: _,
            } => {
                self.insert_char(c, cx);
                (self.callback_fn)(cx, &self.line, PromptEvent::Update);
            }
            _ => (),
        };

        EventResult::Consumed(None)
    }

    fn render(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        self.render_prompt(area, surface, cx)
    }

    fn cursor(&self, area: Rect, editor: &Editor) -> (Option<Position>, CursorKind) {
        if self.multiline {
            // Multiline cursor position
            let prefix_width = self.prompt.width();
            let (row, col) = self.cursor_visual_pos();

            // Calculate which row is visible (scroll-to-bottom)
            let total_lines = self.line_count();
            let visible_count = self.max_rows.min(area.height as usize).max(1);
            let first_visible = if total_lines > visible_count {
                total_lines - visible_count
            } else {
                0
            };
            let visible_row = row.saturating_sub(first_visible).min(visible_count.saturating_sub(1));

            let cursor_x = (area.x as usize) + prefix_width + col;
            let cursor_y = (area.y as usize) + visible_row;

            // Clamp to area bounds
            let max_x = (area.x as usize) + (area.width as usize) - 1;
            let max_y = (area.y as usize) + (area.height as usize) - 1;
            if cursor_x <= max_x && cursor_y <= max_y {
                (
                    Some(Position::new(cursor_y, cursor_x.min(max_x))),
                    editor.config().cursor_shape.from_mode(Mode::Insert),
                )
            } else {
                (None, CursorKind::Hidden)
            }
        } else {
            // Single-line cursor position (original)
            let area = area
                .clip_left(self.prompt.len() as u16)
                .clip_right(if self.prompt.is_empty() { 2 } else { 0 });

            let mut col = area.left() as usize + self.line[self.anchor..self.cursor].width();

            // ensure the cursor does not go beyond elipses
            if self.truncate_end
                && self.line[self.anchor..self.cursor].width() >= self.line_area.width as usize
            {
                col -= 1;
            }

            if self.truncate_start && self.cursor == self.anchor {
                col += self.line[self.cursor..]
                    .graphemes(true)
                    .next()
                    .map_or(0, |g| g.width());
            }

            let line = area.height as usize - 1;

            (
                Some(Position::new(area.y as usize + line, col)),
                editor.config().cursor_shape.from_mode(Mode::Insert),
            )
        }
    }
}
