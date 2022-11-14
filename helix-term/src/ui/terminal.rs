use std::cell::Cell;

use helix_core::Position;
use helix_view::{
    graphics::{CursorKind, Rect},
    input,
    theme::Modifier,
    Editor,
};
use tui::buffer::Buffer;

use crate::compositor::{Component, Context, EventResult};

pub struct Terminal {
    size: Cell<(u16, u16)>,
    prev_pressed: Cell<bool>,
}

impl Terminal {
    pub fn new() -> Self {
        Self {
            size: Cell::new((24, 80)),
            prev_pressed: Cell::new(false),
        }
    }
}

impl Component for Terminal {
    fn render(&mut self, area: Rect, surface: &mut Buffer, ctx: &mut Context) {
        if self.size.get() != (area.height, area.width) && area.height > 0 && area.width > 0 {
            self.size.set((area.height, area.width));

            ctx.editor
                .terminals
                .handle_input_event(&input::Event::Resize(self.size.get().1, self.size.get().0));
        }

        if let Some((_, term)) = ctx.editor.terminals.get_active() {
            let content = term.renderable_content();
            log::info!("offset {:?}", content.display_offset);
            surface.clear(area);

            for cell in content.display_iter {
                if let Some(c) = surface.get_mut(
                    area.left() + cell.point.column.0 as u16,
                    area.top() + (cell.point.line.0 + content.display_offset as i32) as u16,
                ) {
                    let style = helix_view::theme::Style::reset()
                        .bg(cell.bg.into())
                        .fg(cell.fg.into())
                        .add_modifier(unsafe { Modifier::from_bits_unchecked(cell.flags.bits()) });

                    let style = if let Some(col) = cell.underline_color() {
                        style.underline_color(col.into())
                    } else {
                        style
                    };

                    c.reset();
                    c.set_char(cell.c);
                    c.set_style(style);
                }
            }
        }
    }

    fn cursor(&self, area: Rect, editor: &Editor) -> (Option<Position>, CursorKind) {
        editor
            .terminals
            .get_active()
            .map(|(_, term)| {
                let pt = term.grid().cursor.point;
                (
                    Some(Position {
                        row: area.y as usize + pt.line.0 as usize,
                        col: area.x as usize + pt.column.0 as usize,
                    }),
                    term.cursor_style().shape.into(),
                )
            })
            .unwrap_or((None, CursorKind::Hidden))
    }

    fn handle_event(&mut self, event: &input::Event, context: &mut Context) -> EventResult {
        match event {
            input::Event::Key(input::KeyEvent {
                code: input::KeyCode::Char('\\'),
                ..
            }) => {
                if self.prev_pressed.get() {
                    context.editor.terminals.visible = false;
                    EventResult::Consumed(None)
                } else {
                    self.prev_pressed.set(true);

                    if context.editor.terminals.handle_input_event(event) {
                        EventResult::Consumed(None)
                    } else {
                        EventResult::Ignored(None)
                    }
                }
            }

            input::Event::Resize(_, _) => EventResult::Ignored(None),

            event => {
                if let input::Event::Key(..) = &event {
                    self.prev_pressed.set(false);
                }

                if context.editor.terminals.handle_input_event(event) {
                    EventResult::Consumed(None)
                } else {
                    EventResult::Ignored(None)
                }
            }
        }
    }
}
