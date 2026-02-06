use crate::game::{Board, Cell, COLS, ROWS};
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

/// Render a compact board (no column selector, no borders) into the given area.
pub fn render_board_compact(frame: &mut Frame, board: &Board, area: Rect) {
    let mut lines = Vec::new();

    for row in 0..ROWS {
        let mut spans = Vec::new();
        for col in 0..COLS {
            let cell = board.get(row, col);
            let (symbol, color) = match cell {
                Cell::Empty => (" . ", Color::DarkGray),
                Cell::Red => (" \u{25cf} ", Color::Red),
                Cell::Yellow => (" \u{25cf} ", Color::Yellow),
            };
            spans.push(Span::styled(symbol, Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    let widget = Paragraph::new(lines);
    frame.render_widget(widget, area);
}
