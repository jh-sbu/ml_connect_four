use crate::game::{Board, Cell, GameState, Player};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

pub fn render(
    frame: &mut Frame,
    game_state: &GameState,
    selected_column: usize,
    message: &Option<String>,
    game_mode: &str,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(15),  // Board
            Constraint::Length(3), // Message
            Constraint::Length(4), // Controls
        ])
        .split(frame.area());

    render_header(frame, game_state, game_mode, chunks[0]);
    render_board(frame, game_state.board(), selected_column, chunks[1]);
    render_message(frame, message, chunks[2]);
    render_controls(frame, chunks[3]);
}

fn render_header(
    frame: &mut Frame,
    game_state: &GameState,
    game_mode: &str,
    area: ratatui::layout::Rect,
) {
    let current_player = game_state.current_player();
    let (player_name, color) = match current_player {
        Player::Red => ("Red", Color::Red),
        Player::Yellow => ("Yellow", Color::Yellow),
    };

    let status = if game_state.is_terminal() {
        format!("Game Over  |  {}", game_mode)
    } else {
        format!("Current Player: {}  |  {}", player_name, game_mode)
    };

    let header = Paragraph::new(status)
        .style(Style::default().fg(color).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Connect Four"),
        );

    frame.render_widget(header, area);
}

fn render_board(
    frame: &mut Frame,
    board: &Board,
    selected_column: usize,
    area: ratatui::layout::Rect,
) {
    let mut lines = Vec::new();

    // Column numbers with selection indicator
    let mut col_line = vec![Span::raw("   ")]; // Padding (3 chars to match "  ║")
    for col in 0..7 {
        if col == selected_column {
            col_line.push(Span::styled(
                format!(" {} ", col + 1),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            ));
        } else {
            col_line.push(Span::raw(format!(" {} ", col + 1)));
        }
    }
    col_line.push(Span::raw("  ")); // Suffix padding to match " ║"
    lines.push(Line::from(col_line));

    // Top border
    lines.push(Line::from("  ╔══════════════════════╗"));

    // Board rows
    for row in 0..6 {
        let mut row_spans = vec![Span::raw("  ║")];

        for col in 0..7 {
            let cell = board.get(row, col);
            let (symbol, color) = match cell {
                Cell::Empty => (" . ", Color::DarkGray),
                Cell::Red => (" ● ", Color::Red),
                Cell::Yellow => (" ● ", Color::Yellow),
            };
            row_spans.push(Span::styled(symbol, Style::default().fg(color)));
        }

        row_spans.push(Span::raw(" ║"));
        lines.push(Line::from(row_spans));
    }

    // Bottom border
    lines.push(Line::from("  ╚══════════════════════╝"));

    // Selection indicator
    let mut indicator_line = vec![Span::raw("   ")]; // Align with board (3 chars to match "  ║")
    for col in 0..7 {
        if col == selected_column {
            indicator_line.push(Span::styled(
                " ▲ ",
                Style::default().fg(Color::Cyan),
            ));
        } else {
            indicator_line.push(Span::raw("   "));
        }
    }
    indicator_line.push(Span::raw("  ")); // Suffix padding to match " ║"
    lines.push(Line::from(indicator_line));

    let board_widget = Paragraph::new(lines).alignment(Alignment::Center);
    frame.render_widget(board_widget, area);
}

fn render_message(frame: &mut Frame, message: &Option<String>, area: ratatui::layout::Rect) {
    let text = message.as_deref().unwrap_or("");
    let msg_widget = Paragraph::new(text)
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(msg_widget, area);
}

fn render_controls(frame: &mut Frame, area: ratatui::layout::Rect) {
    let line1 = Line::from("←/→: Move  |  Enter: Drop  |  R: Restart  |  Q: Quit");
    let line2 = Line::from(vec![
        Span::styled("Yellow", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(": a Random  d DQN  g PG   "),
        Span::styled("Red", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::raw(": A Random  D DQN  G PG"),
    ]);

    let controls = Paragraph::new(vec![line1, line2])
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Controls"),
        );

    frame.render_widget(controls, area);
}
