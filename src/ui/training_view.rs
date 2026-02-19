use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline},
    Frame,
};

use super::board_widget;
use super::training_dashboard::{DashboardState, TrainingStatus};

/// Render the full training dashboard.
pub fn render(frame: &mut Frame, dashboard: &DashboardState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),  // Main content
            Constraint::Length(3), // Footer
        ])
        .split(frame.area());

    render_header(frame, dashboard, chunks[0]);
    render_main(frame, dashboard, chunks[1]);
    render_footer(frame, chunks[2]);
}

fn render_header(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let status_str = match dashboard.status {
        TrainingStatus::Running => "RUNNING",
        TrainingStatus::Paused => "PAUSED",
        TrainingStatus::Finished => "FINISHED",
    };
    let status_color = match dashboard.status {
        TrainingStatus::Running => Color::Green,
        TrainingStatus::Paused => Color::Yellow,
        TrainingStatus::Finished => Color::Cyan,
    };

    let header_text = Line::from(vec![
        Span::styled(format!("Training: {}", dashboard.algorithm), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw("  |  "),
        Span::raw(format!(
            "Episode: {}/{}",
            dashboard.episode, dashboard.total_episodes
        )),
        Span::raw("  |  ["),
        Span::styled(status_str, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::raw("]"),
    ]);

    let header = Paragraph::new(header_text)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(header, area);
}

fn render_main(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    // Split into left (charts) and right (board + stats) panels
    let main_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    render_left_panel(frame, dashboard, main_cols[0]);
    render_right_panel(frame, dashboard, main_cols[1]);
}

fn render_left_panel(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40), // Win rate chart
            Constraint::Percentage(35), // Loss chart
            Constraint::Length(3),      // Game length sparkline
            Constraint::Length(3),      // Progress gauge
        ])
        .split(area);

    render_win_rate_chart(frame, dashboard, left_rows[0]);
    render_loss_chart(frame, dashboard, left_rows[1]);
    render_game_length_sparkline(frame, dashboard, left_rows[2]);
    render_progress_gauge(frame, dashboard, left_rows[3]);
}

fn render_right_panel(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    render_live_board(frame, dashboard, right_rows[0]);
    render_stats_panel(frame, dashboard, right_rows[1]);
}

fn render_win_rate_chart(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let win_data: Vec<(f64, f64)> = dashboard.win_rate_history.iter().copied().collect();
    let eval_data: Vec<(f64, f64)> = dashboard.eval_history.iter().copied().collect();

    let (x_min, x_max) = x_bounds(&win_data, &eval_data, dashboard.total_episodes);

    let mut datasets = vec![];
    if !win_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Win Rate")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&win_data),
        );
    }
    if !eval_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Eval vs Random")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(Color::Cyan))
                .data(&eval_data),
        );
    }

    let x_labels = vec![
        Span::raw(format!("{}", x_min as usize)),
        Span::raw(format!("{}", x_max as usize)),
    ];
    let y_labels = vec![
        Span::raw("0%"),
        Span::raw("50%"),
        Span::raw("100%"),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Win Rate"),
        )
        .x_axis(
            Axis::default()
                .title("Episode")
                .labels(x_labels)
                .bounds([x_min, x_max]),
        )
        .y_axis(
            Axis::default()
                .title("Rate")
                .labels(y_labels)
                .bounds([0.0, 1.0]),
        );

    frame.render_widget(chart, area);
}

fn render_loss_chart(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let loss_data: Vec<(f64, f64)> = dashboard.loss_history.iter().copied().collect();

    let (x_min, x_max) = if let (Some(first), Some(last)) =
        (loss_data.first(), loss_data.last())
    {
        (first.0, last.0.max(first.0 + 1.0))
    } else {
        (0.0, dashboard.total_episodes.max(1) as f64)
    };

    let y_max = loss_data
        .iter()
        .map(|&(_, y)| y)
        .fold(0.1_f64, f64::max);
    // Round up to nearest 0.1
    let y_max = ((y_max * 10.0).ceil() / 10.0).max(0.1);

    let mut datasets = vec![];
    if !loss_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Loss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red))
                .data(&loss_data),
        );
    }

    let x_labels = vec![
        Span::raw(format!("{}", x_min as usize)),
        Span::raw(format!("{}", x_max as usize)),
    ];
    let y_labels = vec![
        Span::raw("0"),
        Span::raw(format!("{:.2}", y_max)),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Loss"),
        )
        .x_axis(
            Axis::default()
                .title("Episode")
                .labels(x_labels)
                .bounds([x_min, x_max]),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .labels(y_labels)
                .bounds([0.0, y_max]),
        );

    frame.render_widget(chart, area);
}

fn render_game_length_sparkline(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let data: Vec<u64> = dashboard.game_length_history.iter().copied().collect();

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    "Game Length (avg: {:.1})",
                    dashboard.avg_game_length
                )),
        )
        .data(&data)
        .style(Style::default().fg(Color::Magenta));

    frame.render_widget(sparkline, area);
}

fn render_progress_gauge(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let progress = dashboard.progress();
    let label = format!(
        "{}/{} ({:.1}%)",
        dashboard.episode,
        dashboard.total_episodes,
        progress * 100.0
    );

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Progress"),
        )
        .gauge_style(Style::default().fg(Color::Blue))
        .ratio(progress.clamp(0.0, 1.0))
        .label(label);

    frame.render_widget(gauge, area);
}

fn render_live_board(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!("Live Game (move {})", dashboard.live_move_number));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if let Some(ref game_state) = dashboard.live_game {
        board_widget::render_board_compact(frame, game_state.board(), inner);
    } else {
        let placeholder = Paragraph::new("Waiting for first game...")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        frame.render_widget(placeholder, inner);
    }
}

fn render_stats_panel(frame: &mut Frame, dashboard: &DashboardState, area: Rect) {
    let loss_rate = 1.0 - dashboard.win_rate - dashboard.draw_rate;

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Win Rate:   ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{:.1}%", dashboard.win_rate * 100.0),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Draw Rate:  ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{:.1}%", dashboard.draw_rate * 100.0),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Loss Rate:  ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{:.1}%", loss_rate * 100.0),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(""),
    ];

    if dashboard.algorithm == "PG" {
        if let Some(entropy) = dashboard.policy_entropy {
            lines.push(Line::from(vec![
                Span::styled("Entropy:    ", Style::default().fg(Color::White)),
                Span::raw(format!("{:.4}", entropy)),
            ]));
        }
    } else {
        lines.push(Line::from(vec![
            Span::styled("Epsilon:    ", Style::default().fg(Color::White)),
            Span::raw(format!("{:.4}", dashboard.epsilon)),
        ]));
    }

    lines.extend(vec![
        Line::from(vec![
            Span::styled("Loss:       ", Style::default().fg(Color::White)),
            Span::raw(format!("{:.6}", dashboard.loss)),
        ]),
        Line::from(vec![
            Span::styled("Steps:      ", Style::default().fg(Color::White)),
            Span::raw(format!("{}", dashboard.step_count)),
        ]),
        Line::from(vec![
            Span::styled("Avg Length:  ", Style::default().fg(Color::White)),
            Span::raw(format!("{:.1}", dashboard.avg_game_length)),
        ]),
    ]);

    if dashboard.episodes_per_sec > 0.0 {
        lines.extend(vec![
            Line::from(vec![
                Span::styled("Ep/sec:     ", Style::default().fg(Color::White)),
                Span::raw(format!("{:.1}", dashboard.episodes_per_sec)),
            ]),
            Line::from(vec![
                Span::styled("Ep time:    ", Style::default().fg(Color::White)),
                Span::raw(format!("{:.1}ms", dashboard.avg_episode_ms)),
            ]),
            Line::from(vec![
                Span::styled("Upd time:   ", Style::default().fg(Color::White)),
                Span::raw(format!("{:.1}ms", dashboard.avg_update_ms)),
            ]),
        ]);
    }

    if let Some(eval_wr) = dashboard.last_eval_win_rate {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Eval vs Random: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{:.1}%", eval_wr * 100.0),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    if let Some(ref ckpt) = dashboard.last_checkpoint {
        lines.push(Line::from(vec![
            Span::styled("Last Save: ", Style::default().fg(Color::White)),
            Span::styled(ckpt.clone(), Style::default().fg(Color::DarkGray)),
        ]));
    }

    let stats = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Stats"),
    );

    frame.render_widget(stats, area);
}

fn render_footer(frame: &mut Frame, area: Rect) {
    let footer = Paragraph::new("P: Pause/Resume  |  S: Save Checkpoint  |  Q: Quit")
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).title("Controls"));

    frame.render_widget(footer, area);
}

/// Compute x-axis bounds from data points.
fn x_bounds(
    data1: &[(f64, f64)],
    data2: &[(f64, f64)],
    total_episodes: usize,
) -> (f64, f64) {
    let first1 = data1.first().map(|d| d.0);
    let first2 = data2.first().map(|d| d.0);
    let x_min = match (first1, first2) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => 0.0,
    };

    let last1 = data1.last().map(|d| d.0);
    let last2 = data2.last().map(|d| d.0);
    let x_max = match (last1, last2) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => total_episodes.max(1) as f64,
    };

    (x_min, x_max.max(x_min + 1.0))
}
