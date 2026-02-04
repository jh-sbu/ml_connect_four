use crate::ai::{Agent, RandomAgent};
use crate::game::{GameOutcome, GameState, MoveError, Player};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use ratatui::{backend::Backend, Terminal};
use std::io;

pub enum PlayerKind {
    Human,
    Ai(Box<dyn Agent>),
}

impl PlayerKind {
    fn label(&self) -> &str {
        match self {
            PlayerKind::Human => "Human",
            PlayerKind::Ai(agent) => agent.name(),
        }
    }
}

pub struct App {
    game_state: GameState,
    selected_column: usize,
    should_quit: bool,
    message: Option<String>,
    red_player: PlayerKind,
    yellow_player: PlayerKind,
    ai_move_pending: bool,
}

impl App {
    pub fn new() -> Self {
        App {
            game_state: GameState::initial(),
            selected_column: 3,
            should_quit: false,
            message: None,
            red_player: PlayerKind::Human,
            yellow_player: PlayerKind::Human,
            ai_move_pending: false,
        }
    }

    /// Main application loop
    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()>
    where
        B::Error: Into<io::Error>,
    {
        loop {
            terminal
                .draw(|f| self.render(f))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            if self.should_quit {
                break;
            }

            if self.ai_move_pending && !self.game_state.is_terminal() {
                std::thread::sleep(std::time::Duration::from_millis(300));
                self.do_ai_move();
                continue;
            }

            self.handle_events()?;
        }
        Ok(())
    }

    /// Handle keyboard events
    fn handle_events(&mut self) -> io::Result<()> {
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                self.handle_key(key);
            }
        }
        Ok(())
    }

    fn is_current_player_ai(&self) -> bool {
        match self.game_state.current_player() {
            Player::Red => matches!(self.red_player, PlayerKind::Ai(_)),
            Player::Yellow => matches!(self.yellow_player, PlayerKind::Ai(_)),
        }
    }

    fn do_ai_move(&mut self) {
        self.ai_move_pending = false;

        if self.game_state.is_terminal() {
            return;
        }

        let action = match self.game_state.current_player() {
            Player::Red => {
                if let PlayerKind::Ai(ref mut agent) = self.red_player {
                    agent.select_action(&self.game_state, false)
                } else {
                    return;
                }
            }
            Player::Yellow => {
                if let PlayerKind::Ai(ref mut agent) = self.yellow_player {
                    agent.select_action(&self.game_state, false)
                } else {
                    return;
                }
            }
        };

        match self.game_state.apply_move_mut(action) {
            Ok(()) => {
                self.selected_column = action;
                if let Some(outcome) = self.game_state.outcome() {
                    self.message = Some(match outcome {
                        GameOutcome::Winner(player) => format!("{} wins!", player.name()),
                        GameOutcome::Draw => "It's a draw!".to_string(),
                    });
                } else if self.is_current_player_ai() {
                    self.ai_move_pending = true;
                }
            }
            Err(_) => {
                self.message = Some("AI made an invalid move!".to_string());
            }
        }
    }

    /// Handle key press
    fn handle_key(&mut self, key: KeyEvent) {
        self.message = None;

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }
            KeyCode::Left => {
                if self.selected_column > 0 {
                    self.selected_column -= 1;
                }
            }
            KeyCode::Right => {
                if self.selected_column < 6 {
                    self.selected_column += 1;
                }
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.drop_piece();
            }
            KeyCode::Char('r') => {
                self.game_state = GameState::initial();
                self.selected_column = 3;
                self.ai_move_pending = false;
                self.message = Some("New game started!".to_string());

                // If Red is AI, trigger its move
                if self.is_current_player_ai() {
                    self.ai_move_pending = true;
                }
            }
            KeyCode::Char('a') => {
                self.toggle_ai();
            }
            _ => {}
        }
    }

    fn toggle_ai(&mut self) {
        self.yellow_player = match self.yellow_player {
            PlayerKind::Human => PlayerKind::Ai(Box::new(RandomAgent::new())),
            PlayerKind::Ai(_) => PlayerKind::Human,
        };

        // Reset game on toggle
        self.game_state = GameState::initial();
        self.selected_column = 3;
        self.ai_move_pending = false;

        let mode = self.game_mode_label();
        self.message = Some(format!("Mode: {} — New game started!", mode));

        if self.is_current_player_ai() {
            self.ai_move_pending = true;
        }
    }

    /// Drop piece in selected column
    fn drop_piece(&mut self) {
        if self.is_current_player_ai() {
            self.message = Some("Wait for AI to move...".to_string());
            return;
        }

        if self.game_state.is_terminal() {
            self.message = Some("Game over! Press 'r' to restart.".to_string());
            return;
        }

        match self.game_state.apply_move_mut(self.selected_column) {
            Ok(()) => {
                if let Some(outcome) = self.game_state.outcome() {
                    self.message = Some(match outcome {
                        GameOutcome::Winner(player) => format!("{} wins!", player.name()),
                        GameOutcome::Draw => "It's a draw!".to_string(),
                    });
                } else if self.is_current_player_ai() {
                    self.ai_move_pending = true;
                }
            }
            Err(MoveError::ColumnFull) => {
                self.message = Some("Column is full!".to_string());
            }
            Err(MoveError::InvalidColumn) => {
                self.message = Some("Invalid column!".to_string());
            }
            Err(MoveError::GameOver) => {
                self.message = Some("Game is over!".to_string());
            }
        }
    }

    fn game_mode_label(&self) -> String {
        format!(
            "{} vs {}",
            self.red_player.label(),
            self.yellow_player.label()
        )
    }

    /// Render the UI
    fn render(&self, frame: &mut ratatui::Frame) {
        let mode = self.game_mode_label();
        super::game_view::render(
            frame,
            &self.game_state,
            self.selected_column,
            &self.message,
            &mode,
        );
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
