use crate::game::{GameOutcome, GameState, MoveError};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use ratatui::{backend::Backend, Terminal};
use std::io;

pub struct App {
    game_state: GameState,
    selected_column: usize,
    should_quit: bool,
    message: Option<String>,
}

impl App {
    pub fn new() -> Self {
        App {
            game_state: GameState::initial(),
            selected_column: 3, // Start in middle
            should_quit: false,
            message: None,
        }
    }

    /// Main application loop
    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()>
    where
        B::Error: Into<io::Error>,
    {
        loop {
            terminal.draw(|f| self.render(f)).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            if self.should_quit {
                break;
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

    /// Handle key press
    fn handle_key(&mut self, key: KeyEvent) {
        // Clear message on any key press
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
                // Reset game
                self.game_state = GameState::initial();
                self.selected_column = 3;
                self.message = Some("New game started!".to_string());
            }
            _ => {}
        }
    }

    /// Drop piece in selected column
    fn drop_piece(&mut self) {
        if self.game_state.is_terminal() {
            self.message = Some("Game over! Press 'r' to restart.".to_string());
            return;
        }

        match self.game_state.apply_move_mut(self.selected_column) {
            Ok(()) => {
                // Check if game just ended
                if let Some(outcome) = self.game_state.outcome() {
                    self.message = Some(match outcome {
                        GameOutcome::Winner(player) => {
                            format!("{} wins!", player.name())
                        }
                        GameOutcome::Draw => "It's a draw!".to_string(),
                    });
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

    /// Render the UI
    fn render(&self, frame: &mut ratatui::Frame) {
        super::game_view::render(frame, &self.game_state, self.selected_column, &self.message);
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
