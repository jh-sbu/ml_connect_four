use crate::ai::algorithms::{DqnAgent, DqnConfig, PgConfig, PolicyGradientAgent};
use crate::ai::{Agent, RandomAgent};
use crate::checkpoint::{CheckpointManager, CheckpointManagerConfig};
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
    paused: bool,
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
            paused: false,
        }
    }

    fn is_ai_vs_ai(&self) -> bool {
        matches!(self.red_player, PlayerKind::Ai(_))
            && matches!(self.yellow_player, PlayerKind::Ai(_))
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
                // Poll instead of sleep so user can quit/restart during AI-vs-AI
                if event::poll(std::time::Duration::from_millis(300))? {
                    if let Event::Key(key) = event::read()? {
                        self.handle_key(key);
                    }
                }
                // Re-check: key handler may have reset ai_move_pending or quit
                if self.ai_move_pending && !self.game_state.is_terminal() && !self.paused {
                    self.do_ai_move();
                }
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
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
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
            KeyCode::Enter => {
                self.drop_piece();
            }
            KeyCode::Char(' ') => {
                if self.is_ai_vs_ai() && !self.game_state.is_terminal() {
                    self.paused = !self.paused;
                    self.message = Some(if self.paused {
                        "Paused — Space to resume, N to step".to_string()
                    } else {
                        "Resumed".to_string()
                    });
                } else {
                    self.drop_piece();
                }
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                if self.is_ai_vs_ai()
                    && self.paused
                    && self.ai_move_pending
                    && !self.game_state.is_terminal()
                {
                    self.do_ai_move();
                    if !self.game_state.is_terminal() {
                        self.message = Some("Stepped".to_string());
                    }
                }
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.game_state = GameState::initial();
                self.selected_column = 3;
                self.ai_move_pending = false;
                self.paused = false;
                self.message = Some("New game started!".to_string());

                // If Red is AI, trigger its move
                if self.is_current_player_ai() {
                    self.ai_move_pending = true;
                }
            }
            // Yellow player toggles (lowercase)
            KeyCode::Char('a') => self.toggle_random_for(Player::Yellow),
            KeyCode::Char('d') => self.toggle_dqn_for(Player::Yellow),
            KeyCode::Char('g') => self.toggle_pg_for(Player::Yellow),
            // Red player toggles (uppercase / Shift)
            KeyCode::Char('A') => self.toggle_random_for(Player::Red),
            KeyCode::Char('D') => self.toggle_dqn_for(Player::Red),
            KeyCode::Char('G') => self.toggle_pg_for(Player::Red),
            _ => {}
        }
    }

    fn player_slot(&self, target: Player) -> &PlayerKind {
        match target {
            Player::Red => &self.red_player,
            Player::Yellow => &self.yellow_player,
        }
    }

    fn set_player_slot(&mut self, target: Player, kind: PlayerKind) {
        match target {
            Player::Red => self.red_player = kind,
            Player::Yellow => self.yellow_player = kind,
        }
    }

    fn reset_game_after_toggle(&mut self) {
        self.game_state = GameState::initial();
        self.selected_column = 3;
        self.ai_move_pending = false;
        self.paused = false;

        // Append mode label if no load message was set
        if self.message.is_none() {
            let mode = self.game_mode_label();
            self.message = Some(format!("Mode: {} — New game started!", mode));
        } else {
            let mode = self.game_mode_label();
            let existing = self.message.take().unwrap();
            self.message = Some(format!("{} — Mode: {}", existing, mode));
        }

        // If Red is AI, it goes first
        if self.is_current_player_ai() {
            self.ai_move_pending = true;
        }
    }

    fn toggle_random_for(&mut self, target: Player) {
        let is_ai = matches!(self.player_slot(target), PlayerKind::Ai(_));
        let new_kind = if is_ai {
            PlayerKind::Human
        } else {
            PlayerKind::Ai(Box::new(RandomAgent::new()))
        };
        self.set_player_slot(target, new_kind);
        self.reset_game_after_toggle();
    }

    fn toggle_dqn_for(&mut self, target: Player) {
        let is_ai = matches!(self.player_slot(target), PlayerKind::Ai(_));
        let new_kind = if is_ai {
            PlayerKind::Human
        } else {
            let mut agent = DqnAgent::new(DqnConfig::default());
            agent.set_epsilon(0.0);

            let manager = CheckpointManager::new(CheckpointManagerConfig::default());
            let load_msg = match manager.load_latest() {
                Ok(data) => match agent.load_from_dir(&data.path) {
                    Ok(()) => format!("DQN loaded (episode {})", data.metadata.episode),
                    Err(_) => "DQN (failed to load checkpoint)".to_string(),
                },
                Err(_) => "DQN (untrained, no checkpoint)".to_string(),
            };
            self.message = Some(load_msg);
            PlayerKind::Ai(Box::new(agent))
        };
        self.set_player_slot(target, new_kind);
        self.reset_game_after_toggle();
    }

    fn toggle_pg_for(&mut self, target: Player) {
        let is_ai = matches!(self.player_slot(target), PlayerKind::Ai(_));
        let new_kind = if is_ai {
            PlayerKind::Human
        } else {
            let agent = PolicyGradientAgent::new(PgConfig::default());

            let manager = CheckpointManager::new(CheckpointManagerConfig {
                checkpoint_dir: std::path::PathBuf::from("pg_checkpoints"),
                ..Default::default()
            });
            let (boxed_agent, load_msg) = match manager.load_pg_latest() {
                Ok(data) => {
                    let mut a = agent;
                    match a.load_from_dir(&data.path) {
                        Ok(()) => {
                            let msg = format!("PG loaded (episode {})", data.metadata.episode);
                            (a, msg)
                        }
                        Err(_) => (a, "PG (failed to load checkpoint)".to_string()),
                    }
                }
                Err(_) => (agent, "PG (untrained, no checkpoint)".to_string()),
            };
            self.message = Some(load_msg);
            PlayerKind::Ai(Box::new(boxed_agent))
        };
        self.set_player_slot(target, new_kind);
        self.reset_game_after_toggle();
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
            self.is_ai_vs_ai(),
            self.paused,
        );
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_random_yellow_sets_ai_and_toggles_back() {
        let mut app = App::new();
        assert!(matches!(app.yellow_player, PlayerKind::Human));

        app.toggle_random_for(Player::Yellow);
        assert!(matches!(app.yellow_player, PlayerKind::Ai(_)));
        assert_eq!(app.yellow_player.label(), "Random");

        app.toggle_random_for(Player::Yellow);
        assert!(matches!(app.yellow_player, PlayerKind::Human));
    }

    #[test]
    fn toggle_random_red_sets_ai_and_toggles_back() {
        let mut app = App::new();
        assert!(matches!(app.red_player, PlayerKind::Human));

        app.toggle_random_for(Player::Red);
        assert!(matches!(app.red_player, PlayerKind::Ai(_)));
        assert_eq!(app.red_player.label(), "Random");

        app.toggle_random_for(Player::Red);
        assert!(matches!(app.red_player, PlayerKind::Human));
    }

    #[test]
    fn ai_vs_ai_mode_label() {
        let mut app = App::new();
        app.toggle_random_for(Player::Red);
        app.toggle_random_for(Player::Yellow);
        assert_eq!(app.game_mode_label(), "Random vs Random");
    }

    #[test]
    fn ai_move_pending_when_red_is_ai() {
        let mut app = App::new();
        // Red goes first, so toggling Red to AI should set ai_move_pending
        app.toggle_random_for(Player::Red);
        assert!(app.ai_move_pending);
    }

    #[test]
    fn ai_move_not_pending_when_only_yellow_is_ai() {
        let mut app = App::new();
        // Yellow goes second, Red (Human) goes first
        app.toggle_random_for(Player::Yellow);
        assert!(!app.ai_move_pending);
    }

    #[test]
    fn game_resets_on_toggle() {
        let mut app = App::new();
        // Make a move, then toggle — game should reset
        app.game_state.apply_move_mut(3).unwrap();
        assert_eq!(app.game_state.current_player(), Player::Yellow);

        app.toggle_random_for(Player::Yellow);
        // Game reset: current player back to Red
        assert_eq!(app.game_state.current_player(), Player::Red);
        assert_eq!(app.selected_column, 3);
    }

    #[test]
    fn is_current_player_ai_checks_both_players() {
        let mut app = App::new();
        // Initially Red's turn, no AI
        assert!(!app.is_current_player_ai());

        // Set Red to AI
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert!(app.is_current_player_ai());

        // Make a move so it's Yellow's turn
        app.game_state.apply_move_mut(0).unwrap();
        assert!(!app.is_current_player_ai());

        // Set Yellow to AI
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert!(app.is_current_player_ai());
    }

    #[test]
    fn toggle_preserves_other_player() {
        let mut app = App::new();
        app.toggle_random_for(Player::Red);
        assert!(matches!(app.red_player, PlayerKind::Ai(_)));
        assert!(matches!(app.yellow_player, PlayerKind::Human));

        app.toggle_random_for(Player::Yellow);
        // Both should be AI now
        assert!(matches!(app.red_player, PlayerKind::Ai(_)));
        assert!(matches!(app.yellow_player, PlayerKind::Ai(_)));
    }

    #[test]
    fn mode_label_human_vs_human() {
        let app = App::new();
        assert_eq!(app.game_mode_label(), "Human vs Human");
    }

    #[test]
    fn mode_label_mixed() {
        let mut app = App::new();
        app.toggle_random_for(Player::Red);
        assert_eq!(app.game_mode_label(), "Random vs Human");
    }

    #[test]
    fn is_ai_vs_ai_detection() {
        let mut app = App::new();
        // Human vs Human
        assert!(!app.is_ai_vs_ai());

        // AI vs Human
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert!(!app.is_ai_vs_ai());

        // Human vs AI
        app.red_player = PlayerKind::Human;
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert!(!app.is_ai_vs_ai());

        // AI vs AI
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert!(app.is_ai_vs_ai());
    }

    #[test]
    fn space_toggles_pause_in_ai_vs_ai() {
        let mut app = App::new();
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert!(!app.paused);

        // Space toggles pause on
        app.handle_key(KeyEvent::from(KeyCode::Char(' ')));
        assert!(app.paused);

        // Space toggles pause off
        app.handle_key(KeyEvent::from(KeyCode::Char(' ')));
        assert!(!app.paused);
    }

    #[test]
    fn space_drops_piece_when_human() {
        let mut app = App::new();
        assert_eq!(app.game_state.current_player(), Player::Red);
        app.selected_column = 3;

        app.handle_key(KeyEvent::from(KeyCode::Char(' ')));
        // Piece should have been dropped, turn advances to Yellow
        assert_eq!(app.game_state.current_player(), Player::Yellow);
        assert!(!app.paused);
    }

    #[test]
    fn step_advances_one_move_when_paused() {
        let mut app = App::new();
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.ai_move_pending = true;
        app.paused = true;

        let state_before = app.game_state.clone();
        app.handle_key(KeyEvent::from(KeyCode::Char('n')));

        // Game state should have changed (a move was made)
        assert_ne!(state_before.board(), app.game_state.board());
        // Should still be paused
        assert!(app.paused);
    }

    #[test]
    fn step_ignored_when_not_paused() {
        let mut app = App::new();
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.ai_move_pending = true;
        app.paused = false;

        let state_before = app.game_state.clone();
        app.handle_key(KeyEvent::from(KeyCode::Char('n')));

        // Game state should NOT have changed
        assert_eq!(state_before.board(), app.game_state.board());
    }

    #[test]
    fn restart_resets_pause() {
        let mut app = App::new();
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.paused = true;

        app.handle_key(KeyEvent::from(KeyCode::Char('r')));
        assert!(!app.paused);
    }

    #[test]
    fn toggle_resets_pause() {
        let mut app = App::new();
        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.yellow_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        app.paused = true;

        app.toggle_random_for(Player::Yellow);
        assert!(!app.paused);
    }
}
