use crate::ai::algorithms::{DqnAgent, DqnConfig, PgConfig, PolicyGradientAgent};
use crate::ai::{Agent, RandomAgent};
use crate::checkpoint::{CheckpointManager, CheckpointManagerConfig};
use crate::game::{GameOutcome, GameState, MoveError, Player};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use ratatui::{backend::Backend, Terminal};
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    RandomAi,
    DqnAi,
    PgAi,
}

impl PlayerType {
    pub const ALL: [PlayerType; 4] = [Self::Human, Self::RandomAi, Self::DqnAi, Self::PgAi];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Human => "Human",
            Self::RandomAi => "Random AI",
            Self::DqnAi => "DQN AI",
            Self::PgAi => "Policy Gradient AI",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Playing,
    SelectingPlayer { target: Player, cursor: usize },
}

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
    mode: AppMode,
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
            mode: AppMode::Playing,
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

            if self.ai_move_pending
                && !self.game_state.is_terminal()
                && matches!(self.mode, AppMode::Playing)
            {
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

    /// Handle key press — routes by current mode
    fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            AppMode::Playing => self.handle_key_playing(key),
            AppMode::SelectingPlayer { .. } => self.handle_key_menu(key),
        }
    }

    fn handle_key_playing(&mut self, key: KeyEvent) {
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
            KeyCode::Char('1') => self.open_player_menu(Player::Red),
            KeyCode::Char('2') => self.open_player_menu(Player::Yellow),
            _ => {}
        }
    }

    fn handle_key_menu(&mut self, key: KeyEvent) {
        let AppMode::SelectingPlayer { target, cursor } = self.mode else {
            return;
        };

        match key.code {
            KeyCode::Up => {
                if cursor > 0 {
                    self.mode = AppMode::SelectingPlayer {
                        target,
                        cursor: cursor - 1,
                    };
                }
            }
            KeyCode::Down => {
                if cursor < PlayerType::ALL.len() - 1 {
                    self.mode = AppMode::SelectingPlayer {
                        target,
                        cursor: cursor + 1,
                    };
                }
            }
            KeyCode::Enter => {
                let selected = PlayerType::ALL[cursor];
                self.mode = AppMode::Playing;
                self.set_player_type(target, selected);
            }
            KeyCode::Esc => {
                self.mode = AppMode::Playing;
            }
            _ => {}
        }
    }

    fn current_player_type(&self, target: Player) -> PlayerType {
        match self.player_slot(target) {
            PlayerKind::Human => PlayerType::Human,
            PlayerKind::Ai(agent) => match agent.name() {
                "Random" => PlayerType::RandomAi,
                "DQN" => PlayerType::DqnAi,
                "PG" => PlayerType::PgAi,
                _ => PlayerType::Human,
            },
        }
    }

    fn open_player_menu(&mut self, target: Player) {
        let current = self.current_player_type(target);
        let cursor = PlayerType::ALL
            .iter()
            .position(|&pt| pt == current)
            .unwrap_or(0);
        self.mode = AppMode::SelectingPlayer { target, cursor };
    }

    fn create_player_kind(&mut self, player_type: PlayerType) -> PlayerKind {
        match player_type {
            PlayerType::Human => PlayerKind::Human,
            PlayerType::RandomAi => PlayerKind::Ai(Box::new(RandomAgent::new())),
            PlayerType::DqnAi => {
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
            }
            PlayerType::PgAi => {
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
                                let msg =
                                    format!("PG loaded (episode {})", data.metadata.episode);
                                (a, msg)
                            }
                            Err(_) => (a, "PG (failed to load checkpoint)".to_string()),
                        }
                    }
                    Err(_) => (agent, "PG (untrained, no checkpoint)".to_string()),
                };
                self.message = Some(load_msg);
                PlayerKind::Ai(Box::new(boxed_agent))
            }
        }
    }

    fn set_player_type(&mut self, target: Player, player_type: PlayerType) {
        let current = self.current_player_type(target);
        if current == player_type {
            return;
        }
        let kind = self.create_player_kind(player_type);
        self.set_player_slot(target, kind);
        self.reset_game_after_toggle();
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

    #[cfg(test)]
    fn toggle_random_for(&mut self, target: Player) {
        let is_ai = matches!(self.player_slot(target), PlayerKind::Ai(_));
        if is_ai {
            self.set_player_type(target, PlayerType::Human);
        } else {
            self.set_player_type(target, PlayerType::RandomAi);
        }
    }

    #[cfg(test)]
    fn toggle_dqn_for(&mut self, target: Player) {
        let is_ai = matches!(self.player_slot(target), PlayerKind::Ai(_));
        if is_ai {
            self.set_player_type(target, PlayerType::Human);
        } else {
            self.set_player_type(target, PlayerType::DqnAi);
        }
    }

    #[cfg(test)]
    fn toggle_pg_for(&mut self, target: Player) {
        let is_ai = matches!(self.player_slot(target), PlayerKind::Ai(_));
        if is_ai {
            self.set_player_type(target, PlayerType::Human);
        } else {
            self.set_player_type(target, PlayerType::PgAi);
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
        let menu_state = match self.mode {
            AppMode::SelectingPlayer { target, cursor } => {
                Some(super::game_view::MenuRenderState {
                    target,
                    cursor,
                    current_type: self.current_player_type(target),
                })
            }
            AppMode::Playing => None,
        };
        super::game_view::render(
            frame,
            &self.game_state,
            self.selected_column,
            &self.message,
            &mode,
            self.is_ai_vs_ai(),
            self.paused,
            menu_state,
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

    // --- Player Selection Menu Tests ---

    #[test]
    fn open_player_menu_sets_selecting_mode() {
        let mut app = App::new();
        app.open_player_menu(Player::Red);
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer {
                target: Player::Red,
                ..
            }
        ));
    }

    #[test]
    fn open_player_menu_preselects_current_type() {
        let mut app = App::new();
        // Default is Human, which is index 0
        app.open_player_menu(Player::Yellow);
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 0, .. }
        ));

        // Set to Random AI, then open menu — cursor should be at index 1
        app.mode = AppMode::Playing;
        app.set_player_type(Player::Yellow, PlayerType::RandomAi);
        app.open_player_menu(Player::Yellow);
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 1, .. }
        ));
    }

    #[test]
    fn menu_navigation_up_down() {
        let mut app = App::new();
        app.open_player_menu(Player::Red);
        // Starts at cursor 0 (Human)
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 0, .. }
        ));

        // Down moves to 1
        app.handle_key(KeyEvent::from(KeyCode::Down));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 1, .. }
        ));

        // Down again moves to 2
        app.handle_key(KeyEvent::from(KeyCode::Down));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 2, .. }
        ));

        // Up goes back to 1
        app.handle_key(KeyEvent::from(KeyCode::Up));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 1, .. }
        ));
    }

    #[test]
    fn menu_cursor_clamps_at_bounds() {
        let mut app = App::new();
        app.open_player_menu(Player::Red);
        // Cursor at 0, Up should stay at 0
        app.handle_key(KeyEvent::from(KeyCode::Up));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 0, .. }
        ));

        // Move to last item (index 3)
        app.handle_key(KeyEvent::from(KeyCode::Down));
        app.handle_key(KeyEvent::from(KeyCode::Down));
        app.handle_key(KeyEvent::from(KeyCode::Down));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 3, .. }
        ));

        // Down should stay at 3
        app.handle_key(KeyEvent::from(KeyCode::Down));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer { cursor: 3, .. }
        ));
    }

    #[test]
    fn menu_esc_cancels_without_change() {
        let mut app = App::new();
        assert!(matches!(app.yellow_player, PlayerKind::Human));

        app.open_player_menu(Player::Yellow);
        // Move cursor to Random AI
        app.handle_key(KeyEvent::from(KeyCode::Down));
        // Cancel with Esc
        app.handle_key(KeyEvent::from(KeyCode::Esc));

        assert_eq!(app.mode, AppMode::Playing);
        assert!(matches!(app.yellow_player, PlayerKind::Human));
    }

    #[test]
    fn menu_enter_selects_random_ai() {
        let mut app = App::new();
        assert!(matches!(app.yellow_player, PlayerKind::Human));

        app.open_player_menu(Player::Yellow);
        // Move to Random AI (index 1)
        app.handle_key(KeyEvent::from(KeyCode::Down));
        // Select
        app.handle_key(KeyEvent::from(KeyCode::Enter));

        assert_eq!(app.mode, AppMode::Playing);
        assert!(matches!(app.yellow_player, PlayerKind::Ai(_)));
        assert_eq!(app.yellow_player.label(), "Random");
    }

    #[test]
    fn menu_select_same_type_is_noop() {
        let mut app = App::new();
        // Make a move so game state is not initial
        app.game_state.apply_move_mut(3).unwrap();
        let board_before = app.game_state.board().clone();

        // Open menu for Yellow (Human) and select Human (same type)
        app.open_player_menu(Player::Yellow);
        app.handle_key(KeyEvent::from(KeyCode::Enter)); // cursor=0 = Human

        assert_eq!(app.mode, AppMode::Playing);
        // Board should NOT have been reset
        assert_eq!(*app.game_state.board(), board_before);
    }

    #[test]
    fn set_player_type_resets_game() {
        let mut app = App::new();
        // Make a move
        app.game_state.apply_move_mut(3).unwrap();
        assert_eq!(app.game_state.current_player(), Player::Yellow);

        app.set_player_type(Player::Yellow, PlayerType::RandomAi);
        // Game should have reset
        assert_eq!(app.game_state.current_player(), Player::Red);
    }

    #[test]
    fn hotkeys_1_and_2_open_menus() {
        let mut app = App::new();

        app.handle_key(KeyEvent::from(KeyCode::Char('1')));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer {
                target: Player::Red,
                ..
            }
        ));

        // Cancel to return to Playing
        app.handle_key(KeyEvent::from(KeyCode::Esc));
        assert_eq!(app.mode, AppMode::Playing);

        app.handle_key(KeyEvent::from(KeyCode::Char('2')));
        assert!(matches!(
            app.mode,
            AppMode::SelectingPlayer {
                target: Player::Yellow,
                ..
            }
        ));
    }

    #[test]
    fn game_keys_ignored_while_menu_open() {
        let mut app = App::new();
        app.selected_column = 3;
        app.open_player_menu(Player::Red);

        // Arrow keys should NOT move column while in menu
        app.handle_key(KeyEvent::from(KeyCode::Left));
        assert_eq!(app.selected_column, 3);

        app.handle_key(KeyEvent::from(KeyCode::Right));
        assert_eq!(app.selected_column, 3);

        // Enter should NOT drop a piece — it selects the menu item
        let player_before = app.game_state.current_player();
        app.handle_key(KeyEvent::from(KeyCode::Enter));
        assert_eq!(app.game_state.current_player(), player_before);
    }

    #[test]
    fn current_player_type_maps_correctly() {
        let mut app = App::new();
        assert_eq!(app.current_player_type(Player::Red), PlayerType::Human);

        app.red_player = PlayerKind::Ai(Box::new(RandomAgent::new()));
        assert_eq!(app.current_player_type(Player::Red), PlayerType::RandomAi);
    }
}
