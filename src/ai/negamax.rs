use crate::game::{Board, GameState, Player, COLS, ROWS};

use super::agent::Agent;

/// Trait for evaluating a board position from a player's perspective.
pub trait Heuristic: Send {
    fn evaluate(&self, board: &Board, player: Player) -> f64;
}

/// Default heuristic that scans all 4-cell windows and scores threats.
pub struct ConnectFourHeuristic;

impl ConnectFourHeuristic {
    fn score_window(own: usize, opp: usize, empty: usize) -> f64 {
        if own == 3 && empty == 1 {
            50.0
        } else if own == 2 && empty == 2 {
            10.0
        } else if opp == 3 && empty == 1 {
            -80.0
        } else if opp == 2 && empty == 2 {
            -10.0
        } else {
            0.0
        }
    }
}

impl Heuristic for ConnectFourHeuristic {
    fn evaluate(&self, board: &Board, player: Player) -> f64 {
        let own_cell = player.to_cell();
        let opp_cell = player.other().to_cell();
        let mut score = 0.0;

        // Center column bonus
        for row in 0..ROWS {
            let cell = board.get(row, 3);
            if cell == own_cell {
                score += 3.0;
            } else if cell == opp_cell {
                score -= 3.0;
            }
        }

        // Scan all 4-cell windows

        // Horizontal
        for row in 0..ROWS {
            for col in 0..COLS - 3 {
                let mut own = 0;
                let mut opp = 0;
                let mut empty = 0;
                for i in 0..4 {
                    match board.get(row, col + i) {
                        c if c == own_cell => own += 1,
                        c if c == opp_cell => opp += 1,
                        _ => empty += 1,
                    }
                }
                score += Self::score_window(own, opp, empty);
            }
        }

        // Vertical
        for col in 0..COLS {
            for row in 0..ROWS - 3 {
                let mut own = 0;
                let mut opp = 0;
                let mut empty = 0;
                for i in 0..4 {
                    match board.get(row + i, col) {
                        c if c == own_cell => own += 1,
                        c if c == opp_cell => opp += 1,
                        _ => empty += 1,
                    }
                }
                score += Self::score_window(own, opp, empty);
            }
        }

        // Diagonal (top-left to bottom-right)
        for row in 0..ROWS - 3 {
            for col in 0..COLS - 3 {
                let mut own = 0;
                let mut opp = 0;
                let mut empty = 0;
                for i in 0..4 {
                    match board.get(row + i, col + i) {
                        c if c == own_cell => own += 1,
                        c if c == opp_cell => opp += 1,
                        _ => empty += 1,
                    }
                }
                score += Self::score_window(own, opp, empty);
            }
        }

        // Diagonal (bottom-left to top-right)
        for row in 3..ROWS {
            for col in 0..COLS - 3 {
                let mut own = 0;
                let mut opp = 0;
                let mut empty = 0;
                for i in 0..4 {
                    match board.get(row - i, col + i) {
                        c if c == own_cell => own += 1,
                        c if c == opp_cell => opp += 1,
                        _ => empty += 1,
                    }
                }
                score += Self::score_window(own, opp, empty);
            }
        }

        score
    }
}

/// Column ordering: center-first for better alpha-beta pruning.
const MOVE_ORDER: [usize; 7] = [3, 2, 4, 1, 5, 0, 6];

/// Negamax agent with alpha-beta pruning.
pub struct NegamaxAgent {
    depth: usize,
    heuristic: Box<dyn Heuristic>,
}

impl NegamaxAgent {
    pub fn new(depth: usize) -> Self {
        NegamaxAgent {
            depth,
            heuristic: Box::new(ConnectFourHeuristic),
        }
    }

    pub fn with_heuristic(depth: usize, heuristic: Box<dyn Heuristic>) -> Self {
        NegamaxAgent { depth, heuristic }
    }

    fn best_move(&mut self, state: &GameState) -> usize {
        let legal = state.legal_actions();
        assert!(!legal.is_empty(), "No legal actions available");

        let mut best_action = legal[0];
        let mut best_score = f64::NEG_INFINITY;

        for &col in &MOVE_ORDER {
            if !legal.contains(&col) {
                continue;
            }
            let next = state.apply_move(col).unwrap();
            // Negamax: opponent's score is negated
            let score = -self.negamax(&next, self.depth - 1, f64::NEG_INFINITY, f64::INFINITY);
            if score > best_score {
                best_score = score;
                best_action = col;
            }
        }

        best_action
    }

    fn negamax(&self, state: &GameState, depth: usize, mut alpha: f64, beta: f64) -> f64 {
        // Terminal check
        if state.is_terminal() {
            return match state.outcome() {
                Some(crate::game::GameOutcome::Winner(_)) => {
                    // The winner is the player who just moved (opponent of current_player).
                    // From current_player's perspective, this is a loss.
                    -100_000.0
                }
                Some(crate::game::GameOutcome::Draw) => 0.0,
                None => unreachable!(),
            };
        }

        if depth == 0 {
            return self
                .heuristic
                .evaluate(state.board(), state.current_player());
        }

        let legal = state.legal_actions();
        let mut best = f64::NEG_INFINITY;

        for &col in &MOVE_ORDER {
            if !legal.contains(&col) {
                continue;
            }
            let next = state.apply_move(col).unwrap();
            let score = -self.negamax(&next, depth - 1, -beta, -alpha);
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }

        best
    }
}

impl Agent for NegamaxAgent {
    fn select_action(&mut self, state: &GameState, _training: bool) -> usize {
        self.best_move(state)
    }

    fn name(&self) -> &str {
        "Negamax"
    }

    fn clone_agent(&self) -> Box<dyn Agent> {
        Box::new(NegamaxAgent::new(self.depth))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::RandomAgent;
    use crate::game::{Cell, GameOutcome};

    // --- Heuristic tests ---

    #[test]
    fn heuristic_empty_board_is_zero() {
        let board = Board::new();
        let h = ConnectFourHeuristic;
        let score_red = h.evaluate(&board, Player::Red);
        let score_yellow = h.evaluate(&board, Player::Yellow);
        assert!(
            (score_red - 0.0).abs() < f64::EPSILON,
            "Empty board should be 0 for Red, got {score_red}"
        );
        assert!(
            (score_yellow - 0.0).abs() < f64::EPSILON,
            "Empty board should be 0 for Yellow, got {score_yellow}"
        );
    }

    #[test]
    fn heuristic_symmetric_scores() {
        let board = Board::new();
        let h = ConnectFourHeuristic;
        // Symmetric for both players on empty board
        assert_eq!(h.evaluate(&board, Player::Red), h.evaluate(&board, Player::Yellow));
    }

    #[test]
    fn heuristic_center_preference() {
        let h = ConnectFourHeuristic;
        // Board with one red piece in center
        let mut board_center = Board::new();
        board_center.drop_piece(3, Cell::Red).unwrap();
        // Board with one red piece on edge
        let mut board_edge = Board::new();
        board_edge.drop_piece(0, Cell::Red).unwrap();

        let score_center = h.evaluate(&board_center, Player::Red);
        let score_edge = h.evaluate(&board_edge, Player::Red);
        assert!(
            score_center > score_edge,
            "Center ({score_center}) should score higher than edge ({score_edge})"
        );
    }

    #[test]
    fn heuristic_three_in_a_row_scores_high() {
        let h = ConnectFourHeuristic;
        let mut board = Board::new();
        board.drop_piece(0, Cell::Red).unwrap();
        board.drop_piece(1, Cell::Red).unwrap();
        board.drop_piece(2, Cell::Red).unwrap();
        // 3 red in a row with col 3 empty = a threat
        let score = h.evaluate(&board, Player::Red);
        assert!(score > 40.0, "3-in-a-row should score high, got {score}");
    }

    // --- Algorithm tests ---

    #[test]
    fn selects_legal_action() {
        let mut agent = NegamaxAgent::new(4);
        let state = GameState::initial();
        let legal = state.legal_actions();
        let action = agent.select_action(&state, false);
        assert!(legal.contains(&action), "Action {action} is not legal");
    }

    #[test]
    fn takes_winning_move() {
        // Set up: Red has 3 in a row at bottom, col 3 wins
        let mut state = GameState::initial();
        // Red col0, Yellow col0, Red col1, Yellow col1, Red col2, Yellow col2
        for col in 0..3 {
            state = state.apply_move(col).unwrap(); // Red
            state = state.apply_move(col).unwrap(); // Yellow
        }
        // Now Red to move, col 3 completes horizontal win
        let mut agent = NegamaxAgent::new(4);
        let action = agent.select_action(&state, false);
        assert_eq!(action, 3, "Should take winning move at col 3");
    }

    #[test]
    fn blocks_opponent_win() {
        // Set up: Yellow has 3 in a row at bottom, Red must block col 3
        let mut state = GameState::initial();
        // Red col6, Yellow col0, Red col6, Yellow col1, Red col6, Yellow col2
        // Now Red's turn — Yellow threatens col 3 for horizontal win
        state = state.apply_move(6).unwrap(); // Red
        state = state.apply_move(0).unwrap(); // Yellow
        state = state.apply_move(6).unwrap(); // Red
        state = state.apply_move(1).unwrap(); // Yellow
        state = state.apply_move(5).unwrap(); // Red
        state = state.apply_move(2).unwrap(); // Yellow
        // Yellow has [0,1,2] at bottom row. Red must play col 3 to block.
        let mut agent = NegamaxAgent::new(4);
        let action = agent.select_action(&state, false);
        assert_eq!(action, 3, "Should block opponent's winning move at col 3");
    }

    #[test]
    fn prefers_win_over_block() {
        // Set up where Red can win AND Yellow threatens — Red should take the win
        let mut state = GameState::initial();
        // Build Red 3-in-a-row at bottom cols 0,1,2 and Yellow 3-in-a-row at row above
        // Red col0, Yellow col0, Red col1, Yellow col1, Red col2, Yellow col2
        // Red has bottom row 0,1,2. Yellow has second row 0,1,2.
        // Both threaten col 3. Red should take the win.
        for col in 0..3 {
            state = state.apply_move(col).unwrap(); // Red (bottom)
            state = state.apply_move(col).unwrap(); // Yellow (second row)
        }
        let mut agent = NegamaxAgent::new(4);
        let action = agent.select_action(&state, false);
        assert_eq!(action, 3, "Should prefer winning move over blocking");
    }

    // --- Integration tests ---

    #[test]
    fn full_game_vs_self_completes() {
        let mut agent1 = NegamaxAgent::new(4);
        let mut agent2 = NegamaxAgent::new(4);
        let mut state = GameState::initial();
        let mut turn = 0;

        while !state.is_terminal() && turn < 42 {
            let action = if turn % 2 == 0 {
                agent1.select_action(&state, false)
            } else {
                agent2.select_action(&state, false)
            };
            state = state.apply_move(action).unwrap();
            turn += 1;
        }

        assert!(state.is_terminal(), "Game should complete");
        assert!(state.outcome().is_some());
    }

    #[test]
    fn beats_random_agent() {
        let games_per_color = 20;
        let mut negamax_wins = 0;
        let total = games_per_color * 2;

        // Negamax plays as Red (first)
        for _ in 0..games_per_color {
            let mut negamax = NegamaxAgent::new(5);
            let mut random = RandomAgent::new();
            let mut state = GameState::initial();
            let mut turn = 0;

            while !state.is_terminal() {
                let action = if turn % 2 == 0 {
                    negamax.select_action(&state, false)
                } else {
                    random.select_action(&state, false)
                };
                state = state.apply_move(action).unwrap();
                turn += 1;
            }

            if state.outcome() == Some(GameOutcome::Winner(Player::Red)) {
                negamax_wins += 1;
            }
        }

        // Negamax plays as Yellow (second)
        for _ in 0..games_per_color {
            let mut random = RandomAgent::new();
            let mut negamax = NegamaxAgent::new(5);
            let mut state = GameState::initial();
            let mut turn = 0;

            while !state.is_terminal() {
                let action = if turn % 2 == 0 {
                    random.select_action(&state, false)
                } else {
                    negamax.select_action(&state, false)
                };
                state = state.apply_move(action).unwrap();
                turn += 1;
            }

            if state.outcome() == Some(GameOutcome::Winner(Player::Yellow)) {
                negamax_wins += 1;
            }
        }

        let win_rate = negamax_wins as f64 / total as f64;
        assert!(
            win_rate > 0.80,
            "Negamax should beat random >80% of the time, got {:.0}% ({negamax_wins}/{total})",
            win_rate * 100.0
        );
    }

    // --- Agent trait tests ---

    #[test]
    fn name_is_negamax() {
        let agent = NegamaxAgent::new(7);
        assert_eq!(agent.name(), "Negamax");
    }

    #[test]
    fn clone_agent_works() {
        let agent = NegamaxAgent::new(7);
        let cloned = agent.clone_agent();
        assert_eq!(cloned.name(), "Negamax");
    }
}
