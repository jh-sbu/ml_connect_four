use std::sync::mpsc;

use crate::ai::{Agent, Experience, RandomAgent, TrainableAgent};
use crate::game::{GameOutcome, GameState, Player};
use crate::training::dashboard_msg::{LiveGameState, TrainingUpdate};
use crate::training::metrics::EpisodeResult;

/// Result of playing a single self-play episode.
pub struct EpisodeTrace {
    pub experiences: Vec<Experience>,
    pub result: EpisodeResult,
}

/// Play one self-play episode. Agent plays both sides.
pub fn play_self_play_episode(agent: &mut dyn Agent) -> EpisodeTrace {
    let mut state = GameState::initial();
    let mut move_records: Vec<(GameState, usize, Player)> = Vec::new();

    while !state.is_terminal() {
        let player = state.current_player();
        let action = agent.select_action(&state, true);
        move_records.push((state.clone(), action, player));
        state = state.apply_move(action).unwrap_or_else(|_| {
            panic!(
                "Agent selected illegal action {} (legal: {:?})",
                action,
                state.legal_actions()
            )
        });
    }

    let outcome = state
        .outcome()
        .expect("terminal state must have an outcome");
    let experiences = build_experiences(&move_records, &state, &outcome);
    let game_length = move_records.len();

    let winner = match outcome {
        GameOutcome::Winner(p) => Some(p),
        GameOutcome::Draw => None,
    };

    EpisodeTrace {
        experiences,
        result: EpisodeResult { winner, game_length },
    }
}

/// Play one self-play episode, sending live game updates via channel.
pub fn play_self_play_episode_live(
    agent: &mut dyn Agent,
    tx: &mpsc::Sender<TrainingUpdate>,
    live_update_interval: usize,
) -> EpisodeTrace {
    let mut state = GameState::initial();
    let mut move_records: Vec<(GameState, usize, Player)> = Vec::new();
    let mut move_number = 0;

    while !state.is_terminal() {
        let player = state.current_player();
        let action = agent.select_action(&state, true);
        move_records.push((state.clone(), action, player));
        state = state.apply_move(action).unwrap_or_else(|_| {
            panic!(
                "Agent selected illegal action {} (legal: {:?})",
                action,
                state.legal_actions()
            )
        });
        move_number += 1;

        if move_number % live_update_interval == 0 {
            let _ = tx.send(TrainingUpdate::LiveGame(LiveGameState {
                game_state: state.clone(),
                move_number,
            }));
        }
    }

    // Send final state
    let _ = tx.send(TrainingUpdate::LiveGame(LiveGameState {
        game_state: state.clone(),
        move_number,
    }));

    let outcome = state
        .outcome()
        .expect("terminal state must have an outcome");
    let experiences = build_experiences(&move_records, &state, &outcome);
    let game_length = move_records.len();

    let winner = match outcome {
        GameOutcome::Winner(p) => Some(p),
        GameOutcome::Draw => None,
    };

    EpisodeTrace {
        experiences,
        result: EpisodeResult { winner, game_length },
    }
}

/// Build experiences with sparse rewards from move records and final state.
pub fn build_experiences(
    move_records: &[(GameState, usize, Player)],
    final_state: &GameState,
    outcome: &GameOutcome,
) -> Vec<Experience> {
    let game_length = move_records.len();
    let mut experiences = Vec::with_capacity(game_length);

    for (i, (move_state, action, player)) in move_records.iter().enumerate() {
        let is_last = i == game_length - 1;
        let next_state = if i + 1 < game_length {
            move_records[i + 1].0.clone()
        } else {
            final_state.clone()
        };

        let reward = if is_last {
            match outcome {
                GameOutcome::Winner(winner) => {
                    if winner == player {
                        1.0
                    } else {
                        -1.0
                    }
                }
                GameOutcome::Draw => 0.0,
            }
        } else if i + 2 == game_length {
            match outcome {
                GameOutcome::Winner(winner) => {
                    if winner == player {
                        1.0
                    } else {
                        -1.0
                    }
                }
                GameOutcome::Draw => 0.0,
            }
        } else {
            0.0
        };

        experiences.push(Experience {
            state: move_state.clone(),
            action: *action,
            reward,
            next_state,
            done: is_last,
            player: *player,
        });
    }

    experiences
}

/// Play a single evaluation game between two agents.
/// Returns Some(true) if agent won, Some(false) if agent lost, None if draw.
pub fn play_eval_game(
    agent: &mut dyn Agent,
    opponent: &mut dyn Agent,
    agent_is_red: bool,
) -> Option<bool> {
    let mut state = GameState::initial();

    while !state.is_terminal() {
        let is_agent_turn = (state.current_player() == Player::Red) == agent_is_red;
        let action = if is_agent_turn {
            agent.select_action(&state, false)
        } else {
            opponent.select_action(&state, false)
        };
        state = state.apply_move(action).unwrap_or_else(|_| {
            panic!(
                "illegal move {} during eval (legal: {:?})",
                action,
                state.legal_actions()
            )
        });
    }

    if let Some(GameOutcome::Winner(winner)) = state.outcome() {
        let agent_won = (winner == Player::Red) == agent_is_red;
        Some(agent_won)
    } else {
        None
    }
}

/// Evaluate agent vs random over N games, alternating sides.
pub fn evaluate(agent: &mut dyn TrainableAgent, eval_games: usize) -> f32 {
    let mut random = RandomAgent::new();
    let mut wins = 0;
    let mut total = 0;

    let eval_state = agent.enter_eval_mode();

    for game_idx in 0..eval_games {
        let agent_is_red = game_idx % 2 == 0;
        if let Some(true) = play_eval_game(agent, &mut random, agent_is_red) {
            wins += 1;
        }
        total += 1;
    }

    agent.exit_eval_mode(eval_state);
    wins as f32 / total as f32
}

/// Derive a deterministic seed for a given episode index.
pub fn episode_seed(base_seed: u64, episode_index: usize) -> u64 {
    // FNV-1a-inspired mixing for deterministic, well-distributed seeds
    let mut hash = base_seed ^ 0x517cc1b727220a95;
    let index = episode_index as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    hash ^= index;
    hash = hash.wrapping_mul(0x100000001b3);
    hash ^= index >> 32;
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_play_self_play_episode_terminates() {
        let mut agent = RandomAgent::new();
        let trace = play_self_play_episode(&mut agent);
        assert!(!trace.experiences.is_empty());
        assert!(trace.result.game_length > 0);
    }

    #[test]
    fn test_build_experiences_rewards() {
        let s0 = GameState::initial();
        let s1 = s0.apply_move(0).unwrap();
        let s2 = s1.apply_move(1).unwrap();
        let s3 = s2.apply_move(0).unwrap();
        let s4 = s3.apply_move(1).unwrap();
        let s5 = s4.apply_move(0).unwrap();
        let s6 = s5.apply_move(1).unwrap();
        let s7 = s6.apply_move(0).unwrap(); // Red wins (4 in a row in col 0)

        let move_records = vec![
            (s0.clone(), 0, Player::Red),
            (s1.clone(), 1, Player::Yellow),
            (s2.clone(), 0, Player::Red),
            (s3.clone(), 1, Player::Yellow),
            (s4.clone(), 0, Player::Red),
            (s5.clone(), 1, Player::Yellow),
            (s6.clone(), 0, Player::Red),
        ];

        let outcome = s7.outcome().unwrap();
        let experiences = build_experiences(&move_records, &s7, &outcome);
        assert_eq!(experiences.len(), 7);

        // Last move (Red): winner gets +1
        assert_eq!(experiences[6].reward, 1.0);
        // Second to last (Yellow): loser gets -1
        assert_eq!(experiences[5].reward, -1.0);
        // Earlier moves: 0
        assert_eq!(experiences[0].reward, 0.0);
    }

    #[test]
    fn test_play_eval_game_terminates() {
        let mut agent = RandomAgent::new();
        let mut opponent = RandomAgent::new();
        let _result = play_eval_game(&mut agent, &mut opponent, true);
        // Just verify it doesn't hang or panic
    }

    #[test]
    fn test_episode_seed_deterministic() {
        let s1 = episode_seed(42, 100);
        let s2 = episode_seed(42, 100);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_episode_seed_varies() {
        let s1 = episode_seed(42, 0);
        let s2 = episode_seed(42, 1);
        let s3 = episode_seed(42, 2);
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
        assert_ne!(s1, s3);

        // Different base seeds
        let s4 = episode_seed(1, 0);
        let s5 = episode_seed(2, 0);
        assert_ne!(s4, s5);
    }
}
