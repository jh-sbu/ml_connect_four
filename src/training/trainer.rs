use std::path::PathBuf;

use crate::ai::algorithms::DqnAgent;
use crate::ai::{Agent, Experience, RandomAgent};
use crate::checkpoint::{CheckpointManager, CheckpointManagerConfig, CheckpointMetrics};
use crate::game::{GameOutcome, GameState, Player};
use crate::training::metrics::{EpisodeResult, TrainingMetrics};

/// Trainer configuration.
pub struct TrainerConfig {
    pub num_episodes: usize,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub eval_games: usize,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: PathBuf,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            num_episodes: 10_000,
            log_interval: 100,
            eval_interval: 500,
            eval_games: 100,
            checkpoint_interval: 1000,
            checkpoint_dir: PathBuf::from("checkpoints"),
        }
    }
}

/// Self-play trainer for DQN agents.
pub struct Trainer {
    config: TrainerConfig,
    checkpoint_manager: CheckpointManager,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let checkpoint_manager = CheckpointManager::new(CheckpointManagerConfig {
            checkpoint_dir: config.checkpoint_dir.clone(),
            ..Default::default()
        });
        Trainer {
            config,
            checkpoint_manager,
        }
    }

    /// Run the full training loop.
    pub fn train(&self, agent: &mut DqnAgent) {
        let mut metrics = TrainingMetrics::new();

        let start_episode = agent.episode_count() + 1;
        let end_episode = agent.episode_count() + self.config.num_episodes;

        println!(
            "Starting DQN training for {} episodes (episodes {}..{})...",
            self.config.num_episodes, start_episode, end_episode
        );
        println!("-------------------------------------------");

        for episode in start_episode..=end_episode {
            let (experiences, result) = self.play_episode(agent);
            let update_metrics = agent.batch_update(&experiences);

            if update_metrics.loss > 0.0 {
                metrics.record_update(update_metrics.loss);
            }
            metrics.record_episode(result);

            if episode % self.config.log_interval == 0 {
                let window = self.config.log_interval;
                println!(
                    "Episode {}/{} | eps: {:.3} | loss: {:.4} | win_rate({}): {:.1}% | draw: {:.1}% | avg_len: {:.1}",
                    episode,
                    end_episode,
                    agent.epsilon(),
                    metrics.average_loss(window),
                    window,
                    metrics.win_rate(window) * 100.0,
                    metrics.draw_rate(window) * 100.0,
                    metrics.average_game_length(window),
                );
            }

            if episode % self.config.eval_interval == 0 {
                let eval_wr = self.evaluate(agent);
                println!(
                    "  >> Eval vs Random ({} games): {:.1}% win rate",
                    self.config.eval_games,
                    eval_wr * 100.0
                );
            }

            if episode % self.config.checkpoint_interval == 0 {
                let eval_wr = self.evaluate(agent);
                let window = self.config.log_interval;
                let ckpt_metrics = CheckpointMetrics {
                    win_rate: eval_wr,
                    draw_rate: metrics.draw_rate(window),
                    average_game_length: metrics.average_game_length(window),
                    current_loss: metrics.average_loss(window),
                    training_steps: agent.step_count(),
                };
                match self
                    .checkpoint_manager
                    .save_checkpoint(agent, &ckpt_metrics, episode)
                {
                    Ok(path) => println!("  >> Checkpoint saved: {}", path.display()),
                    Err(e) => eprintln!("  >> Checkpoint failed: {}", e),
                }
            }
        }

        println!("-------------------------------------------");
        println!("Training complete. Total episodes: {}", metrics.total_episodes());

        let final_wr = self.evaluate(agent);
        println!("Final eval vs Random: {:.1}% win rate", final_wr * 100.0);
    }

    /// Play one self-play episode. Agent plays both sides.
    /// Returns (experiences, episode_result).
    fn play_episode(&self, agent: &mut DqnAgent) -> (Vec<Experience>, EpisodeResult) {
        let mut state = GameState::initial();
        let mut move_records: Vec<(GameState, usize, Player)> = Vec::new();

        while !state.is_terminal() {
            let player = state.current_player();
            let action = agent.select_action(&state, true);
            move_records.push((state.clone(), action, player));
            state = state.apply_move(action).expect("Agent selected illegal action");
        }

        let outcome = state.outcome().expect("Game should be terminal");
        let game_length = move_records.len();

        // Build experiences with sparse rewards
        let mut experiences = Vec::with_capacity(game_length);
        for (i, (move_state, action, player)) in move_records.iter().enumerate() {
            let is_last = i == game_length - 1;
            let next_state = if i + 1 < game_length {
                move_records[i + 1].0.clone()
            } else {
                state.clone()
            };

            let reward = if is_last {
                match &outcome {
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
                // The second-to-last move: the *other* player's last move won/drew.
                // Give this player a reward too.
                match &outcome {
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

        let winner = match outcome {
            GameOutcome::Winner(p) => Some(p),
            GameOutcome::Draw => None,
        };

        (experiences, EpisodeResult { winner, game_length })
    }

    /// Evaluate the agent against RandomAgent over `eval_games`, alternating first player.
    pub fn evaluate(&self, agent: &mut DqnAgent) -> f32 {
        let mut random = RandomAgent::new();
        let mut wins = 0;
        let mut total = 0;

        let saved_epsilon = agent.epsilon();
        agent.set_epsilon(0.0); // greedy for evaluation

        for game_idx in 0..self.config.eval_games {
            let agent_is_red = game_idx % 2 == 0;
            let mut state = GameState::initial();

            while !state.is_terminal() {
                let is_agent_turn = (state.current_player() == Player::Red) == agent_is_red;
                let action = if is_agent_turn {
                    agent.select_action(&state, false)
                } else {
                    random.select_action(&state, false)
                };
                state = state.apply_move(action).expect("Illegal move during eval");
            }

            if let Some(GameOutcome::Winner(winner)) = state.outcome() {
                let agent_won = (winner == Player::Red) == agent_is_red;
                if agent_won {
                    wins += 1;
                }
            }
            total += 1;
        }

        agent.set_epsilon(saved_epsilon); // restore
        wins as f32 / total as f32
    }
}
