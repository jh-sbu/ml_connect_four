use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::ai::algorithms::DqnAgent;
use crate::ai::{Agent, Experience, RandomAgent};
use crate::checkpoint::{CheckpointManager, CheckpointManagerConfig, CheckpointMetrics};
use crate::game::{GameOutcome, GameState, Player};
use crate::training::dashboard_msg::{
    LiveGameState, MetricsSnapshot, TrainingCommand, TrainingUpdate,
};
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

    /// Run the full training loop (headless, stdout output).
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
        println!(
            "Training complete. Total episodes: {}",
            metrics.total_episodes()
        );

        let final_wr = self.evaluate(agent);
        println!("Final eval vs Random: {:.1}% win rate", final_wr * 100.0);
    }

    /// Run the training loop with dashboard communication via channels.
    pub fn train_with_dashboard(
        &self,
        agent: &mut DqnAgent,
        tx: mpsc::Sender<TrainingUpdate>,
        cmd_rx: mpsc::Receiver<TrainingCommand>,
        pause: Arc<AtomicBool>,
        quit: Arc<AtomicBool>,
    ) {
        let mut metrics = TrainingMetrics::new();

        let start_episode = agent.episode_count() + 1;
        let end_episode = agent.episode_count() + self.config.num_episodes;

        for episode in start_episode..=end_episode {
            // Check quit
            if quit.load(Ordering::Relaxed) {
                break;
            }

            // Check pause - spin-sleep
            while pause.load(Ordering::Relaxed) {
                if quit.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if quit.load(Ordering::Relaxed) {
                break;
            }

            // Check for commands
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    TrainingCommand::SaveCheckpoint => {
                        self.save_checkpoint_with_tx(agent, &metrics, episode, &tx);
                    }
                }
            }

            // Play episode with live game updates
            let (experiences, result) =
                self.play_episode_with_live(agent, &tx);
            let update_metrics = agent.batch_update(&experiences);

            if update_metrics.loss > 0.0 {
                metrics.record_update(update_metrics.loss);
            }
            metrics.record_episode(result);

            // Send metrics every log_interval episodes
            if episode % self.config.log_interval == 0 {
                let window = self.config.log_interval;
                let snap = MetricsSnapshot {
                    episode,
                    total_episodes: end_episode,
                    epsilon: agent.epsilon(),
                    win_rate: metrics.win_rate(window),
                    draw_rate: metrics.draw_rate(window),
                    loss: metrics.average_loss(window),
                    avg_game_length: metrics.average_game_length(window),
                    step_count: agent.step_count(),
                };
                let _ = tx.send(TrainingUpdate::Metrics(snap));
            }

            // Eval
            if episode % self.config.eval_interval == 0 {
                let eval_wr = self.evaluate(agent);
                let _ = tx.send(TrainingUpdate::EvalResult {
                    episode,
                    win_rate: eval_wr,
                });
            }

            // Checkpoint
            if episode % self.config.checkpoint_interval == 0 {
                self.save_checkpoint_with_tx(agent, &metrics, episode, &tx);
            }
        }

        let _ = tx.send(TrainingUpdate::Finished);
    }

    fn save_checkpoint_with_tx(
        &self,
        agent: &mut DqnAgent,
        metrics: &TrainingMetrics,
        episode: usize,
        tx: &mpsc::Sender<TrainingUpdate>,
    ) {
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
            Ok(path) => {
                let _ = tx.send(TrainingUpdate::CheckpointSaved {
                    episode,
                    path,
                });
            }
            Err(_) => {}
        }
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
        let experiences = Self::build_experiences(&move_records, &state, &outcome);
        let game_length = move_records.len();

        let winner = match outcome {
            GameOutcome::Winner(p) => Some(p),
            GameOutcome::Draw => None,
        };

        (experiences, EpisodeResult { winner, game_length })
    }

    /// Play one self-play episode, sending live game updates via channel.
    fn play_episode_with_live(
        &self,
        agent: &mut DqnAgent,
        tx: &mpsc::Sender<TrainingUpdate>,
    ) -> (Vec<Experience>, EpisodeResult) {
        let mut state = GameState::initial();
        let mut move_records: Vec<(GameState, usize, Player)> = Vec::new();
        let mut move_number = 0;

        while !state.is_terminal() {
            let player = state.current_player();
            let action = agent.select_action(&state, true);
            move_records.push((state.clone(), action, player));
            state = state.apply_move(action).expect("Agent selected illegal action");
            move_number += 1;

            // Send live game update every 4 moves
            if move_number % 4 == 0 {
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

        let outcome = state.outcome().expect("Game should be terminal");
        let experiences = Self::build_experiences(&move_records, &state, &outcome);
        let game_length = move_records.len();

        let winner = match outcome {
            GameOutcome::Winner(p) => Some(p),
            GameOutcome::Draw => None,
        };

        (experiences, EpisodeResult { winner, game_length })
    }

    /// Build experiences with sparse rewards from move records and final state.
    fn build_experiences(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::algorithms::DqnConfig;

    #[test]
    fn test_train_with_dashboard_sends_metrics_and_finished() {
        let config = TrainerConfig {
            num_episodes: 20,
            log_interval: 10,
            eval_interval: 20,
            eval_games: 10,
            checkpoint_interval: 100_000, // no checkpoint in short run
            checkpoint_dir: PathBuf::from("/tmp/test_dashboard_ckpt"),
        };
        let trainer = Trainer::new(config);
        let mut agent = DqnAgent::new(DqnConfig {
            min_replay_size: 5,
            batch_size: 2,
            ..Default::default()
        });

        let (tx, rx) = mpsc::channel();
        let (_, cmd_rx) = mpsc::channel();
        let pause = Arc::new(AtomicBool::new(false));
        let quit = Arc::new(AtomicBool::new(false));

        trainer.train_with_dashboard(&mut agent, tx, cmd_rx, pause, quit);

        let mut got_metrics = false;
        let mut got_finished = false;
        while let Ok(update) = rx.try_recv() {
            match update {
                TrainingUpdate::Metrics(_) => got_metrics = true,
                TrainingUpdate::Finished => got_finished = true,
                _ => {}
            }
        }
        assert!(got_metrics, "Should have received at least one Metrics update");
        assert!(got_finished, "Should have received Finished");
    }

    #[test]
    fn test_quit_atomic_stops_training() {
        let config = TrainerConfig {
            num_episodes: 10_000,
            log_interval: 10,
            eval_interval: 100_000,
            eval_games: 10,
            checkpoint_interval: 100_000,
            checkpoint_dir: PathBuf::from("/tmp/test_quit_ckpt"),
        };
        let trainer = Trainer::new(config);
        let mut agent = DqnAgent::new(DqnConfig {
            min_replay_size: 999_999, // prevent actual training
            ..Default::default()
        });

        let (tx, rx) = mpsc::channel();
        let (_, cmd_rx) = mpsc::channel();
        let pause = Arc::new(AtomicBool::new(false));
        let quit = Arc::new(AtomicBool::new(false));

        // Set quit after a very short delay
        let quit_clone = quit.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            quit_clone.store(true, Ordering::Relaxed);
        });

        trainer.train_with_dashboard(&mut agent, tx, cmd_rx, pause, quit);

        // Should have finished well before 10K episodes
        assert!(
            agent.episode_count() < 10_000,
            "Training should have stopped early, but ran {} episodes",
            agent.episode_count()
        );

        // Should still get Finished
        let mut got_finished = false;
        while let Ok(update) = rx.try_recv() {
            if matches!(update, TrainingUpdate::Finished) {
                got_finished = true;
            }
        }
        assert!(got_finished);
    }
}
