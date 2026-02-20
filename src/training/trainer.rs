use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::ai::TrainableAgent;
use crate::checkpoint::{CheckpointManager, CheckpointManagerConfig, CheckpointMetrics};
use crate::training::dashboard_msg::{MetricsSnapshot, TrainingCommand, TrainingUpdate};
use crate::training::episode;
use crate::training::metrics::{TimingMetrics, TrainingMetrics};

/// Trainer configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TrainerConfig {
    pub num_episodes: usize,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub eval_games: usize,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: PathBuf,
    pub live_update_interval: usize,
    pub base_seed: Option<u64>,
    pub num_eval_threads: usize,
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
            live_update_interval: 4,
            base_seed: None,
            num_eval_threads: 1,
        }
    }
}

/// Self-play trainer for RL agents.
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
    pub fn train(&self, agent: &mut dyn TrainableAgent) {
        let mut metrics = TrainingMetrics::with_capacity(self.config.log_interval);
        let mut timing = TimingMetrics::with_capacity(self.config.log_interval);
        let mut last_entropy: Option<f32> = None;

        let start_episode = agent.episode_count() + 1;
        let end_episode = self.config.num_episodes;

        println!(
            "Starting {} training for {} episodes (episodes {}..{})...",
            agent.algorithm_name(),
            end_episode - start_episode + 1,
            start_episode,
            end_episode
        );
        println!("-------------------------------------------");

        for ep in start_episode..=end_episode {
            if let Some(base) = self.config.base_seed {
                agent.set_episode_seed(episode::episode_seed(base, ep));
            }
            let t0 = std::time::Instant::now();
            let trace = episode::play_self_play_episode(agent);
            timing.record_episode_time(t0.elapsed());

            let t0 = std::time::Instant::now();
            let update_metrics = agent.batch_update(&trace.experiences);
            timing.record_update_time(t0.elapsed());

            if update_metrics.loss > 0.0 {
                metrics.record_update(update_metrics.loss);
            }
            if let Some(ent) = update_metrics.policy_entropy {
                last_entropy = Some(ent);
            }
            metrics.record_episode(trace.result);

            if ep % self.config.log_interval == 0 {
                let window = self.config.log_interval;
                println!(
                    "Episode {}/{} | {}: {:.3} | loss: {:.4} | win_rate({}): {:.1}% | draw: {:.1}% | avg_len: {:.1} | ep: {:.1}ms | upd: {:.1}ms | {:.1}ep/s",
                    ep,
                    end_episode,
                    agent.algorithm_metric_label(),
                    agent.algorithm_metric_value(),
                    metrics.average_loss(window),
                    window,
                    metrics.win_rate(window) * 100.0,
                    metrics.draw_rate(window) * 100.0,
                    metrics.average_game_length(window),
                    timing.avg_episode_ms(window),
                    timing.avg_update_ms(window),
                    timing.episodes_per_sec(),
                );
                timing.reset_window();
            }

            let needs_eval = ep % self.config.eval_interval == 0;
            let needs_checkpoint = ep % self.config.checkpoint_interval == 0;
            let eval_wr = if needs_eval || needs_checkpoint {
                let t0 = std::time::Instant::now();
                let wr = Some(self.do_evaluate(agent));
                timing.record_overhead(t0.elapsed());
                wr
            } else {
                None
            };

            if needs_eval {
                println!(
                    "  >> Eval vs Random ({} games): {:.1}% win rate",
                    self.config.eval_games,
                    eval_wr.unwrap() * 100.0
                );
            }

            if needs_checkpoint {
                let window = self.config.log_interval;
                let ckpt_metrics = CheckpointMetrics {
                    win_rate: eval_wr.unwrap(),
                    draw_rate: metrics.draw_rate(window),
                    average_game_length: metrics.average_game_length(window),
                    current_loss: metrics.average_loss(window),
                    training_steps: agent.step_count(),
                };
                let t0 = std::time::Instant::now();
                match self
                    .checkpoint_manager
                    .save_agent_checkpoint(agent, &ckpt_metrics, ep)
                {
                    Ok(path) => println!("  >> Checkpoint saved: {}", path.display()),
                    Err(e) => eprintln!("  >> Checkpoint failed: {}", e),
                }
                timing.record_overhead(t0.elapsed());
            }
        }

        println!("-------------------------------------------");
        println!(
            "Training complete. Total episodes: {}",
            metrics.total_episodes()
        );

        let final_wr = self.do_evaluate(agent);
        println!("Final eval vs Random: {:.1}% win rate", final_wr * 100.0);
        let _ = last_entropy; // suppress unused warning when not logging
    }

    /// Run the training loop with dashboard communication via channels.
    pub fn train_with_dashboard(
        &self,
        agent: &mut dyn TrainableAgent,
        tx: mpsc::Sender<TrainingUpdate>,
        cmd_rx: mpsc::Receiver<TrainingCommand>,
        pause: Arc<AtomicBool>,
        quit: Arc<AtomicBool>,
    ) {
        let mut metrics = TrainingMetrics::with_capacity(self.config.log_interval);
        let mut timing = TimingMetrics::with_capacity(self.config.log_interval);
        let mut last_entropy: Option<f32> = None;

        let start_episode = agent.episode_count() + 1;
        let end_episode = self.config.num_episodes;

        for ep in start_episode..=end_episode {
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
                        let t0 = std::time::Instant::now();
                        self.save_checkpoint_with_tx(agent, &metrics, ep, &tx);
                        timing.record_overhead(t0.elapsed());
                    }
                }
            }

            if let Some(base) = self.config.base_seed {
                agent.set_episode_seed(episode::episode_seed(base, ep));
            }

            // Play episode with live game updates
            let t0 = std::time::Instant::now();
            let trace = episode::play_self_play_episode_live(
                agent,
                &tx,
                self.config.live_update_interval,
            );
            timing.record_episode_time(t0.elapsed());

            let t0 = std::time::Instant::now();
            let update_metrics = agent.batch_update(&trace.experiences);
            timing.record_update_time(t0.elapsed());

            if update_metrics.loss > 0.0 {
                metrics.record_update(update_metrics.loss);
            }
            if let Some(ent) = update_metrics.policy_entropy {
                last_entropy = Some(ent);
            }
            metrics.record_episode(trace.result);

            // Send metrics every log_interval episodes
            if ep % self.config.log_interval == 0 {
                let window = self.config.log_interval;
                let snap = MetricsSnapshot {
                    episode: ep,
                    total_episodes: end_episode,
                    epsilon: agent.algorithm_metric_value(),
                    win_rate: metrics.win_rate(window),
                    draw_rate: metrics.draw_rate(window),
                    loss: metrics.average_loss(window),
                    avg_game_length: metrics.average_game_length(window),
                    step_count: agent.step_count(),
                    algorithm: agent.algorithm_name().to_string(),
                    policy_entropy: last_entropy,
                    episodes_per_sec: timing.episodes_per_sec(),
                    avg_episode_ms: timing.avg_episode_ms(window),
                    avg_update_ms: timing.avg_update_ms(window),
                };
                let _ = tx.send(TrainingUpdate::Metrics(snap));
                timing.reset_window();
            }

            // Eval and checkpoint (share a single evaluate call)
            let needs_eval = ep % self.config.eval_interval == 0;
            let needs_checkpoint = ep % self.config.checkpoint_interval == 0;
            let eval_wr = if needs_eval || needs_checkpoint {
                let t0 = std::time::Instant::now();
                let wr = Some(self.do_evaluate(agent));
                timing.record_overhead(t0.elapsed());
                wr
            } else {
                None
            };

            if needs_eval {
                let _ = tx.send(TrainingUpdate::EvalResult {
                    episode: ep,
                    win_rate: eval_wr.unwrap(),
                });
            }

            if needs_checkpoint {
                let t0 = std::time::Instant::now();
                self.save_checkpoint_with_tx_precomputed(
                    agent, &metrics, ep, eval_wr.unwrap(), &tx,
                );
                timing.record_overhead(t0.elapsed());
            }
        }

        let _ = tx.send(TrainingUpdate::Finished);
    }

    fn do_evaluate(&self, agent: &mut dyn TrainableAgent) -> f32 {
        if self.config.num_eval_threads > 1 {
            episode::parallel_evaluate(agent, self.config.eval_games, self.config.num_eval_threads)
        } else {
            episode::evaluate(agent, self.config.eval_games)
        }
    }

    fn save_checkpoint_with_tx(
        &self,
        agent: &mut dyn TrainableAgent,
        metrics: &TrainingMetrics,
        episode: usize,
        tx: &mpsc::Sender<TrainingUpdate>,
    ) {
        let eval_wr = self.do_evaluate(agent);
        self.save_checkpoint_with_tx_precomputed(agent, metrics, episode, eval_wr, tx);
    }

    fn save_checkpoint_with_tx_precomputed(
        &self,
        agent: &dyn TrainableAgent,
        metrics: &TrainingMetrics,
        episode: usize,
        eval_wr: f32,
        tx: &mpsc::Sender<TrainingUpdate>,
    ) {
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
            .save_agent_checkpoint(agent, &ckpt_metrics, episode)
        {
            Ok(path) => {
                let _ = tx.send(TrainingUpdate::CheckpointSaved { episode, path });
            }
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::algorithms::{DqnAgent, DqnConfig, PgConfig, PolicyGradientAgent};

    #[test]
    fn test_train_with_dashboard_sends_metrics_and_finished() {
        let config = TrainerConfig {
            num_episodes: 20,
            log_interval: 10,
            eval_interval: 20,
            eval_games: 10,
            checkpoint_interval: 100_000, // no checkpoint in short run
            checkpoint_dir: PathBuf::from("/tmp/test_dashboard_ckpt"),
            ..Default::default()
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
            ..Default::default()
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

    #[test]
    fn test_train_pg_20_episodes() {
        let config = TrainerConfig {
            num_episodes: 20,
            log_interval: 10,
            eval_interval: 20,
            eval_games: 10,
            checkpoint_interval: 100_000,
            checkpoint_dir: PathBuf::from("/tmp/test_pg_train_ckpt"),
            ..Default::default()
        };
        let trainer = Trainer::new(config);
        let mut agent = PolicyGradientAgent::new(PgConfig {
            ppo_epochs: 1,
            ..Default::default()
        });

        trainer.train(&mut agent);
        assert_eq!(agent.episode_count(), 20);
    }

    #[test]
    fn test_pg_dashboard_sends_metrics() {
        let config = TrainerConfig {
            num_episodes: 20,
            log_interval: 10,
            eval_interval: 20,
            eval_games: 10,
            checkpoint_interval: 100_000,
            checkpoint_dir: PathBuf::from("/tmp/test_pg_dashboard_ckpt"),
            ..Default::default()
        };
        let trainer = Trainer::new(config);
        let mut agent = PolicyGradientAgent::new(PgConfig {
            ppo_epochs: 1,
            ..Default::default()
        });

        let (tx, rx) = mpsc::channel();
        let (_, cmd_rx) = mpsc::channel();
        let pause = Arc::new(AtomicBool::new(false));
        let quit = Arc::new(AtomicBool::new(false));

        trainer.train_with_dashboard(&mut agent, tx, cmd_rx, pause, quit);

        let mut got_metrics = false;
        let mut got_finished = false;
        let mut algorithm_is_pg = false;
        while let Ok(update) = rx.try_recv() {
            match update {
                TrainingUpdate::Metrics(snap) => {
                    got_metrics = true;
                    if snap.algorithm == "PG" {
                        algorithm_is_pg = true;
                    }
                }
                TrainingUpdate::Finished => got_finished = true,
                _ => {}
            }
        }
        assert!(got_metrics, "Should have received at least one Metrics update");
        assert!(got_finished, "Should have received Finished");
        assert!(algorithm_is_pg, "Algorithm should be PG");
    }
}
