use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ai::algorithms::DqnAgent;
use crate::checkpoint::metadata::{
    CheckpointHyperparameters, CheckpointMetadata, CheckpointMetrics, DqnTrainingState,
};

/// Configuration for the checkpoint manager.
pub struct CheckpointManagerConfig {
    pub checkpoint_dir: PathBuf,
    pub keep_last_n: usize,
    pub keep_best_n: usize,
}

impl Default for CheckpointManagerConfig {
    fn default() -> Self {
        CheckpointManagerConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            keep_last_n: 5,
            keep_best_n: 3,
        }
    }
}

/// Data returned when loading a checkpoint.
pub struct CheckpointData {
    pub path: PathBuf,
    pub metadata: CheckpointMetadata,
    pub training_state: DqnTrainingState,
}

/// Manages saving, loading, listing, and pruning checkpoints.
pub struct CheckpointManager {
    config: CheckpointManagerConfig,
}

impl CheckpointManager {
    pub fn new(config: CheckpointManagerConfig) -> Self {
        fs::create_dir_all(&config.checkpoint_dir).ok();
        CheckpointManager { config }
    }

    /// Save a checkpoint: network weights, metadata, and training state.
    pub fn save_checkpoint(
        &self,
        agent: &DqnAgent,
        metrics: &CheckpointMetrics,
        episode: usize,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let dir_name = format!("checkpoint_{:07}", episode);
        let tmp_dir = self.config.checkpoint_dir.join(format!("{}.tmp", dir_name));
        let final_dir = self.config.checkpoint_dir.join(&dir_name);

        // Write to temp dir first for atomicity
        fs::create_dir_all(&tmp_dir)?;

        // Save network weights
        agent.save_to_dir(&tmp_dir)?;

        // Save training state
        let training_state = agent.training_state();
        let ts_json = serde_json::to_string_pretty(&training_state)?;
        fs::write(tmp_dir.join("training_state.json"), ts_json)?;

        // Save metadata
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let metadata = CheckpointMetadata {
            episode,
            timestamp,
            algorithm: "DQN".to_string(),
            metrics: metrics.clone(),
            hyperparameters: CheckpointHyperparameters {
                learning_rate: training_state.learning_rate,
                gamma: training_state.gamma,
                epsilon: training_state.epsilon,
                batch_size: training_state.batch_size,
                target_update_interval: training_state.target_update_interval,
                replay_capacity: training_state.replay_capacity,
                min_replay_size: training_state.min_replay_size,
                epsilon_start: training_state.epsilon_start,
                epsilon_end: training_state.epsilon_end,
                epsilon_decay_episodes: training_state.epsilon_decay_episodes,
            },
        };
        let meta_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(tmp_dir.join("metadata.json"), meta_json)?;

        // Atomic rename
        if final_dir.exists() {
            fs::remove_dir_all(&final_dir)?;
        }
        fs::rename(&tmp_dir, &final_dir)?;

        // Update latest symlink
        self.update_latest_symlink(&dir_name)?;

        // Prune old checkpoints
        self.prune_old_checkpoints()?;

        Ok(final_dir)
    }

    /// Load checkpoint data from a specific directory.
    pub fn load_checkpoint(&self, dir: &Path) -> Result<CheckpointData, Box<dyn Error>> {
        let meta_path = dir.join("metadata.json");
        let ts_path = dir.join("training_state.json");

        let meta_json = fs::read_to_string(&meta_path)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&meta_json)?;

        let ts_json = fs::read_to_string(&ts_path)?;
        let training_state: DqnTrainingState = serde_json::from_str(&ts_json)?;

        Ok(CheckpointData {
            path: dir.to_path_buf(),
            metadata,
            training_state,
        })
    }

    /// Load the latest checkpoint (via the `latest` symlink).
    pub fn load_latest(&self) -> Result<CheckpointData, Box<dyn Error>> {
        let latest_link = self.config.checkpoint_dir.join("latest");
        if !latest_link.exists() {
            return Err("No 'latest' symlink found".into());
        }
        let resolved = fs::read_link(&latest_link)?;
        let target = if resolved.is_relative() {
            self.config.checkpoint_dir.join(resolved)
        } else {
            resolved
        };
        self.load_checkpoint(&target)
    }

    /// List all checkpoints sorted by episode (ascending).
    pub fn list_checkpoints(
        &self,
    ) -> Result<Vec<(PathBuf, CheckpointMetadata)>, Box<dyn Error>> {
        let mut results = Vec::new();
        for entry in fs::read_dir(&self.config.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with("checkpoint_") || name_str.ends_with(".tmp") {
                continue;
            }
            let meta_path = path.join("metadata.json");
            if meta_path.exists() {
                let meta_json = fs::read_to_string(&meta_path)?;
                let metadata: CheckpointMetadata = serde_json::from_str(&meta_json)?;
                results.push((path, metadata));
            }
        }
        results.sort_by_key(|(_, m)| m.episode);
        Ok(results)
    }

    /// Prune old checkpoints, keeping the union of the last N and best N by win_rate.
    fn prune_old_checkpoints(&self) -> Result<(), Box<dyn Error>> {
        let checkpoints = self.list_checkpoints()?;
        if checkpoints.len() <= self.config.keep_last_n {
            return Ok(());
        }

        // Indices to keep: last N by episode
        let total = checkpoints.len();
        let mut keep: std::collections::HashSet<usize> = (total
            .saturating_sub(self.config.keep_last_n)..total)
            .collect();

        // Also keep best N by win_rate
        let mut by_win_rate: Vec<(usize, f32)> = checkpoints
            .iter()
            .enumerate()
            .map(|(i, (_, m))| (i, m.metrics.win_rate))
            .collect();
        by_win_rate.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, _) in by_win_rate.iter().take(self.config.keep_best_n) {
            keep.insert(*i);
        }

        // Delete checkpoints not in keep set
        for (i, (path, _)) in checkpoints.iter().enumerate() {
            if !keep.contains(&i) {
                fs::remove_dir_all(path)?;
            }
        }

        Ok(())
    }

    /// Update the `latest` symlink to point to the given checkpoint directory name.
    fn update_latest_symlink(&self, dir_name: &str) -> Result<(), Box<dyn Error>> {
        let link_path = self.config.checkpoint_dir.join("latest");
        // Remove old symlink if it exists
        if link_path.exists() || link_path.symlink_metadata().is_ok() {
            fs::remove_file(&link_path)?;
        }
        std::os::unix::fs::symlink(dir_name, &link_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::algorithms::DqnConfig;

    fn test_metrics() -> CheckpointMetrics {
        CheckpointMetrics {
            win_rate: 0.65,
            draw_rate: 0.10,
            average_game_length: 20.0,
            current_loss: 0.05,
            training_steps: 1000,
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let config = CheckpointManagerConfig {
            checkpoint_dir: dir.path().to_path_buf(),
            keep_last_n: 5,
            keep_best_n: 3,
        };
        let manager = CheckpointManager::new(config);
        let agent = DqnAgent::new(DqnConfig::default());

        let path = manager
            .save_checkpoint(&agent, &test_metrics(), 1000)
            .unwrap();
        assert!(path.exists());
        assert!(path.join("metadata.json").exists());
        assert!(path.join("training_state.json").exists());
        assert!(path.join("q_network.mpk").exists());
        assert!(path.join("target_network.mpk").exists());

        // Load into a new agent
        let data = manager.load_checkpoint(&path).unwrap();
        assert_eq!(data.metadata.episode, 1000);
        assert_eq!(data.metadata.algorithm, "DQN");

        let mut new_agent = DqnAgent::new(DqnConfig::default());
        new_agent.load_from_dir(&data.path).unwrap();
    }

    #[test]
    fn test_training_state_roundtrip() {
        let mut agent = DqnAgent::new(DqnConfig {
            learning_rate: 0.001,
            gamma: 0.95,
            epsilon_start: 0.8,
            epsilon_end: 0.05,
            ..Default::default()
        });
        agent.set_epsilon(0.42);

        let state = agent.training_state();
        assert!((state.epsilon - 0.42).abs() < 1e-6);
        assert!((state.learning_rate - 0.001).abs() < 1e-9);
        assert!((state.gamma - 0.95).abs() < 1e-6);

        let mut new_agent = DqnAgent::new(DqnConfig::default());
        new_agent.restore_training_state(&state);
        assert!((new_agent.epsilon() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_metadata_serde() {
        let meta = CheckpointMetadata {
            episode: 5000,
            timestamp: 1700000000,
            algorithm: "DQN".to_string(),
            metrics: test_metrics(),
            hyperparameters: CheckpointHyperparameters {
                learning_rate: 1e-4,
                gamma: 0.99,
                epsilon: 0.5,
                batch_size: 64,
                target_update_interval: 1000,
                replay_capacity: 50000,
                min_replay_size: 1000,
                epsilon_start: 1.0,
                epsilon_end: 0.1,
                epsilon_decay_episodes: 10000,
            },
        };

        let json = serde_json::to_string_pretty(&meta).unwrap();
        let deserialized: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.episode, 5000);
        assert_eq!(deserialized.algorithm, "DQN");
        assert!((deserialized.metrics.win_rate - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_latest_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let config = CheckpointManagerConfig {
            checkpoint_dir: dir.path().to_path_buf(),
            keep_last_n: 5,
            keep_best_n: 3,
        };
        let manager = CheckpointManager::new(config);
        let agent = DqnAgent::new(DqnConfig::default());

        manager
            .save_checkpoint(&agent, &test_metrics(), 1000)
            .unwrap();
        manager
            .save_checkpoint(&agent, &test_metrics(), 2000)
            .unwrap();

        let latest = manager.load_latest().unwrap();
        assert_eq!(latest.metadata.episode, 2000);
    }

    #[test]
    fn test_list_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let config = CheckpointManagerConfig {
            checkpoint_dir: dir.path().to_path_buf(),
            keep_last_n: 10,
            keep_best_n: 10,
        };
        let manager = CheckpointManager::new(config);
        let agent = DqnAgent::new(DqnConfig::default());

        for ep in [1000, 2000, 3000] {
            manager
                .save_checkpoint(&agent, &test_metrics(), ep)
                .unwrap();
        }

        let list = manager.list_checkpoints().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].1.episode, 1000);
        assert_eq!(list[1].1.episode, 2000);
        assert_eq!(list[2].1.episode, 3000);
    }

    #[test]
    fn test_pruning() {
        let dir = tempfile::tempdir().unwrap();
        let config = CheckpointManagerConfig {
            checkpoint_dir: dir.path().to_path_buf(),
            keep_last_n: 2,
            keep_best_n: 1,
        };
        let manager = CheckpointManager::new(config);
        let agent = DqnAgent::new(DqnConfig::default());

        // Save 5 checkpoints with varying win rates
        let win_rates = [0.5, 0.9, 0.3, 0.6, 0.7];
        for (i, &wr) in win_rates.iter().enumerate() {
            let ep = (i + 1) * 1000;
            let mut metrics = test_metrics();
            metrics.win_rate = wr;
            manager.save_checkpoint(&agent, &metrics, ep).unwrap();
        }

        let list = manager.list_checkpoints().unwrap();
        // Should keep: last 2 (ep 4000, 5000) + best 1 (ep 2000, wr=0.9) = 3 checkpoints
        assert_eq!(list.len(), 3);

        let episodes: Vec<usize> = list.iter().map(|(_, m)| m.episode).collect();
        assert!(episodes.contains(&2000)); // best win rate
        assert!(episodes.contains(&4000)); // second to last
        assert!(episodes.contains(&5000)); // last
    }
}
