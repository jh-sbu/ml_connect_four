use serde::{Deserialize, Serialize};

/// Metrics snapshot at checkpoint time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetrics {
    pub win_rate: f32,
    pub draw_rate: f32,
    pub average_game_length: f32,
    pub current_loss: f32,
    pub training_steps: usize,
}

/// Hyperparameters recorded in checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHyperparameters {
    pub learning_rate: f64,
    pub gamma: f32,
    pub epsilon: f32,
    pub batch_size: usize,
    pub target_update_interval: usize,
    pub replay_capacity: usize,
    pub min_replay_size: usize,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay_episodes: usize,
}

/// Top-level checkpoint metadata written to metadata.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub episode: usize,
    pub timestamp: u64,
    pub algorithm: String,
    pub metrics: CheckpointMetrics,
    pub hyperparameters: CheckpointHyperparameters,
}

/// DQN-specific training state written to training_state.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnTrainingState {
    pub epsilon: f32,
    pub step_count: usize,
    pub episode_count: usize,
    pub learning_rate: f64,
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay_episodes: usize,
    pub target_update_interval: usize,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub min_replay_size: usize,
}
