use std::path::PathBuf;

use crate::game::GameState;

/// Periodic metrics snapshot sent from training thread to dashboard.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub episode: usize,
    pub total_episodes: usize,
    pub epsilon: f32,
    pub win_rate: f32,
    pub draw_rate: f32,
    pub loss: f32,
    pub avg_game_length: f32,
    pub step_count: usize,
    pub algorithm: String,
    pub policy_entropy: Option<f32>,
    pub episodes_per_sec: f32,
    pub avg_episode_ms: f32,
    pub avg_update_ms: f32,
}

/// Live game state sent during training episodes.
#[derive(Debug, Clone)]
pub struct LiveGameState {
    pub game_state: GameState,
    pub move_number: usize,
}

/// Updates sent from training thread to UI.
#[derive(Debug, Clone)]
pub enum TrainingUpdate {
    Metrics(MetricsSnapshot),
    LiveGame(LiveGameState),
    EvalResult {
        episode: usize,
        win_rate: f32,
    },
    CheckpointSaved {
        episode: usize,
        path: PathBuf,
    },
    Finished,
}

/// Commands sent from UI to training thread.
#[derive(Debug, Clone)]
pub enum TrainingCommand {
    SaveCheckpoint,
}
