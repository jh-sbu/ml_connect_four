//! Checkpoint system for saving and loading trained models, with metadata
//! tracking, symlink-based latest resolution, and automatic pruning.

mod manager;
mod metadata;

pub use manager::{AgentCheckpointData, CheckpointManager, CheckpointManagerConfig};
pub use metadata::{
    AlphaZeroHyperparameters, AlphaZeroTrainingState, CheckpointHyperparameters,
    CheckpointMetadata, CheckpointMetrics, DqnTrainingState, PgHyperparameters, PgTrainingState,
};
