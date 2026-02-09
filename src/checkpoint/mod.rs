//! Checkpoint system for saving and loading trained models, with metadata
//! tracking, symlink-based latest resolution, and automatic pruning.

mod manager;
mod metadata;

pub use manager::{CheckpointData, CheckpointManager, CheckpointManagerConfig, PgCheckpointData};
pub use metadata::{
    CheckpointHyperparameters, CheckpointMetadata, CheckpointMetrics, DqnTrainingState,
    PgTrainingState,
};
