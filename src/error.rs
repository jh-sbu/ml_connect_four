use std::path::PathBuf;

/// Errors that can occur during checkpoint operations.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint directory not found: {0}")]
    DirNotFound(PathBuf),

    #[error("no 'latest' symlink found in {0}")]
    NoLatestSymlink(PathBuf),

    #[error("failed to read metadata from {path}: {source}")]
    MetadataRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to parse metadata from {path}: {source}")]
    MetadataParse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("failed to save model: {0}")]
    ModelSave(String),

    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Errors that can occur during training.
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("agent selected illegal action {action} (legal: {legal:?})")]
    IllegalAction { action: usize, legal: Vec<usize> },

    #[error("game should be terminal but has no outcome")]
    MissingOutcome,

    #[error("checkpoint error: {0}")]
    Checkpoint(#[from] CheckpointError),
}

/// Errors that can occur when loading configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to parse TOML: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("config validation error: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_error_display() {
        let err = CheckpointError::NoLatestSymlink(PathBuf::from("checkpoints"));
        assert_eq!(
            err.to_string(),
            "no 'latest' symlink found in checkpoints"
        );
    }

    #[test]
    fn test_training_error_display() {
        let err = TrainingError::IllegalAction {
            action: 5,
            legal: vec![0, 1, 2],
        };
        assert_eq!(
            err.to_string(),
            "agent selected illegal action 5 (legal: [0, 1, 2])"
        );
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::Validation("learning_rate must be > 0".to_string());
        assert_eq!(
            err.to_string(),
            "config validation error: learning_rate must be > 0"
        );
    }
}
