use std::path::Path;

use crate::ai::algorithms::{AlphaZeroConfig, DqnConfig, PgConfig};
use crate::checkpoint::CheckpointManagerConfig;
use crate::error::ConfigError;
use crate::training::trainer::TrainerConfig;

/// Top-level application configuration, loadable from TOML.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub dqn: DqnConfig,
    pub pg: PgConfig,
    pub az: AlphaZeroConfig,
    pub training: TrainerConfig,
    pub checkpoint: CheckpointManagerConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            dqn: DqnConfig::default(),
            pg: PgConfig::default(),
            az: AlphaZeroConfig::default(),
            training: TrainerConfig::default(),
            checkpoint: CheckpointManagerConfig::default(),
        }
    }
}

impl AppConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::FileRead {
            path: path.to_path_buf(),
            source: e,
        })?;
        let config: AppConfig = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a TOML file, falling back to defaults if the file
    /// does not exist.
    pub fn load_or_default(path: &Path) -> Result<Self, ConfigError> {
        if path.exists() {
            Self::load(path)
        } else {
            eprintln!("Warning: config file '{}' not found, using defaults", path.display());
            Ok(Self::default())
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.dqn.learning_rate <= 0.0 {
            return Err(ConfigError::Validation(
                "dqn.learning_rate must be > 0".into(),
            ));
        }
        if self.pg.learning_rate <= 0.0 {
            return Err(ConfigError::Validation(
                "pg.learning_rate must be > 0".into(),
            ));
        }
        if self.training.num_episodes == 0 {
            return Err(ConfigError::Validation(
                "training.num_episodes must be > 0".into(),
            ));
        }
        if self.dqn.batch_size == 0 {
            return Err(ConfigError::Validation(
                "dqn.batch_size must be > 0".into(),
            ));
        }
        if self.dqn.gamma < 0.0 || self.dqn.gamma > 1.0 {
            return Err(ConfigError::Validation(
                "dqn.gamma must be in [0, 1]".into(),
            ));
        }
        if self.pg.gamma < 0.0 || self.pg.gamma > 1.0 {
            return Err(ConfigError::Validation(
                "pg.gamma must be in [0, 1]".into(),
            ));
        }

        // DQN epsilon validations
        if self.dqn.epsilon_start < 0.0 || self.dqn.epsilon_start > 1.0 {
            return Err(ConfigError::Validation(
                "dqn.epsilon_start must be in [0, 1]".into(),
            ));
        }
        if self.dqn.epsilon_end < 0.0 || self.dqn.epsilon_end > 1.0 {
            return Err(ConfigError::Validation(
                "dqn.epsilon_end must be in [0, 1]".into(),
            ));
        }
        if self.dqn.epsilon_end > self.dqn.epsilon_start {
            return Err(ConfigError::Validation(
                "dqn.epsilon_end must be <= dqn.epsilon_start".into(),
            ));
        }
        if self.dqn.replay_capacity < self.dqn.batch_size {
            return Err(ConfigError::Validation(
                "dqn.replay_capacity must be >= dqn.batch_size".into(),
            ));
        }
        if self.dqn.min_replay_size < self.dqn.batch_size {
            return Err(ConfigError::Validation(
                "dqn.min_replay_size must be >= dqn.batch_size".into(),
            ));
        }

        // PG validations
        if self.pg.gae_lambda < 0.0 || self.pg.gae_lambda > 1.0 {
            return Err(ConfigError::Validation(
                "pg.gae_lambda must be in [0, 1]".into(),
            ));
        }
        if self.pg.ppo_epsilon <= 0.0 {
            return Err(ConfigError::Validation(
                "pg.ppo_epsilon must be > 0".into(),
            ));
        }
        if self.pg.ppo_epochs == 0 {
            return Err(ConfigError::Validation(
                "pg.ppo_epochs must be > 0".into(),
            ));
        }
        if self.pg.entropy_coeff < 0.0 {
            return Err(ConfigError::Validation(
                "pg.entropy_coeff must be >= 0".into(),
            ));
        }
        if self.pg.value_coeff < 0.0 {
            return Err(ConfigError::Validation(
                "pg.value_coeff must be >= 0".into(),
            ));
        }
        if self.pg.max_grad_norm <= 0.0 {
            return Err(ConfigError::Validation(
                "pg.max_grad_norm must be > 0".into(),
            ));
        }
        if self.pg.rollout_episodes == 0 {
            return Err(ConfigError::Validation(
                "pg.rollout_episodes must be >= 1".into(),
            ));
        }
        // AZ validations
        if self.az.learning_rate <= 0.0 {
            return Err(ConfigError::Validation(
                "az.learning_rate must be > 0".into(),
            ));
        }
        if self.az.num_simulations == 0 {
            return Err(ConfigError::Validation(
                "az.num_simulations must be >= 1".into(),
            ));
        }
        if self.az.c_puct <= 0.0 {
            return Err(ConfigError::Validation(
                "az.c_puct must be > 0".into(),
            ));
        }
        if self.az.batch_size == 0 {
            return Err(ConfigError::Validation(
                "az.batch_size must be > 0".into(),
            ));
        }
        if self.az.replay_capacity < self.az.batch_size {
            return Err(ConfigError::Validation(
                "az.replay_capacity must be >= az.batch_size".into(),
            ));
        }
        if self.az.min_replay_size < self.az.batch_size {
            return Err(ConfigError::Validation(
                "az.min_replay_size must be >= az.batch_size".into(),
            ));
        }
        if self.az.dirichlet_epsilon <= 0.0 || self.az.dirichlet_epsilon >= 1.0 {
            return Err(ConfigError::Validation(
                "az.dirichlet_epsilon must be in (0, 1)".into(),
            ));
        }
        if self.az.value_weight < 0.0 {
            return Err(ConfigError::Validation(
                "az.value_weight must be >= 0".into(),
            ));
        }
        if self.az.max_grad_norm <= 0.0 {
            return Err(ConfigError::Validation(
                "az.max_grad_norm must be > 0".into(),
            ));
        }

        if self.training.live_update_interval == 0 {
            return Err(ConfigError::Validation(
                "training.live_update_interval must be > 0".into(),
            ));
        }
        if self.training.num_eval_threads == 0 {
            return Err(ConfigError::Validation(
                "training.num_eval_threads must be >= 1".into(),
            ));
        }

        Ok(())
    }

    /// Generate a TOML string with all default values (useful for creating
    /// example config files).
    pub fn default_toml() -> String {
        toml::to_string_pretty(&AppConfig::default()).expect("default config serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_default_config_is_valid() {
        let config = AppConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
[dqn]
learning_rate = 0.001
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert!((config.dqn.learning_rate - 0.001).abs() < 1e-9);
        // Other fields should be defaults
        assert!((config.dqn.gamma - 0.99).abs() < 1e-6);
        assert_eq!(config.training.num_episodes, 10_000);
    }

    #[test]
    fn test_empty_toml_uses_all_defaults() {
        let config: AppConfig = toml::from_str("").unwrap();
        let default = AppConfig::default();
        assert!((config.dqn.learning_rate - default.dqn.learning_rate).abs() < 1e-9);
        assert_eq!(config.training.num_episodes, default.training.num_episodes);
    }

    #[test]
    fn test_validation_rejects_zero_episodes() {
        let mut config = AppConfig::default();
        config.training.num_episodes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_negative_lr() {
        let mut config = AppConfig::default();
        config.dqn.learning_rate = -0.001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_invalid_gamma() {
        let mut config = AppConfig::default();
        config.pg.gamma = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_load_or_default_missing_file() {
        let config = AppConfig::load_or_default(Path::new("nonexistent_config.toml")).unwrap();
        assert_eq!(config.training.num_episodes, 10_000);
    }

    #[test]
    fn test_load_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"
[training]
num_episodes = 500
"#
        )
        .unwrap();

        let config = AppConfig::load(&path).unwrap();
        assert_eq!(config.training.num_episodes, 500);
        // Others are defaults
        assert!((config.dqn.learning_rate - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_default_toml_roundtrips() {
        let toml_str = AppConfig::default_toml();
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        config.validate().expect("roundtripped config should be valid");
    }

    #[test]
    fn test_validation_rejects_epsilon_start_out_of_range() {
        let mut config = AppConfig::default();
        config.dqn.epsilon_start = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_epsilon_end_out_of_range() {
        let mut config = AppConfig::default();
        config.dqn.epsilon_end = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_epsilon_end_gt_start() {
        let mut config = AppConfig::default();
        config.dqn.epsilon_start = 0.1;
        config.dqn.epsilon_end = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_replay_capacity_lt_batch() {
        let mut config = AppConfig::default();
        config.dqn.replay_capacity = 10;
        config.dqn.batch_size = 64;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_min_replay_lt_batch() {
        let mut config = AppConfig::default();
        config.dqn.min_replay_size = 10;
        config.dqn.batch_size = 64;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_gae_lambda_out_of_range() {
        let mut config = AppConfig::default();
        config.pg.gae_lambda = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_ppo_epsilon_zero() {
        let mut config = AppConfig::default();
        config.pg.ppo_epsilon = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_ppo_epochs_zero() {
        let mut config = AppConfig::default();
        config.pg.ppo_epochs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_negative_entropy_coeff() {
        let mut config = AppConfig::default();
        config.pg.entropy_coeff = -0.01;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_negative_value_coeff() {
        let mut config = AppConfig::default();
        config.pg.value_coeff = -0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_max_grad_norm_zero() {
        let mut config = AppConfig::default();
        config.pg.max_grad_norm = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_rollout_episodes_zero() {
        let mut config = AppConfig::default();
        config.pg.rollout_episodes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_zero_live_update_interval() {
        let mut config = AppConfig::default();
        config.training.live_update_interval = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_zero_eval_threads() {
        let mut config = AppConfig::default();
        config.training.num_eval_threads = 0;
        assert!(config.validate().is_err());
    }
}
