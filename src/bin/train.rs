use std::env;

use ml_connect_four::ai::algorithms::{DqnAgent, DqnConfig};
use ml_connect_four::checkpoint::{CheckpointManager, CheckpointManagerConfig};
use ml_connect_four::training::trainer::{Trainer, TrainerConfig};

fn main() {
    let args: Vec<String> = env::args().collect();
    let resume = args.iter().any(|a| a == "--resume");

    let trainer_config = TrainerConfig::default();
    let mut agent = DqnAgent::new(DqnConfig::default());

    if resume {
        let manager = CheckpointManager::new(CheckpointManagerConfig {
            checkpoint_dir: trainer_config.checkpoint_dir.clone(),
            ..Default::default()
        });
        match manager.load_latest() {
            Ok(data) => {
                agent
                    .load_from_dir(&data.path)
                    .expect("Failed to load checkpoint weights");
                agent.restore_training_state(&data.training_state);
                println!("Resumed from episode {}", data.metadata.episode);
            }
            Err(e) => println!("No checkpoint found ({}), starting fresh", e),
        }
    }

    let trainer = Trainer::new(trainer_config);
    trainer.train(&mut agent);
}
