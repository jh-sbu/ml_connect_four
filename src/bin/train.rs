use ml_connect_four::ai::algorithms::{DqnAgent, DqnConfig};
use ml_connect_four::training::trainer::{Trainer, TrainerConfig};

fn main() {
    let mut agent = DqnAgent::new(DqnConfig::default());
    let trainer = Trainer::new(TrainerConfig::default());
    trainer.train(&mut agent);
}
