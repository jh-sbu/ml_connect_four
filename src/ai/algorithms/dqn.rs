use std::error::Error;
use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::DefaultRecorder;
use burn::tensor::TensorData;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::ai::agent::{Agent, AgentMetrics, EvalState, Experience, TrainableAgent, UpdateMetrics};
use crate::checkpoint::{CheckpointHyperparameters, CheckpointMetadata};
use crate::ai::networks::{DqnNetwork, DqnNetworkConfig};
use crate::ai::state_encoding::{encode_state, encode_states_batch};
use crate::checkpoint::DqnTrainingState;
use crate::game::GameState;
use crate::training::replay_buffer::ReplayBuffer;

type InferBackend = Wgpu<f32, i32>;
type TrainBackend = Autodiff<InferBackend>;

/// DQN hyperparameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DqnConfig {
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

impl Default for DqnConfig {
    fn default() -> Self {
        DqnConfig {
            learning_rate: 1e-4,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.1,
            epsilon_decay_episodes: 10_000,
            target_update_interval: 1000,
            batch_size: 64,
            replay_capacity: 50_000,
            min_replay_size: 1000,
        }
    }
}

/// DQN agent with online + target networks, replay buffer, and Adam optimizer.
pub struct DqnAgent {
    q_network: DqnNetwork<TrainBackend>,
    target_network: DqnNetwork<InferBackend>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, DqnNetwork<TrainBackend>, TrainBackend>,
    replay_buffer: ReplayBuffer,
    config: DqnConfig,
    device: <TrainBackend as Backend>::Device,
    epsilon: f32,
    step_count: usize,
    episode_count: usize,
    rng: StdRng,
}


impl DqnAgent {
    pub fn new(config: DqnConfig) -> Self {
        let device = Default::default();
        let net_config = DqnNetworkConfig {};
        let q_network: DqnNetwork<TrainBackend> = net_config.init(&device);
        let target_network: DqnNetwork<InferBackend> = net_config.init(&device);
        let optimizer = AdamConfig::new().init();

        let epsilon = config.epsilon_start;
        let replay_buffer = ReplayBuffer::new(config.replay_capacity);

        DqnAgent {
            q_network,
            target_network,
            optimizer,
            replay_buffer,
            config,
            device,
            epsilon,
            step_count: 0,
            episode_count: 0,
            rng: StdRng::from_os_rng(),
        }
    }

    /// Select action: epsilon-greedy when training, greedy otherwise.
    fn pick_action(&mut self, state: &GameState, training: bool) -> usize {
        let legal = state.legal_actions();
        assert!(!legal.is_empty(), "No legal actions");

        if training && self.rng.random_range(0.0..1.0) < self.epsilon {
            // Random exploration
            let idx = self.rng.random_range(0..legal.len());
            return legal[idx];
        }

        // Greedy: forward pass through q_network, pick best legal action
        let state_tensor = encode_state::<InferBackend>(state, &self.device)
            .unsqueeze::<4>(); // [1, 3, 6, 7]
        let q_values = self.target_network.forward(state_tensor); // [1, 7]
        let q_vec: Vec<f32> = q_values
            .into_data()
            .to_vec()
            .expect("f32 tensor data extraction");

        // Pick legal action with highest Q-value
        let mut best_action = legal[0];
        let mut best_q = f32::NEG_INFINITY;
        for &col in &legal {
            if q_vec[col] > best_q {
                best_q = q_vec[col];
                best_action = col;
            }
        }
        best_action
    }

    /// Perform one gradient update step from replay buffer.
    fn train_step(&mut self) -> f32 {
        let batch = self.replay_buffer.sample(self.config.batch_size);
        let batch_size = batch.len();

        // Encode states for training backend
        let states: Vec<GameState> = batch.iter().map(|e| e.state.clone()).collect();
        let next_states: Vec<GameState> = batch.iter().map(|e| e.next_state.clone()).collect();
        let actions: Vec<usize> = batch.iter().map(|e| e.action).collect();
        let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
        let dones: Vec<f32> = batch.iter().map(|e| if e.done { 1.0 } else { 0.0 }).collect();

        // Forward pass on current states: [B, 7]
        let state_tensors = encode_states_batch::<TrainBackend>(&states, &self.device);
        let q_all = self.q_network.forward(state_tensors);

        // Create one-hot action mask [B, 7] to extract Q(s, a)
        let mut action_mask_data = vec![0.0f32; batch_size * 7];
        for (i, &a) in actions.iter().enumerate() {
            action_mask_data[i * 7 + a] = 1.0;
        }
        let action_mask = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(action_mask_data.as_slice()),
            &self.device,
        )
        .reshape([batch_size as i32, 7]);

        // Q(s, a) = sum(q_all * mask, dim=1) -> [B, 1]
        let q_taken = (q_all * action_mask).sum_dim(1);

        // Compute targets using target network (inference backend, no grad)
        let next_state_tensors = encode_states_batch::<InferBackend>(&next_states, &self.device);
        let next_q_all = self.target_network.forward(next_state_tensors); // [B, 7]
        let next_q_data: Vec<f32> = next_q_all
            .into_data()
            .to_vec()
            .expect("f32 tensor data extraction");

        // For each experience, compute target = reward + gamma * max_legal_q (if not done)
        let mut target_data = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if dones[i] > 0.5 {
                target_data.push(rewards[i]);
            } else {
                // Find max Q over legal actions of next state
                let legal = next_states[i].legal_actions();
                let max_q = if legal.is_empty() {
                    0.0
                } else {
                    legal
                        .iter()
                        .map(|&col| next_q_data[i * 7 + col])
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                target_data.push(rewards[i] + self.config.gamma * max_q);
            }
        }

        let targets = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(target_data.as_slice()),
            &self.device,
        )
        .reshape([batch_size as i32, 1]);

        // MSE loss
        let diff = q_taken - targets;
        let loss = (diff.clone() * diff).mean();

        // Extract scalar loss value before backward
        let loss_val: f32 = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 loss tensor extraction")[0];

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.q_network);

        // Optimizer step: consumes q_network, returns updated one
        self.q_network =
            self.optimizer
                .step(self.config.learning_rate, self.q_network.clone(), grads);

        // Periodically sync target network
        self.step_count += 1;
        if self.step_count % self.config.target_update_interval == 0 {
            self.target_network = self.q_network.valid();
        }

        loss_val
    }

    /// Decay epsilon linearly over configured episodes.
    fn decay_epsilon(&mut self) {
        if self.config.epsilon_decay_episodes == 0 {
            self.epsilon = self.config.epsilon_end;
            return;
        }
        let progress = (self.episode_count as f32)
            / (self.config.epsilon_decay_episodes as f32);
        let progress = progress.min(1.0);
        self.epsilon = self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * progress;
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    /// Set epsilon directly (e.g. 0.0 for pure greedy inference).
    pub fn set_epsilon(&mut self, eps: f32) {
        self.epsilon = eps;
    }

    /// Save network weights to a directory.
    pub fn save_to_dir(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultRecorder::default();
        self.q_network
            .clone()
            .valid()
            .save_file(dir.join("q_network"), &recorder)?;
        self.target_network
            .clone()
            .save_file(dir.join("target_network"), &recorder)?;
        Ok(())
    }

    /// Load network weights from a directory.
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultRecorder::default();
        let net_config = DqnNetworkConfig {};

        let q: DqnNetwork<TrainBackend> = net_config
            .init(&self.device)
            .load_file(dir.join("q_network"), &recorder, &self.device)?;
        self.q_network = q;

        let target: DqnNetwork<InferBackend> = net_config
            .init(&self.device)
            .load_file(dir.join("target_network"), &recorder, &self.device)?;
        self.target_network = target;
        Ok(())
    }

    /// Export current training state for checkpointing.
    pub fn training_state(&self) -> DqnTrainingState {
        DqnTrainingState {
            epsilon: self.epsilon,
            step_count: self.step_count,
            episode_count: self.episode_count,
            learning_rate: self.config.learning_rate,
            gamma: self.config.gamma,
            epsilon_start: self.config.epsilon_start,
            epsilon_end: self.config.epsilon_end,
            epsilon_decay_episodes: self.config.epsilon_decay_episodes,
            target_update_interval: self.config.target_update_interval,
            batch_size: self.config.batch_size,
            replay_capacity: self.config.replay_capacity,
            min_replay_size: self.config.min_replay_size,
        }
    }

    /// Restore training state from a checkpoint.
    pub fn restore_training_state(&mut self, state: &DqnTrainingState) {
        self.epsilon = state.epsilon;
        self.step_count = state.step_count;
        self.episode_count = state.episode_count;
        self.config = DqnConfig {
            learning_rate: state.learning_rate,
            gamma: state.gamma,
            epsilon_start: state.epsilon_start,
            epsilon_end: state.epsilon_end,
            epsilon_decay_episodes: state.epsilon_decay_episodes,
            target_update_interval: state.target_update_interval,
            batch_size: state.batch_size,
            replay_capacity: state.replay_capacity,
            min_replay_size: state.min_replay_size,
        };
    }
}

impl Agent for DqnAgent {
    fn select_action(&mut self, state: &GameState, training: bool) -> usize {
        self.pick_action(state, training)
    }

    fn name(&self) -> &str {
        "DQN"
    }

    fn batch_update(&mut self, experiences: &[Experience]) -> UpdateMetrics {
        // Push all experiences into replay buffer
        for exp in experiences {
            self.replay_buffer.push(exp.clone());
        }

        self.episode_count += 1;
        self.decay_epsilon();

        // Train if enough experiences (guard against min_replay_size < batch_size)
        let threshold = self.config.min_replay_size.max(self.config.batch_size);
        if self.replay_buffer.len() >= threshold {
            let loss = self.train_step();
            UpdateMetrics {
                loss,
                ..Default::default()
            }
        } else {
            UpdateMetrics::default()
        }
    }

    fn current_metrics(&self) -> AgentMetrics {
        AgentMetrics {
            total_games: self.episode_count as u64,
            ..Default::default()
        }
    }
}

impl TrainableAgent for DqnAgent {
    fn algorithm_name(&self) -> &str {
        "DQN"
    }

    fn episode_count(&self) -> usize {
        self.episode_count
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn enter_eval_mode(&mut self) -> EvalState {
        let saved = self.epsilon;
        self.epsilon = 0.0;
        EvalState::Epsilon(saved)
    }

    fn exit_eval_mode(&mut self, state: EvalState) {
        if let EvalState::Epsilon(eps) = state {
            self.epsilon = eps;
        }
    }

    fn algorithm_metric_value(&self) -> f32 {
        self.epsilon
    }

    fn algorithm_metric_label(&self) -> &str {
        "eps"
    }

    fn last_policy_entropy(&self) -> Option<f32> {
        None
    }

    fn save_weights_to_dir(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.save_to_dir(dir)
    }

    fn training_state_json(&self) -> String {
        serde_json::to_string_pretty(&self.training_state()).expect("DQN training state serializes")
    }

    fn build_checkpoint_metadata(
        &self,
        metrics: &crate::checkpoint::CheckpointMetrics,
        episode: usize,
        timestamp: u64,
    ) -> CheckpointMetadata {
        let ts = self.training_state();
        CheckpointMetadata {
            episode,
            timestamp,
            algorithm: "DQN".to_string(),
            metrics: metrics.clone(),
            hyperparameters: CheckpointHyperparameters {
                learning_rate: ts.learning_rate,
                gamma: ts.gamma,
                epsilon: ts.epsilon,
                batch_size: ts.batch_size,
                target_update_interval: ts.target_update_interval,
                replay_capacity: ts.replay_capacity,
                min_replay_size: ts.min_replay_size,
                epsilon_start: ts.epsilon_start,
                epsilon_end: ts.epsilon_end,
                epsilon_decay_episodes: ts.epsilon_decay_episodes,
            },
            pg_hyperparameters: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;

    #[test]
    fn test_dqn_agent_selects_legal_action() {
        let mut agent = DqnAgent::new(DqnConfig::default());
        agent.set_epsilon(0.0); // greedy
        let state = GameState::initial();
        let legal = state.legal_actions();

        for _ in 0..10 {
            let action = agent.select_action(&state, false);
            assert!(legal.contains(&action), "Action {} is not legal", action);
        }
    }

    #[test]
    fn test_dqn_agent_batch_update_with_few_experiences() {
        let mut agent = DqnAgent::new(DqnConfig {
            min_replay_size: 5,
            batch_size: 2,
            ..Default::default()
        });

        let state = GameState::initial();
        let mut exps = Vec::new();
        for col in 0..7 {
            let next = state.apply_move(col).unwrap();
            exps.push(Experience {
                state: state.clone(),
                action: col,
                reward: 0.0,
                next_state: next,
                done: false,
                player: Player::Red,
            });
        }

        // First batch: not enough to train yet
        let metrics = agent.batch_update(&exps[..3]);
        assert_eq!(metrics.loss, 0.0);

        // Second batch: now we have enough
        let _metrics = agent.batch_update(&exps[3..]);
        // Should have trained (loss may be anything)
        assert!(agent.episode_count() == 2);
    }

    #[test]
    fn test_dqn_epsilon_decay() {
        let mut agent = DqnAgent::new(DqnConfig {
            epsilon_start: 1.0,
            epsilon_end: 0.1,
            epsilon_decay_episodes: 100,
            min_replay_size: 999_999, // prevent actual training
            ..Default::default()
        });

        // Simulate 50 episodes (half of decay)
        let dummy_exp = vec![Experience {
            state: GameState::initial(),
            action: 0,
            reward: 0.0,
            next_state: GameState::initial().apply_move(0).unwrap(),
            done: false,
            player: Player::Red,
        }];

        for _ in 0..50 {
            agent.batch_update(&dummy_exp);
        }

        // At 50/100 episodes, epsilon should be ~0.55
        let expected = 1.0 + (0.1 - 1.0) * 0.5;
        assert!(
            (agent.epsilon() - expected).abs() < 0.05,
            "epsilon {} not close to {}",
            agent.epsilon(),
            expected
        );
    }

    #[test]
    fn test_dqn_epsilon_decay_zero_episodes() {
        let mut agent = DqnAgent::new(DqnConfig {
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_episodes: 0,
            min_replay_size: 999_999,
            ..Default::default()
        });

        let dummy_exp = vec![Experience {
            state: GameState::initial(),
            action: 0,
            reward: 0.0,
            next_state: GameState::initial().apply_move(0).unwrap(),
            done: false,
            player: Player::Red,
        }];

        agent.batch_update(&dummy_exp);
        assert!(
            (agent.epsilon() - 0.05).abs() < 1e-6,
            "epsilon should jump to epsilon_end, got {}",
            agent.epsilon()
        );
    }

    #[test]
    fn test_dqn_no_panic_min_replay_less_than_batch() {
        let mut agent = DqnAgent::new(DqnConfig {
            min_replay_size: 2,
            batch_size: 10,
            ..Default::default()
        });

        let state = GameState::initial();
        let next = state.apply_move(0).unwrap();

        // Push 5 experiences (more than min_replay_size=2, but less than batch_size=10)
        let exps: Vec<Experience> = (0..5)
            .map(|_| Experience {
                state: state.clone(),
                action: 0,
                reward: 0.0,
                next_state: next.clone(),
                done: false,
                player: Player::Red,
            })
            .collect();

        // Should NOT panic despite min_replay_size < batch_size
        let metrics = agent.batch_update(&exps);
        assert_eq!(metrics.loss, 0.0, "should skip training when buffer < batch_size");
    }
}
