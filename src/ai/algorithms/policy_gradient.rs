use std::error::Error;
use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::module::AutodiffModule;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::DefaultRecorder;
use burn::tensor::TensorData;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::ai::agent::{Agent, AgentMetrics, EvalState, Experience, TrainableAgent, UpdateMetrics};
use crate::checkpoint::{CheckpointHyperparameters, CheckpointMetadata, PgHyperparameters};
use crate::ai::networks::{PolicyValueNetwork, PolicyValueNetworkConfig};
use crate::ai::state_encoding::{encode_state, encode_states_batch_into};
use crate::checkpoint::PgTrainingState;
use crate::game::{GameState, Player, ROWS, COLS};

type InferBackend = Wgpu<f32, i32>;
type TrainBackend = Autodiff<InferBackend>;

/// Policy Gradient hyperparameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PgConfig {
    pub learning_rate: f64,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub ppo_epsilon: f32,
    pub entropy_coeff: f32,
    pub value_coeff: f32,
    pub ppo_epochs: usize,
    pub max_grad_norm: f32,
    pub rollout_episodes: usize,
}

impl Default for PgConfig {
    fn default() -> Self {
        PgConfig {
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            ppo_epsilon: 0.2,
            entropy_coeff: 0.01,
            value_coeff: 0.5,
            ppo_epochs: 4,
            max_grad_norm: 0.5,
            rollout_episodes: 1,
        }
    }
}

/// Pre-computed per-player data for one episode, ready to concatenate for a PPO batch.
struct PlayerRolloutData {
    states: Vec<GameState>,
    actions: Vec<usize>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    old_log_probs: Vec<f32>,
}

/// Pre-allocated working buffers for PG training updates.
struct PgTrainBuffers {
    states: Vec<GameState>,
    actions: Vec<usize>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    old_log_probs: Vec<f32>,
    action_mask_data: Vec<f32>,
    legal_mask_data: Vec<f32>,
    flat_encode: Vec<f32>, // capacity: max_batch * 3 * ROWS * COLS
}

impl PgTrainBuffers {
    fn new(rollout_episodes: usize) -> Self {
        let cap = rollout_episodes * 21; // 21 = ceil(42 moves / 2 players)
        PgTrainBuffers {
            states: Vec::with_capacity(cap),
            actions: Vec::with_capacity(cap),
            advantages: Vec::with_capacity(cap),
            returns: Vec::with_capacity(cap),
            old_log_probs: Vec::with_capacity(cap),
            action_mask_data: Vec::with_capacity(cap * 7),
            legal_mask_data: Vec::with_capacity(cap * 7),
            flat_encode: Vec::with_capacity(cap * 3 * ROWS * COLS),
        }
    }
}

/// Policy Gradient agent with PPO clipping and GAE.
pub struct PolicyGradientAgent {
    network: PolicyValueNetwork<TrainBackend>,
    optimizer:
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, PolicyValueNetwork<TrainBackend>, TrainBackend>,
    config: PgConfig,
    device: <TrainBackend as Backend>::Device,
    episode_count: usize,
    step_count: usize,
    last_entropy: Option<f32>,
    rng: StdRng,
    episode_buffer: Vec<Vec<Experience>>,
    buffers: PgTrainBuffers,
}

impl PolicyGradientAgent {
    pub fn new(config: PgConfig) -> Self {
        let device = Default::default();
        let net_config = PolicyValueNetworkConfig {};
        let network: PolicyValueNetwork<TrainBackend> = net_config.init(&device);
        let optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)))
            .init();

        let rollout_episodes = config.rollout_episodes;
        PolicyGradientAgent {
            network,
            optimizer,
            config,
            device,
            episode_count: 0,
            step_count: 0,
            last_entropy: None,
            rng: StdRng::from_os_rng(),
            episode_buffer: Vec::new(),
            buffers: PgTrainBuffers::new(rollout_episodes),
        }
    }

    /// Select action using the policy network.
    fn pick_action(&mut self, state: &GameState, training: bool) -> usize {
        let legal = state.legal_actions();
        assert!(!legal.is_empty(), "No legal actions");

        // Forward pass through inference network
        let state_tensor =
            encode_state::<InferBackend>(state, &self.device).unsqueeze::<4>(); // [1, 3, 6, 7]
        let (logits, _value) = self.network.valid().forward(state_tensor);
        let logits_vec: Vec<f32> = logits.into_data().to_vec().expect("f32 tensor data extraction");

        // Apply legal action mask and compute softmax
        let probs = masked_softmax(&logits_vec, &legal);

        if training {
            // Sample from categorical distribution
            sample_categorical(&probs, &legal, &mut self.rng)
        } else {
            // Greedy: argmax over legal actions
            let mut best_action = legal[0];
            let mut best_prob = f32::NEG_INFINITY;
            for &col in &legal {
                if probs[col] > best_prob {
                    best_prob = probs[col];
                    best_action = col;
                }
            }
            best_action
        }
    }

    /// Compute pre-processed rollout data for one player's sub-trajectory in one episode.
    /// GAE is computed within this single episode's trace (no cross-episode bleed).
    fn compute_player_rollout_data(&mut self, exps: &[&Experience]) -> PlayerRolloutData {
        let values = self.compute_values_no_grad(exps);
        let (advantages, returns) = self.compute_gae(exps, &values);
        let old_log_probs = self.compute_log_probs_no_grad(exps);
        let states: Vec<GameState> = exps.iter().map(|e| e.state.clone()).collect();
        let actions: Vec<usize> = exps.iter().map(|e| e.action).collect();
        PlayerRolloutData { states, actions, advantages, returns, old_log_probs }
    }

    /// PPO update over a concatenated batch of per-episode player rollouts.
    fn ppo_update_for_player_from_data(&mut self, rollouts: &[PlayerRolloutData]) -> (f32, f32) {
        self.buffers.states.clear();
        self.buffers.actions.clear();
        self.buffers.advantages.clear();
        self.buffers.returns.clear();
        self.buffers.old_log_probs.clear();

        for rollout in rollouts {
            self.buffers.states.extend_from_slice(&rollout.states);
            self.buffers.actions.extend_from_slice(&rollout.actions);
            self.buffers.advantages.extend_from_slice(&rollout.advantages);
            self.buffers.returns.extend_from_slice(&rollout.returns);
            self.buffers.old_log_probs.extend_from_slice(&rollout.old_log_probs);
        }

        let n = self.buffers.states.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Re-normalize advantages across the full concatenated batch for better statistics
        if n > 1 {
            let mean: f32 = self.buffers.advantages.iter().sum::<f32>() / n as f32;
            let var: f32 =
                self.buffers.advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
            let std = var.sqrt().max(1e-8);
            for a in &mut self.buffers.advantages {
                *a = (*a - mean) / std;
            }
        }

        // Pre-compute masks (constant across PPO epochs)
        self.buffers.action_mask_data.resize(n * 7, 0.0f32);
        self.buffers.action_mask_data.fill(0.0f32);
        for (i, &action) in self.buffers.actions.iter().enumerate() {
            self.buffers.action_mask_data[i * 7 + action] = 1.0;
        }

        self.buffers.legal_mask_data.resize(n * 7, -1e9f32);
        self.buffers.legal_mask_data.fill(-1e9f32);
        for i in 0..n {
            let legal = self.buffers.states[i].legal_actions();
            for &col in &legal {
                self.buffers.legal_mask_data[i * 7 + col] = 0.0;
            }
        }

        // Encode all states once directly on GPU (no CPU roundtrip)
        let state_tensor_base = encode_states_batch_into::<TrainBackend>(
            &self.buffers.states,
            &mut self.buffers.flat_encode,
            &self.device,
        );

        // Pre-compute constant tensors (hoisted out of PPO loop; clone is shallow/Arc)
        let action_mask_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(self.buffers.action_mask_data.as_slice()),
            &self.device,
        )
        .reshape([n as i32, 7]);

        let legal_mask_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(self.buffers.legal_mask_data.as_slice()),
            &self.device,
        )
        .reshape([n as i32, 7]);

        let old_lp_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(self.buffers.old_log_probs.as_slice()),
            &self.device,
        );

        let adv_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(self.buffers.advantages.as_slice()),
            &self.device,
        );

        let returns_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(self.buffers.returns.as_slice()),
            &self.device,
        )
        .reshape([n as i32, 1]);

        // PPO optimization loop
        let mut last_loss = 0.0f32;
        let mut last_entropy = 0.0f32;

        for _epoch in 0..self.config.ppo_epochs {
            let (logits_batch, values_batch) =
                self.network.forward(state_tensor_base.clone());

            // Apply legal action mask to logits for proper softmax
            let masked_logits = logits_batch.clone() + legal_mask_tensor.clone();
            let log_probs_tensor = burn::tensor::activation::log_softmax(masked_logits.clone(), 1);

            // Extract log pi(a|s) for selected actions: [n]
            let selected_log_probs = (log_probs_tensor.clone() * action_mask_tensor.clone())
                .sum_dim(1)
                .reshape([n as i32]);

            // Ratio r = exp(log_pi_new - log_pi_old)
            let ratio_tensor = (selected_log_probs - old_lp_tensor.clone()).exp();

            let surr1 = ratio_tensor.clone() * adv_tensor.clone();
            let clamped_ratio = ratio_tensor.clamp(
                1.0 - self.config.ppo_epsilon,
                1.0 + self.config.ppo_epsilon,
            );
            let surr2 = clamped_ratio * adv_tensor.clone();

            // PPO clipped objective: min(surr1, surr2)
            // min(a, b) = (a + b - |a - b|) / 2
            let diff = surr1.clone() - surr2.clone();
            let abs_diff = diff.clone() * diff.sign();
            let policy_objective = (surr1 + surr2 - abs_diff) / 2.0;
            let policy_loss = -policy_objective.mean();

            // Value loss: MSE(predicted_value, returns)
            let value_diff = values_batch - returns_tensor.clone();
            let value_loss = (value_diff.clone() * value_diff).mean();

            // Extract logits data for entropy reporting before consuming logits_batch
            let logits_data: Vec<f32> =
                logits_batch.clone().into_data().to_vec().expect("f32 tensor data extraction");

            // Entropy bonus from log_probs_tensor (use masked_logits for consistent distributions)
            let probs_tensor = burn::tensor::activation::softmax(masked_logits, 1);
            let entropy_tensor = -(probs_tensor * log_probs_tensor).sum_dim(1).mean();

            // Total loss: policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            let total_loss = policy_loss
                + value_loss * self.config.value_coeff
                - entropy_tensor * self.config.entropy_coeff;

            last_loss = total_loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("f32 loss tensor extraction")[0];

            // Compute entropy scalar for reporting (from detached data)
            let mut entropy_sum = 0.0f32;
            for i in 0..n {
                let logits_i: [f32; 7] = std::array::from_fn(|j| logits_data[i * 7 + j]);
                let legal = self.buffers.states[i].legal_actions();
                let (lp, pr) = masked_log_softmax(&logits_i, &legal);
                entropy_sum += legal.iter().map(|&a| -pr[a] * lp[a]).sum::<f32>();
            }
            last_entropy = entropy_sum / n as f32;

            // Backward pass and optimizer step (skip if loss is non-finite to avoid corrupting weights)
            if last_loss.is_finite() {
                let grads = total_loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.network);
                self.network = self.optimizer.step(
                    self.config.learning_rate,
                    self.network.clone(),
                    grads,
                );
            }
        }

        (last_loss, last_entropy)
    }

    /// Combined PPO update over multiple buffered episodes.
    /// GAE is computed per-episode independently to avoid cross-episode bleed.
    fn ppo_update_multi(&mut self, episodes: &[Vec<Experience>]) -> UpdateMetrics {
        let mut red_rollouts: Vec<PlayerRolloutData> = Vec::with_capacity(episodes.len());
        let mut yellow_rollouts: Vec<PlayerRolloutData> = Vec::with_capacity(episodes.len());

        for episode in episodes {
            let red_exps: Vec<&Experience> =
                episode.iter().filter(|e| e.player == Player::Red).collect();
            let yellow_exps: Vec<&Experience> =
                episode.iter().filter(|e| e.player == Player::Yellow).collect();

            if !red_exps.is_empty() {
                red_rollouts.push(self.compute_player_rollout_data(&red_exps));
            }
            if !yellow_exps.is_empty() {
                yellow_rollouts.push(self.compute_player_rollout_data(&yellow_exps));
            }
        }

        let mut total_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0;

        for rollouts in [red_rollouts.as_slice(), yellow_rollouts.as_slice()] {
            if rollouts.is_empty() {
                continue;
            }
            let (loss, entropy) = self.ppo_update_for_player_from_data(rollouts);
            total_loss += loss;
            total_entropy += entropy;
            update_count += 1;
        }

        self.step_count += 1;

        if update_count > 0 {
            UpdateMetrics {
                loss: total_loss / update_count as f32,
                policy_entropy: Some(total_entropy / update_count as f32),
                ..Default::default()
            }
        } else {
            UpdateMetrics::default()
        }
    }

    /// Compute state values for experiences using inference network (no gradient).
    fn compute_values_no_grad(&mut self, exps: &[&Experience]) -> Vec<f32> {
        if exps.is_empty() {
            return Vec::new();
        }
        self.buffers.states.clear();
        self.buffers.states.extend(exps.iter().map(|e| e.state.clone()));
        let state_batch = encode_states_batch_into::<InferBackend>(
            &self.buffers.states,
            &mut self.buffers.flat_encode,
            &self.device,
        );
        let infer_net = self.network.valid();
        let (_logits, values_batch) = infer_net.forward(state_batch);
        values_batch
            .reshape([exps.len() as i32])
            .into_data()
            .to_vec()
            .expect("f32 tensor data extraction")
    }

    /// Compute log probabilities of taken actions (no gradient).
    fn compute_log_probs_no_grad(&mut self, exps: &[&Experience]) -> Vec<f32> {
        if exps.is_empty() {
            return Vec::new();
        }
        self.buffers.states.clear();
        self.buffers.states.extend(exps.iter().map(|e| e.state.clone()));
        let state_batch = encode_states_batch_into::<InferBackend>(
            &self.buffers.states,
            &mut self.buffers.flat_encode,
            &self.device,
        );
        let infer_net = self.network.valid();
        let (logits_batch, _) = infer_net.forward(state_batch);
        let logits_data: Vec<f32> = logits_batch
            .into_data()
            .to_vec()
            .expect("f32 tensor data extraction");

        let mut log_probs = Vec::with_capacity(exps.len());
        for (i, exp) in exps.iter().enumerate() {
            let logits_i: [f32; 7] = std::array::from_fn(|j| logits_data[i * 7 + j]);
            let legal = exp.state.legal_actions();
            let (lp, _) = masked_log_softmax(&logits_i, &legal);
            log_probs.push(lp[exp.action]);
        }

        log_probs
    }

    /// Compute Generalized Advantage Estimation (GAE).
    fn compute_gae(&self, exps: &[&Experience], values: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = exps.len();
        let mut advantages = vec![0.0f32; n];
        let mut returns = vec![0.0f32; n];

        let gamma = self.config.gamma;
        let lam = self.config.gae_lambda;

        // Process backwards
        let mut gae = 0.0f32;
        for i in (0..n).rev() {
            let next_value = if exps[i].done {
                0.0
            } else if i + 1 < n {
                values[i + 1]
            } else {
                0.0
            };

            let delta = exps[i].reward + gamma * next_value - values[i];
            gae = if exps[i].done {
                delta
            } else {
                delta + gamma * lam * gae
            };

            advantages[i] = gae;
            returns[i] = gae + values[i];
        }

        // Normalize advantages
        if n > 1 {
            let mean: f32 = advantages.iter().sum::<f32>() / n as f32;
            let var: f32 =
                advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
            let std = var.sqrt().max(1e-8);
            for a in &mut advantages {
                *a = (*a - mean) / std;
            }
        }

        (advantages, returns)
    }

    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Save network weights to a directory.
    pub fn save_to_dir(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultRecorder::default();
        self.network
            .clone()
            .valid()
            .save_file(dir.join("policy_value_network"), &recorder)?;
        Ok(())
    }

    /// Load network weights from a directory.
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultRecorder::default();
        let net_config = PolicyValueNetworkConfig {};
        let net: PolicyValueNetwork<TrainBackend> = net_config
            .init(&self.device)
            .load_file(dir.join("policy_value_network"), &recorder, &self.device)?;
        self.network = net;
        Ok(())
    }

    /// Export current training state for checkpointing.
    pub fn training_state(&self) -> PgTrainingState {
        PgTrainingState {
            episode_count: self.episode_count,
            step_count: self.step_count,
            learning_rate: self.config.learning_rate,
            gamma: self.config.gamma,
            gae_lambda: self.config.gae_lambda,
            ppo_epsilon: self.config.ppo_epsilon,
            entropy_coeff: self.config.entropy_coeff,
            value_coeff: self.config.value_coeff,
            ppo_epochs: self.config.ppo_epochs,
            max_grad_norm: self.config.max_grad_norm,
            rollout_episodes: self.config.rollout_episodes,
        }
    }

    /// Restore training state from a checkpoint.
    pub fn restore_training_state(&mut self, state: &PgTrainingState) {
        self.episode_count = state.episode_count;
        self.step_count = state.step_count;
        self.config = PgConfig {
            learning_rate: state.learning_rate,
            gamma: state.gamma,
            gae_lambda: state.gae_lambda,
            ppo_epsilon: state.ppo_epsilon,
            entropy_coeff: state.entropy_coeff,
            value_coeff: state.value_coeff,
            ppo_epochs: state.ppo_epochs,
            max_grad_norm: state.max_grad_norm,
            rollout_episodes: state.rollout_episodes,
        };
        // Re-create optimizer so restored max_grad_norm takes effect
        self.optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(self.config.max_grad_norm)))
            .init();
    }
}

/// Inference-only, Send-safe PG agent for parallel evaluation.
struct PgEvalAgent {
    network: PolicyValueNetwork<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl Agent for PgEvalAgent {
    fn select_action(&mut self, state: &GameState, _training: bool) -> usize {
        let legal = state.legal_actions();
        assert!(!legal.is_empty(), "No legal actions");
        let state_tensor =
            encode_state::<InferBackend>(state, &self.device).unsqueeze::<4>();
        let (logits, _value) = self.network.forward(state_tensor);
        let logits_vec: Vec<f32> = logits
            .into_data()
            .to_vec()
            .expect("f32 tensor data extraction");
        let probs = masked_softmax(&logits_vec, &legal);
        let mut best_action = legal[0];
        let mut best_prob = f32::NEG_INFINITY;
        for &col in &legal {
            if probs[col] > best_prob {
                best_prob = probs[col];
                best_action = col;
            }
        }
        best_action
    }

    fn name(&self) -> &str {
        "PgEval"
    }
}

impl Agent for PolicyGradientAgent {
    fn select_action(&mut self, state: &GameState, training: bool) -> usize {
        self.pick_action(state, training)
    }

    fn name(&self) -> &str {
        "PG"
    }

    fn batch_update(&mut self, experiences: &[Experience]) -> UpdateMetrics {
        self.episode_count += 1;
        self.episode_buffer.push(experiences.to_vec());
        if self.episode_buffer.len() < self.config.rollout_episodes {
            return UpdateMetrics::default();
        }
        let episodes = std::mem::take(&mut self.episode_buffer);
        let metrics = self.ppo_update_multi(&episodes);
        if let Some(ent) = metrics.policy_entropy {
            self.last_entropy = Some(ent);
        }
        metrics
    }

    fn current_metrics(&self) -> AgentMetrics {
        AgentMetrics {
            total_games: self.episode_count as u64,
            ..Default::default()
        }
    }
}

impl TrainableAgent for PolicyGradientAgent {
    fn algorithm_name(&self) -> &str {
        "PG"
    }

    fn episode_count(&self) -> usize {
        self.episode_count
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn enter_eval_mode(&mut self) -> EvalState {
        EvalState::NoOp
    }

    fn exit_eval_mode(&mut self, _state: EvalState) {}

    fn algorithm_metric_value(&self) -> f32 {
        0.0
    }

    fn algorithm_metric_label(&self) -> &str {
        "entropy"
    }

    fn last_policy_entropy(&self) -> Option<f32> {
        self.last_entropy
    }

    fn save_weights_to_dir(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.save_to_dir(dir)
    }

    fn training_state_json(&self) -> String {
        serde_json::to_string_pretty(&self.training_state()).expect("PG training state serializes")
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
            algorithm: "PG".to_string(),
            metrics: metrics.clone(),
            hyperparameters: CheckpointHyperparameters {
                learning_rate: ts.learning_rate,
                gamma: ts.gamma,
                epsilon: 0.0,
                batch_size: 0,
                target_update_interval: 0,
                replay_capacity: 0,
                min_replay_size: 0,
                epsilon_start: 0.0,
                epsilon_end: 0.0,
                epsilon_decay_episodes: 0,
            },
            pg_hyperparameters: Some(PgHyperparameters {
                learning_rate: ts.learning_rate,
                gamma: ts.gamma,
                gae_lambda: ts.gae_lambda,
                ppo_epsilon: ts.ppo_epsilon,
                entropy_coeff: ts.entropy_coeff,
                value_coeff: ts.value_coeff,
                ppo_epochs: ts.ppo_epochs,
                max_grad_norm: ts.max_grad_norm,
                rollout_episodes: ts.rollout_episodes,
            }),
            az_hyperparameters: None,
        }
    }

    fn load_weights_from_dir(&mut self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.load_from_dir(dir)
    }

    fn restore_training_state_json(&mut self, json: &str) -> Result<(), Box<dyn std::error::Error>> {
        let state: PgTrainingState = serde_json::from_str(json)?;
        self.restore_training_state(&state);
        Ok(())
    }

    fn clone_for_eval(&self) -> Box<dyn Agent + Send> {
        Box::new(PgEvalAgent {
            network: self.network.clone().valid(),
            device: self.device.clone(),
        })
    }
}

/// Apply legal action mask and compute softmax probabilities.
fn masked_softmax(logits: &[f32], legal: &[usize]) -> [f32; 7] {
    let mut masked = [f32::NEG_INFINITY; 7];
    for &col in legal {
        masked[col] = logits[col];
    }

    // Numerically stable softmax
    let max_val = masked.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Guard against NaN logits: fall back to uniform over legal actions
    if !max_val.is_finite() {
        return uniform_over_legal(legal);
    }

    let mut probs = [0.0f32; 7];
    let mut sum = 0.0f32;
    for (i, &m) in masked.iter().enumerate() {
        let v = (m - max_val).exp();
        probs[i] = v;
        sum += v;
    }

    if !sum.is_finite() || sum == 0.0 {
        return uniform_over_legal(legal);
    }

    for p in &mut probs {
        *p /= sum;
    }

    probs
}

/// Return a uniform probability distribution over legal actions.
fn uniform_over_legal(legal: &[usize]) -> [f32; 7] {
    let mut probs = [0.0f32; 7];
    let p = 1.0 / legal.len() as f32;
    for &col in legal {
        probs[col] = p;
    }
    probs
}

/// Compute masked log-softmax. Returns (log_probs, probs).
fn masked_log_softmax(logits: &[f32], legal: &[usize]) -> ([f32; 7], [f32; 7]) {
    let probs = masked_softmax(logits, legal);
    let mut log_probs = [0.0f32; 7];
    for (i, &p) in probs.iter().enumerate() {
        log_probs[i] = if p > 0.0 { p.ln() } else { -1e9 };
    }
    (log_probs, probs)
}

/// Sample an action from a categorical distribution defined by probs.
/// Falls back to a random legal action if sampling fails (e.g. NaN probs).
fn sample_categorical(probs: &[f32], legal: &[usize], rng: &mut StdRng) -> usize {
    let r: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    // Fallback to random legal action (guards against NaN probs)
    legal[rng.random_range(0..legal.len())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pg_eval_agent_selects_legal_action() {
        let agent = PolicyGradientAgent::new(PgConfig::default());
        let mut eval_agent = agent.clone_for_eval();
        let state = GameState::initial();
        let legal = state.legal_actions();
        for _ in 0..10 {
            let action = eval_agent.select_action(&state, false);
            assert!(legal.contains(&action), "Action {} is not legal", action);
        }
    }

    #[test]
    fn test_pg_agent_selects_legal_action() {
        let mut agent = PolicyGradientAgent::new(PgConfig::default());
        let state = GameState::initial();
        let legal = state.legal_actions();

        for _ in 0..10 {
            let action = agent.select_action(&state, false);
            assert!(legal.contains(&action), "Action {} is not legal", action);
        }
    }

    #[test]
    fn test_pg_agent_training_mode_explores() {
        let mut agent = PolicyGradientAgent::new(PgConfig::default());
        let state = GameState::initial();

        let mut actions = std::collections::HashSet::new();
        for _ in 0..100 {
            let action = agent.select_action(&state, true);
            actions.insert(action);
        }
        // Should explore multiple columns
        assert!(
            actions.len() > 1,
            "Expected exploration across multiple actions, got {:?}",
            actions
        );
    }

    #[test]
    fn test_pg_batch_update_runs() {
        let mut agent = PolicyGradientAgent::new(PgConfig {
            ppo_epochs: 1,
            ..Default::default()
        });
        let state = GameState::initial();

        let mut exps = Vec::new();
        // Build a short trajectory
        let s0 = state.clone();
        let s1 = s0.apply_move(3).unwrap();
        let s2 = s1.apply_move(4).unwrap();
        let s3 = s2.apply_move(3).unwrap();

        exps.push(Experience {
            state: s0.clone(),
            action: 3,
            reward: 0.0,
            next_state: s1.clone(),
            done: false,
            player: Player::Red,
        });
        exps.push(Experience {
            state: s1.clone(),
            action: 4,
            reward: 0.0,
            next_state: s2.clone(),
            done: false,
            player: Player::Yellow,
        });
        exps.push(Experience {
            state: s2.clone(),
            action: 3,
            reward: 1.0,
            next_state: s3.clone(),
            done: true,
            player: Player::Red,
        });

        let metrics = agent.batch_update(&exps);
        // Should not panic, and episode_count should increment
        assert_eq!(agent.episode_count(), 1);
        assert!(metrics.policy_entropy.is_some());
    }

    #[test]
    fn test_gae_single_terminal_step() {
        let agent = PolicyGradientAgent::new(PgConfig {
            gamma: 0.99,
            gae_lambda: 0.95,
            ..Default::default()
        });

        let state = GameState::initial();
        let next = state.apply_move(0).unwrap();

        let exp = Experience {
            state: state.clone(),
            action: 0,
            reward: 1.0,
            next_state: next,
            done: true,
            player: Player::Red,
        };

        let exps: Vec<&Experience> = vec![&exp];
        let values = vec![0.5]; // V(s) = 0.5

        let (advantages, returns) = agent.compute_gae(&exps, &values);

        // delta = reward + gamma * 0 (done) - V(s) = 1.0 - 0.5 = 0.5
        // GAE = delta (single step, done) = 0.5
        // With normalization of a single value, advantage becomes 0
        // returns = advantages + values = 0.5 + 0.5 = 1.0
        assert_eq!(advantages.len(), 1);
        assert_eq!(returns.len(), 1);
        // returns should be advantage + value = delta + value = reward = 1.0
        assert!((returns[0] - 1.0).abs() < 1e-5, "returns[0] = {}", returns[0]);
    }

    fn make_short_trajectory() -> Vec<Experience> {
        let s0 = GameState::initial();
        let s1 = s0.apply_move(3).unwrap();
        let s2 = s1.apply_move(4).unwrap();
        let s3 = s2.apply_move(3).unwrap();
        vec![
            Experience {
                state: s0.clone(),
                action: 3,
                reward: 0.0,
                next_state: s1.clone(),
                done: false,
                player: Player::Red,
            },
            Experience {
                state: s1.clone(),
                action: 4,
                reward: 0.0,
                next_state: s2.clone(),
                done: false,
                player: Player::Yellow,
            },
            Experience {
                state: s2.clone(),
                action: 3,
                reward: 1.0,
                next_state: s3.clone(),
                done: true,
                player: Player::Red,
            },
        ]
    }

    #[test]
    fn test_pg_rollout_buffering_defers_update() {
        let mut agent = PolicyGradientAgent::new(PgConfig {
            ppo_epochs: 1,
            rollout_episodes: 2,
            ..Default::default()
        });
        let exps = make_short_trajectory();

        // First episode: buffered — no update yet
        let metrics1 = agent.batch_update(&exps);
        assert_eq!(agent.step_count(), 0);
        assert_eq!(metrics1.loss, 0.0);
        assert!(metrics1.policy_entropy.is_none());

        // Second episode: buffer full — update fires
        let metrics2 = agent.batch_update(&exps);
        assert_eq!(agent.step_count(), 1);
        assert!(metrics2.policy_entropy.is_some());
    }

    #[test]
    fn test_pg_rollout_multi_episode_update() {
        let mut agent = PolicyGradientAgent::new(PgConfig {
            ppo_epochs: 1,
            rollout_episodes: 3,
            ..Default::default()
        });
        let exps = make_short_trajectory();

        agent.batch_update(&exps);
        agent.batch_update(&exps);
        let metrics = agent.batch_update(&exps);

        assert_eq!(agent.episode_count(), 3);
        assert_eq!(agent.step_count(), 1);
        assert!(metrics.policy_entropy.is_some());
    }

    #[test]
    fn test_gae_no_cross_episode_bleed() {
        // Validates why compute_player_rollout_data must be called per-episode:
        // a Red experience with done=false (Yellow made the terminal move) has
        // return == -1.0 when GAE is computed in isolation, but a different value
        // when naively concatenated with the next episode.
        let agent = PolicyGradientAgent::new(PgConfig {
            gamma: 0.99,
            gae_lambda: 0.95,
            ..Default::default()
        });

        let s0 = GameState::initial();
        let s1 = s0.apply_move(0).unwrap();
        let s2 = s1.apply_move(1).unwrap();

        // Red's last experience in ep1: game ended by Yellow's move, done=false
        let exp1 = Experience {
            state: s0.clone(),
            action: 0,
            reward: -1.0,
            next_state: s1.clone(),
            done: false,
            player: Player::Red,
        };

        // Red's first experience in ep2
        let exp2 = Experience {
            state: s1.clone(),
            action: 1,
            reward: 0.0,
            next_state: s2.clone(),
            done: false,
            player: Player::Red,
        };

        // Per-episode GAE: only exp1 — return should be -1.0
        let ep1_exps = vec![&exp1];
        let values_ep1 = vec![0.0f32]; // V(s0) = 0
        let (_adv_ep1, returns_ep1) = agent.compute_gae(&ep1_exps, &values_ep1);
        assert!(
            (returns_ep1[0] - (-1.0)).abs() < 1e-5,
            "per-episode return = {} (expected -1.0)",
            returns_ep1[0]
        );

        // Naive 2-episode concat: exp1 followed by exp2
        let both_exps = vec![&exp1, &exp2];
        let values_both = vec![0.0f32, 0.5f32]; // V(s0)=0, V(s1)=0.5
        let (_adv_both, returns_both) = agent.compute_gae(&both_exps, &values_both);
        // With bleed: exp1's next_value = V(s1) = 0.5, so return != -1.0
        assert!(
            (returns_both[0] - (-1.0)).abs() > 1e-3,
            "naive-concat return = {} (should differ from -1.0 to prove bleed)",
            returns_both[0]
        );
    }

    #[test]
    fn test_masked_softmax_legal_actions() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let legal = vec![0, 2, 4, 6];
        let probs = masked_softmax(&logits, &legal);

        // Non-legal actions should have ~0 probability
        assert!(probs[1] < 1e-6);
        assert!(probs[3] < 1e-6);
        assert!(probs[5] < 1e-6);

        // Legal actions should have positive probability
        assert!(probs[0] > 0.0);
        assert!(probs[2] > 0.0);
        assert!(probs[4] > 0.0);
        assert!(probs[6] > 0.0);

        // Sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    }
}
