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

use crate::ai::agent::{Agent, AgentMetrics, Experience, UpdateMetrics};
use crate::ai::networks::{PolicyValueNetwork, PolicyValueNetworkConfig};
use crate::ai::state_encoding::encode_state;
use crate::checkpoint::PgTrainingState;
use crate::game::{GameState, Player};

type InferBackend = Wgpu<f32, i32>;
type TrainBackend = Autodiff<InferBackend>;

/// Policy Gradient hyperparameters.
#[derive(Debug, Clone)]
pub struct PgConfig {
    pub learning_rate: f64,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub ppo_epsilon: f32,
    pub entropy_coeff: f32,
    pub value_coeff: f32,
    pub ppo_epochs: usize,
    pub max_grad_norm: f32,
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
    rng: StdRng,
}

impl PolicyGradientAgent {
    pub fn new(config: PgConfig) -> Self {
        let device = Default::default();
        let net_config = PolicyValueNetworkConfig {};
        let network: PolicyValueNetwork<TrainBackend> = net_config.init(&device);
        let optimizer = AdamConfig::new().init();

        PolicyGradientAgent {
            network,
            optimizer,
            config,
            device,
            episode_count: 0,
            step_count: 0,
            rng: StdRng::from_os_rng(),
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
        let logits_vec: Vec<f32> = logits.into_data().to_vec().unwrap();

        // Apply legal action mask and compute softmax
        let probs = masked_softmax(&logits_vec, &legal);

        if training {
            // Sample from categorical distribution
            sample_categorical(&probs, &mut self.rng)
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

    /// PPO-style batch update from a full episode trajectory.
    fn ppo_update(&mut self, experiences: &[Experience]) -> UpdateMetrics {
        if experiences.is_empty() {
            return UpdateMetrics::default();
        }

        // Split experiences by player for proper advantage computation
        let red_exps: Vec<&Experience> = experiences
            .iter()
            .filter(|e| e.player == Player::Red)
            .collect();
        let yellow_exps: Vec<&Experience> = experiences
            .iter()
            .filter(|e| e.player == Player::Yellow)
            .collect();

        let mut total_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut update_count = 0;

        for player_exps in [&red_exps, &yellow_exps] {
            if player_exps.is_empty() {
                continue;
            }

            let (loss, entropy) = self.ppo_update_for_player(player_exps);
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

    /// PPO update for a single player's sub-trajectory.
    fn ppo_update_for_player(&mut self, exps: &[&Experience]) -> (f32, f32) {
        let n = exps.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Compute values for all states using current network (inference mode, no grad)
        let values = self.compute_values_no_grad(exps);

        // Compute GAE advantages and returns
        let (advantages, returns) = self.compute_gae(exps, &values);

        // Compute old log probs (no grad)
        let old_log_probs = self.compute_log_probs_no_grad(exps);

        // Pre-compute masks (constant across PPO epochs)
        let mut action_mask_data = vec![0.0f32; n * 7];
        for (i, exp) in exps.iter().enumerate() {
            action_mask_data[i * 7 + exp.action] = 1.0;
        }
        let mut legal_mask_data = vec![-1e9f32; n * 7];
        for (i, exp) in exps.iter().enumerate() {
            for &col in &exp.state.legal_actions() {
                legal_mask_data[i * 7 + col] = 0.0;
            }
        }

        // Encode all states once
        let state_data: Vec<f32> = exps
            .iter()
            .flat_map(|e| {
                let t = encode_state::<TrainBackend>(&e.state, &self.device);
                let data: Vec<f32> = t.into_data().to_vec().unwrap();
                data
            })
            .collect();

        // PPO optimization loop
        let mut last_loss = 0.0f32;
        let mut last_entropy = 0.0f32;

        for _epoch in 0..self.config.ppo_epochs {
            let state_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::from(state_data.as_slice()),
                &self.device,
            )
            .reshape([n as i32, 3, 6, 7]);

            let (logits_batch, values_batch) = self.network.forward(state_tensor);

            // Apply legal action mask to logits for proper softmax
            let mask_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::from(legal_mask_data.as_slice()),
                &self.device,
            )
            .reshape([n as i32, 7]);

            let masked_logits = logits_batch.clone() + mask_tensor;
            let log_probs_tensor = burn::tensor::activation::log_softmax(masked_logits, 1);

            // Extract log pi(a|s) for selected actions: [n]
            let action_mask_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::from(action_mask_data.as_slice()),
                &self.device,
            )
            .reshape([n as i32, 7]);
            let selected_log_probs =
                (log_probs_tensor.clone() * action_mask_tensor).sum_dim(1).reshape([n as i32]);

            // Old log probs (detached, no grad)
            let old_lp_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::from(old_log_probs.as_slice()),
                &self.device,
            );

            // Ratio r = exp(log_pi_new - log_pi_old)
            let ratio_tensor = (selected_log_probs - old_lp_tensor).exp();

            // Advantages tensor (detached)
            let adv_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::from(advantages.as_slice()),
                &self.device,
            );

            let surr1 = ratio_tensor.clone() * adv_tensor.clone();
            let clamped_ratio = ratio_tensor.clamp(
                1.0 - self.config.ppo_epsilon,
                1.0 + self.config.ppo_epsilon,
            );
            let surr2 = clamped_ratio * adv_tensor;

            // PPO clipped objective: min(surr1, surr2)
            // min(a, b) = (a + b - |a - b|) / 2
            let diff = surr1.clone() - surr2.clone();
            let abs_diff = diff.clone() * diff.sign();
            let policy_objective = (surr1 + surr2 - abs_diff) / 2.0;
            let policy_loss = -policy_objective.mean();

            // Value loss: MSE(predicted_value, returns)
            let returns_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::from(returns.as_slice()),
                &self.device,
            )
            .reshape([n as i32, 1]);
            let value_diff = values_batch - returns_tensor;
            let value_loss = (value_diff.clone() * value_diff).mean();

            // Extract logits data for entropy reporting before consuming logits_batch
            let logits_data: Vec<f32> = logits_batch.clone().into_data().to_vec().unwrap();

            // Entropy bonus from log_probs_tensor
            let probs_tensor = burn::tensor::activation::softmax(logits_batch, 1);
            let entropy_tensor = -(probs_tensor * log_probs_tensor).sum_dim(1).mean();

            // Total loss: policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            let total_loss = policy_loss
                + value_loss * self.config.value_coeff
                - entropy_tensor * self.config.entropy_coeff;

            last_loss = total_loss.clone().into_data().to_vec::<f32>().unwrap()[0];

            // Compute entropy scalar for reporting (from detached data)
            let mut entropy_sum = 0.0f32;
            for i in 0..n {
                let logits_i: Vec<f32> = (0..7).map(|j| logits_data[i * 7 + j]).collect();
                let legal = exps[i].state.legal_actions();
                let (lp, pr) = masked_log_softmax(&logits_i, &legal);
                entropy_sum += legal.iter().map(|&a| -pr[a] * lp[a]).sum::<f32>();
            }
            last_entropy = entropy_sum / n as f32;

            // Backward pass and optimizer step
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.network);
            self.network = self.optimizer.step(
                self.config.learning_rate,
                self.network.clone(),
                grads,
            );
        }

        (last_loss, last_entropy)
    }

    /// Compute state values for experiences using inference network (no gradient).
    fn compute_values_no_grad(&self, exps: &[&Experience]) -> Vec<f32> {
        let mut values = Vec::with_capacity(exps.len());
        let infer_net = self.network.valid();

        for exp in exps {
            let state_tensor =
                encode_state::<InferBackend>(&exp.state, &self.device).unsqueeze::<4>();
            let (_logits, value) = infer_net.forward(state_tensor);
            let v: Vec<f32> = value.into_data().to_vec().unwrap();
            values.push(v[0]);
        }

        values
    }

    /// Compute log probabilities of taken actions (no gradient).
    fn compute_log_probs_no_grad(&self, exps: &[&Experience]) -> Vec<f32> {
        let mut log_probs = Vec::with_capacity(exps.len());
        let infer_net = self.network.valid();

        for exp in exps {
            let state_tensor =
                encode_state::<InferBackend>(&exp.state, &self.device).unsqueeze::<4>();
            let (logits, _) = infer_net.forward(state_tensor);
            let logits_vec: Vec<f32> = logits.into_data().to_vec().unwrap();

            let legal = exp.state.legal_actions();
            let (lp, _) = masked_log_softmax(&logits_vec, &legal);
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
        };
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
        self.ppo_update(experiences)
    }

    fn current_metrics(&self) -> AgentMetrics {
        AgentMetrics {
            total_games: self.episode_count as u64,
            ..Default::default()
        }
    }
}

/// Apply legal action mask and compute softmax probabilities.
fn masked_softmax(logits: &[f32], legal: &[usize]) -> Vec<f32> {
    let mut masked = vec![f32::NEG_INFINITY; logits.len()];
    for &col in legal {
        masked[col] = logits[col];
    }

    // Numerically stable softmax
    let max_val = masked
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (i, &m) in masked.iter().enumerate() {
        let v = (m - max_val).exp();
        probs[i] = v;
        sum += v;
    }
    for p in &mut probs {
        *p /= sum;
    }

    probs
}

/// Compute masked log-softmax. Returns (log_probs, probs).
fn masked_log_softmax(logits: &[f32], legal: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let probs = masked_softmax(logits, legal);
    let log_probs: Vec<f32> = probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { -1e9 })
        .collect();
    (log_probs, probs)
}

/// Sample an action from a categorical distribution defined by probs.
fn sample_categorical(probs: &[f32], rng: &mut StdRng) -> usize {
    let r: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    // Fallback to last non-zero probability action
    probs
        .iter()
        .rposition(|&p| p > 0.0)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
