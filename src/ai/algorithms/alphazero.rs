use std::cmp::Ordering;
use std::collections::VecDeque;
use std::error::Error;
use std::path::Path;

use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::DefaultRecorder;
use burn::tensor::TensorData;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma};

use crate::ai::agent::{Agent, AgentMetrics, EvalState, Experience, TrainableAgent, UpdateMetrics};
use crate::ai::networks::{PolicyValueNetwork, PolicyValueNetworkConfig};
use crate::ai::state_encoding::{encode_state, encode_states_batch_into};
use crate::checkpoint::{
    AlphaZeroHyperparameters, AlphaZeroTrainingState, CheckpointHyperparameters, CheckpointMetadata,
};
use crate::game::{GameOutcome, GameState, Player};

type InferBackend = Wgpu<f32, i32>;
type TrainBackend = Autodiff<InferBackend>;

// ─── Config ──────────────────────────────────────────────────────────────────

/// AlphaZero hyperparameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AlphaZeroConfig {
    pub learning_rate: f64,
    /// Number of MCTS simulations per move during training.
    pub num_simulations: usize,
    /// Number of MCTS simulations per move in the game UI (0 → use 1).
    pub eval_simulations: usize,
    /// Exploration constant in the PUCT formula.
    pub c_puct: f32,
    /// Dirichlet noise concentration parameter.
    pub dirichlet_alpha: f32,
    /// Weight of Dirichlet noise mixed into root priors.
    pub dirichlet_epsilon: f32,
    /// First N moves use `temperature`; after that `temperature_final` is used.
    pub temperature_moves: usize,
    pub temperature: f32,
    pub temperature_final: f32,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub min_replay_size: usize,
    /// Weight applied to the value loss term.
    pub value_weight: f32,
    pub max_grad_norm: f32,
}

impl Default for AlphaZeroConfig {
    fn default() -> Self {
        AlphaZeroConfig {
            learning_rate: 1e-3,
            num_simulations: 200,
            eval_simulations: 50,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature_moves: 10,
            temperature: 1.0,
            temperature_final: 0.1,
            batch_size: 256,
            replay_capacity: 20_000,
            min_replay_size: 512,
            value_weight: 1.0,
            max_grad_norm: 1.0,
        }
    }
}

// ─── Experience ───────────────────────────────────────────────────────────────

/// One training sample for the AlphaZero replay buffer.
#[derive(Clone)]
struct AzExperience {
    state: GameState,
    /// MCTS visit-count distribution (normalized, over 7 columns).
    mcts_policy: [f32; 7],
    /// Game outcome from the current player's perspective: +1 win, -1 loss, 0 draw.
    outcome: f32,
}

// ─── MCTS Tree (arena-based) ──────────────────────────────────────────────────

struct MctsNode {
    state: GameState,
    visit_count: u32,
    /// Cumulative value from THIS node's current_player's perspective.
    value_sum: f32,
    prior: f32,
    /// Child node indices in MctsTree::nodes; -1 means not yet created.
    children: [i32; 7],
    is_expanded: bool,
}

struct MctsTree {
    nodes: Vec<MctsNode>,
}

impl MctsTree {
    fn new() -> Self {
        MctsTree { nodes: Vec::with_capacity(512) }
    }

    /// Reset the tree and install a fresh root node.
    fn init(&mut self, root_state: GameState) {
        self.nodes.clear();
        self.nodes.push(MctsNode {
            state: root_state,
            visit_count: 0,
            value_sum: 0.0,
            prior: 1.0,
            children: [-1i32; 7],
            is_expanded: false,
        });
    }

    /// Expand node `node_idx` using the given prior policy.
    ///
    /// Child states are pre-computed before any push to avoid borrow conflicts
    /// when the `nodes` Vec reallocates.
    fn expand(&mut self, node_idx: usize, policy: &[f32; 7]) {
        let parent_state = self.nodes[node_idx].state.clone();
        let legal = parent_state.legal_actions();

        // Pre-compute all child (action, state, prior) tuples.
        let child_data: Vec<(usize, GameState)> = legal
            .iter()
            .filter_map(|&action| {
                parent_state.apply_move(action).ok().map(|s| (action, s))
            })
            .collect();

        let mut child_indices = [-1i32; 7];
        for (action, child_state) in child_data {
            let idx = self.nodes.len() as i32;
            child_indices[action] = idx;
            self.nodes.push(MctsNode {
                state: child_state,
                visit_count: 0,
                value_sum: 0.0,
                prior: policy[action],
                children: [-1i32; 7],
                is_expanded: false,
            });
        }

        self.nodes[node_idx].children = child_indices;
        self.nodes[node_idx].is_expanded = true;
    }

    /// Return the action index of the child with the highest PUCT score.
    ///
    /// Q is negated because the child's value is from the opponent's perspective.
    /// Returns 7 (sentinel) if no valid child exists.
    fn select_best_child(&self, node_idx: usize, c_puct: f32) -> usize {
        let node = &self.nodes[node_idx];
        let parent_visits_sqrt = (node.visit_count as f32).sqrt();

        let mut best_action = 7usize; // sentinel for "none"
        let mut best_score = f32::NEG_INFINITY;

        for action in 0..7 {
            let ci = node.children[action];
            if ci < 0 {
                continue;
            }
            let child = &self.nodes[ci as usize];

            let q = if child.visit_count == 0 {
                0.0
            } else {
                -child.value_sum / child.visit_count as f32
            };
            let u = c_puct * child.prior * parent_visits_sqrt
                / (1.0 + child.visit_count as f32);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }

    /// Propagate `leaf_value` back through `path`, alternating sign at each step.
    fn backup(&mut self, path: &[usize], leaf_value: f32) {
        let mut v = leaf_value;
        for &idx in path.iter().rev() {
            self.nodes[idx].visit_count += 1;
            self.nodes[idx].value_sum += v;
            v = -v;
        }
    }
}

// ─── Agent ────────────────────────────────────────────────────────────────────

/// AlphaZero agent: MCTS guided by a policy-value network, trained from self-play.
pub struct AlphaZeroAgent {
    network: PolicyValueNetwork<TrainBackend>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<
        burn::optim::Adam,
        PolicyValueNetwork<TrainBackend>,
        TrainBackend,
    >,
    config: AlphaZeroConfig,
    device: <TrainBackend as Backend>::Device,
    tree: MctsTree,
    replay_buffer: VecDeque<AzExperience>,
    /// States encountered in the current episode (buffered until batch_update).
    episode_states: Vec<GameState>,
    /// MCTS policies produced for each state in the current episode.
    episode_policies: Vec<[f32; 7]>,
    episode_count: usize,
    step_count: usize,
    last_loss: f32,
    rng: StdRng,
    /// Pre-allocated flat buffer for batch state encoding.
    flat_buf: Vec<f32>,
}

impl AlphaZeroAgent {
    pub fn new(config: AlphaZeroConfig) -> Self {
        let device: <TrainBackend as Backend>::Device = Default::default();
        let network: PolicyValueNetwork<TrainBackend> = PolicyValueNetworkConfig {}.init(&device);
        let optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)))
            .init();

        AlphaZeroAgent {
            network,
            optimizer,
            config,
            device,
            tree: MctsTree::new(),
            replay_buffer: VecDeque::new(),
            episode_states: Vec::new(),
            episode_policies: Vec::new(),
            episode_count: 0,
            step_count: 0,
            last_loss: 0.0,
            rng: StdRng::from_os_rng(),
            flat_buf: Vec::new(),
        }
    }

    /// Run a single MCTS simulation from `root_idx`, expanding and backing up.
    fn run_simulation(&mut self, root_idx: usize) {
        let mut path = Vec::new();
        let mut current = root_idx;

        loop {
            // Terminal state check.
            if self.tree.nodes[current].state.is_terminal() {
                let value = terminal_value(&self.tree.nodes[current].state);
                path.push(current);
                self.tree.backup(&path, value);
                return;
            }

            if !self.tree.nodes[current].is_expanded {
                // Clone state before eval_state to release borrow on nodes.
                let state_clone = self.tree.nodes[current].state.clone();
                let (policy, value) = eval_state(&self.network, &self.device, &state_clone);
                self.tree.expand(current, &policy);
                path.push(current);
                self.tree.backup(&path, value);
                return;
            }

            path.push(current);
            let action = self.tree.select_best_child(current, self.config.c_puct);
            if action >= 7 {
                // No valid children (should not happen for non-terminal expanded node).
                self.tree.backup(&path, 0.0);
                return;
            }
            let child_raw = self.tree.nodes[current].children[action];
            if child_raw < 0 {
                self.tree.backup(&path, 0.0);
                return;
            }
            current = child_raw as usize;
        }
    }

    /// Run MCTS and return (policy, action).
    fn mcts_search(
        &mut self,
        state: &GameState,
        training: bool,
        num_sims: usize,
    ) -> ([f32; 7], usize) {
        self.tree.init(state.clone());

        // First simulation expands the root.
        self.run_simulation(0);

        // Add Dirichlet noise to root priors for training exploration.
        if training {
            let legal = state.legal_actions();
            if !legal.is_empty() {
                self.add_dirichlet_noise_to_root(0, &legal);
            }
        }

        for _ in 1..num_sims {
            self.run_simulation(0);
        }

        // Compute visit-count policy over legal actions.
        let legal = state.legal_actions();
        let mut visit_counts = [0u32; 7];
        for action in 0..7 {
            let ci = self.tree.nodes[0].children[action];
            if ci >= 0 {
                visit_counts[action] = self.tree.nodes[ci as usize].visit_count;
            }
        }

        let total: u32 = legal.iter().map(|&a| visit_counts[a]).sum();
        let mut policy = [0.0f32; 7];
        if total > 0 {
            for &a in &legal {
                policy[a] = visit_counts[a] as f32 / total as f32;
            }
        } else {
            let p = 1.0 / legal.len() as f32;
            for &a in &legal {
                policy[a] = p;
            }
        }

        // Select action: temperature sampling during training, greedy otherwise.
        let action = if training {
            let move_num = self.episode_states.len();
            let temp = if move_num <= self.config.temperature_moves {
                self.config.temperature
            } else {
                self.config.temperature_final
            };
            sample_with_temperature(&visit_counts, &legal, temp, &mut self.rng)
        } else {
            *legal.iter().max_by_key(|&&a| visit_counts[a]).unwrap()
        };

        (policy, action)
    }

    /// Mix Dirichlet noise into the root's children priors.
    ///
    /// Dirichlet(alpha, ...) is sampled via Gamma(alpha, 1) variates + normalize.
    fn add_dirichlet_noise_to_root(&mut self, root_idx: usize, legal: &[usize]) {
        let n = legal.len();
        let alpha = self.config.dirichlet_alpha as f64;
        let Ok(gamma_dist) = Gamma::new(alpha, 1.0) else { return };

        let mut samples: Vec<f32> = (0..n)
            .map(|_| gamma_dist.sample(&mut self.rng) as f32)
            .collect();
        let sum: f32 = samples.iter().sum();
        if sum > 0.0 {
            for s in &mut samples {
                *s /= sum;
            }
        } else {
            let p = 1.0 / n as f32;
            samples.iter_mut().for_each(|s| *s = p);
        }

        let eps = self.config.dirichlet_epsilon;
        for (i, &action) in legal.iter().enumerate() {
            let ci = self.tree.nodes[root_idx].children[action];
            if ci >= 0 {
                let child = &mut self.tree.nodes[ci as usize];
                child.prior = (1.0 - eps) * child.prior + eps * samples[i];
            }
        }
    }

    /// Sample a random batch from the replay buffer (with replacement).
    fn sample_batch(&mut self) -> Vec<AzExperience> {
        let buf_len = self.replay_buffer.len();
        let n = self.config.batch_size.min(buf_len);
        (0..n)
            .map(|_| {
                let i = self.rng.random_range(0..buf_len);
                self.replay_buffer[i].clone()
            })
            .collect()
    }

    /// Perform one gradient update on a batch of AzExperiences.
    fn update_network(&mut self, batch: &[AzExperience]) -> f32 {
        let n = batch.len();

        // Encode states: [n, 3, 6, 7].
        let states: Vec<GameState> = batch.iter().map(|e| e.state.clone()).collect();
        let state_tensor =
            encode_states_batch_into::<TrainBackend>(&states, &mut self.flat_buf, &self.device);

        // Policy targets: [n, 7].
        let policy_flat: Vec<f32> =
            batch.iter().flat_map(|e| e.mcts_policy.iter().copied()).collect();
        let policy_targets = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(policy_flat.as_slice()),
            &self.device,
        )
        .reshape([n as i32, 7]);

        // Value targets: [n, 1].
        let value_flat: Vec<f32> = batch.iter().map(|e| e.outcome).collect();
        let value_targets = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(value_flat.as_slice()),
            &self.device,
        )
        .reshape([n as i32, 1]);

        // Legal mask: -1e9 for illegal columns, 0 for legal.
        let mut legal_mask_data = vec![-1e9f32; n * 7];
        for (i, exp) in batch.iter().enumerate() {
            for &col in &exp.state.legal_actions() {
                legal_mask_data[i * 7 + col] = 0.0;
            }
        }
        let legal_mask_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::from(legal_mask_data.as_slice()),
            &self.device,
        )
        .reshape([n as i32, 7]);

        // Forward pass.
        let (logits, values) = self.network.forward(state_tensor);

        // Policy loss: cross-entropy(-mcts_policy, log_softmax(masked_logits)).
        let masked_logits = logits + legal_mask_tensor;
        let log_probs = burn::tensor::activation::log_softmax(masked_logits, 1);
        let policy_loss = -(policy_targets * log_probs).sum_dim(1).mean();

        // Value loss: MSE(tanh(predicted), target).
        let value_pred = values.tanh();
        let value_diff = value_pred - value_targets;
        let value_loss = (value_diff.clone() * value_diff).mean();

        let total_loss = policy_loss + value_loss * self.config.value_weight;

        let loss_scalar = total_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 loss tensor")[0];

        if loss_scalar.is_finite() {
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.network);
            self.network =
                self.optimizer.step(self.config.learning_rate, self.network.clone(), grads);
        }

        loss_scalar
    }

    fn training_state(&self) -> AlphaZeroTrainingState {
        AlphaZeroTrainingState {
            episode_count: self.episode_count,
            step_count: self.step_count,
            learning_rate: self.config.learning_rate,
            num_simulations: self.config.num_simulations,
            eval_simulations: self.config.eval_simulations,
            c_puct: self.config.c_puct,
            dirichlet_alpha: self.config.dirichlet_alpha,
            dirichlet_epsilon: self.config.dirichlet_epsilon,
            temperature_moves: self.config.temperature_moves,
            temperature: self.config.temperature,
            temperature_final: self.config.temperature_final,
            batch_size: self.config.batch_size,
            replay_capacity: self.config.replay_capacity,
            min_replay_size: self.config.min_replay_size,
            value_weight: self.config.value_weight,
            max_grad_norm: self.config.max_grad_norm,
        }
    }

    fn restore_training_state_from(&mut self, state: &AlphaZeroTrainingState) {
        self.episode_count = state.episode_count;
        self.step_count = state.step_count;
        self.config = AlphaZeroConfig {
            learning_rate: state.learning_rate,
            num_simulations: state.num_simulations,
            eval_simulations: state.eval_simulations,
            c_puct: state.c_puct,
            dirichlet_alpha: state.dirichlet_alpha,
            dirichlet_epsilon: state.dirichlet_epsilon,
            temperature_moves: state.temperature_moves,
            temperature: state.temperature,
            temperature_final: state.temperature_final,
            batch_size: state.batch_size,
            replay_capacity: state.replay_capacity,
            min_replay_size: state.min_replay_size,
            value_weight: state.value_weight,
            max_grad_norm: state.max_grad_norm,
        };
        self.optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(self.config.max_grad_norm)))
            .init();
    }
}

// ─── Agent impl ───────────────────────────────────────────────────────────────

impl Agent for AlphaZeroAgent {
    fn select_action(&mut self, state: &GameState, training: bool) -> usize {
        let num_sims = if training {
            self.config.num_simulations
        } else {
            self.config.eval_simulations.max(1)
        };

        let (policy, action) = self.mcts_search(state, training, num_sims);

        if training {
            self.episode_states.push(state.clone());
            self.episode_policies.push(policy);
        }

        action
    }

    fn name(&self) -> &str {
        "AZ"
    }

    fn batch_update(&mut self, experiences: &[Experience]) -> UpdateMetrics {
        self.episode_count += 1;

        let winner = extract_game_winner(experiences);

        // Convert buffered (state, policy) pairs → AzExperiences.
        let states = std::mem::take(&mut self.episode_states);
        let policies = std::mem::take(&mut self.episode_policies);

        for (state, policy) in states.into_iter().zip(policies.into_iter()) {
            let player = state.current_player();
            let outcome = match winner {
                Some(p) if p == player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            if self.replay_buffer.len() >= self.config.replay_capacity {
                self.replay_buffer.pop_front();
            }
            self.replay_buffer.push_back(AzExperience { state, mcts_policy: policy, outcome });
        }

        if self.replay_buffer.len() < self.config.min_replay_size {
            return UpdateMetrics::default();
        }

        let batch = self.sample_batch();
        let loss = self.update_network(&batch);
        self.step_count += 1;
        self.last_loss = loss;

        UpdateMetrics { loss, ..Default::default() }
    }

    fn current_metrics(&self) -> AgentMetrics {
        AgentMetrics { total_games: self.episode_count as u64, ..Default::default() }
    }
}

// ─── TrainableAgent impl ──────────────────────────────────────────────────────

impl TrainableAgent for AlphaZeroAgent {
    fn algorithm_name(&self) -> &str {
        "AZ"
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
        self.config.num_simulations as f32
    }

    fn algorithm_metric_label(&self) -> &str {
        "sims"
    }

    fn last_policy_entropy(&self) -> Option<f32> {
        None
    }

    fn save_weights_to_dir(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultRecorder::default();
        self.network
            .clone()
            .valid()
            .save_file(dir.join("policy_value_network"), &recorder)?;
        Ok(())
    }

    fn training_state_json(&self) -> String {
        serde_json::to_string_pretty(&self.training_state())
            .expect("AZ training state serializes")
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
            algorithm: "AZ".to_string(),
            metrics: metrics.clone(),
            hyperparameters: CheckpointHyperparameters {
                learning_rate: ts.learning_rate,
                gamma: 0.0,
                epsilon: 0.0,
                batch_size: ts.batch_size,
                target_update_interval: 0,
                replay_capacity: ts.replay_capacity,
                min_replay_size: ts.min_replay_size,
                epsilon_start: 0.0,
                epsilon_end: 0.0,
                epsilon_decay_episodes: 0,
            },
            pg_hyperparameters: None,
            az_hyperparameters: Some(AlphaZeroHyperparameters {
                learning_rate: ts.learning_rate,
                num_simulations: ts.num_simulations,
                eval_simulations: ts.eval_simulations,
                c_puct: ts.c_puct,
                dirichlet_alpha: ts.dirichlet_alpha,
                dirichlet_epsilon: ts.dirichlet_epsilon,
                temperature_moves: ts.temperature_moves,
                temperature: ts.temperature,
                temperature_final: ts.temperature_final,
                batch_size: ts.batch_size,
                replay_capacity: ts.replay_capacity,
                min_replay_size: ts.min_replay_size,
                value_weight: ts.value_weight,
                max_grad_norm: ts.max_grad_norm,
            }),
        }
    }

    fn load_weights_from_dir(&mut self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultRecorder::default();
        let net: PolicyValueNetwork<TrainBackend> = PolicyValueNetworkConfig {}
            .init(&self.device)
            .load_file(dir.join("policy_value_network"), &recorder, &self.device)?;
        self.network = net;
        Ok(())
    }

    fn restore_training_state_json(&mut self, json: &str) -> Result<(), Box<dyn Error>> {
        let state: AlphaZeroTrainingState = serde_json::from_str(json)?;
        self.restore_training_state_from(&state);
        Ok(())
    }

    fn set_episode_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn clone_for_eval(&self) -> Box<dyn Agent + Send> {
        Box::new(AzEvalAgent {
            network: self.network.clone().valid(),
            device: self.device.clone(),
        })
    }
}

// ─── Inference-only eval agent (for parallel evaluation) ─────────────────────

/// Greedy policy-only agent used for fast parallel evaluation during training.
struct AzEvalAgent {
    network: PolicyValueNetwork<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl Agent for AzEvalAgent {
    fn select_action(&mut self, state: &GameState, _training: bool) -> usize {
        let legal = state.legal_actions();
        assert!(!legal.is_empty(), "No legal actions");
        let t = encode_state::<InferBackend>(state, &self.device).unsqueeze::<4>();
        let (logits, _) = self.network.forward(t);
        let v: Vec<f32> = logits.into_data().to_vec().expect("f32 logits");
        *legal
            .iter()
            .max_by(|&&a, &&b| v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    fn name(&self) -> &str {
        "AZ"
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Evaluate a state with the network (inference only, no grad).
///
/// Takes `&PolicyValueNetwork<TrainBackend>` so it can be called while
/// `self.tree` is separately mutably borrowed in `run_simulation`.
fn eval_state(
    network: &PolicyValueNetwork<TrainBackend>,
    device: &<TrainBackend as Backend>::Device,
    state: &GameState,
) -> ([f32; 7], f32) {
    let t = encode_state::<InferBackend>(state, device).unsqueeze::<4>();
    let (logits, value_t) = network.valid().forward(t);
    let logits_vec: Vec<f32> = logits.into_data().to_vec().expect("f32 logits");
    let raw_value = value_t.into_data().to_vec::<f32>().expect("f32 value")[0];
    let policy = masked_softmax_7(&logits_vec, &state.legal_actions());
    (policy, raw_value.tanh())
}

/// Terminal state value from the perspective of the node's current player.
///
/// In Connect Four the current player at a terminal Winner state is always
/// the loser (the winning move was the opponent's last move).
fn terminal_value(state: &GameState) -> f32 {
    match state.outcome() {
        Some(GameOutcome::Winner(_)) => -1.0, // current player lost
        Some(GameOutcome::Draw) => 0.0,
        None => 0.0,
    }
}

/// Extract the game winner from an experience trajectory.
fn extract_game_winner(experiences: &[Experience]) -> Option<Player> {
    for exp in experiences.iter().rev() {
        if exp.done {
            return match exp.reward.partial_cmp(&0.0) {
                Some(Ordering::Greater) => Some(exp.player),
                Some(Ordering::Less) => Some(exp.player.other()),
                _ => None,
            };
        }
    }
    None
}

/// Apply legal-action mask and compute softmax over 7 columns.
fn masked_softmax_7(logits: &[f32], legal: &[usize]) -> [f32; 7] {
    let mut masked = [f32::NEG_INFINITY; 7];
    for &col in legal {
        masked[col] = logits[col];
    }

    let max_val = masked.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return uniform_7(legal);
    }

    let mut probs = [0.0f32; 7];
    let mut sum = 0.0f32;
    for i in 0..7 {
        if masked[i].is_finite() {
            let v = (masked[i] - max_val).exp();
            probs[i] = v;
            sum += v;
        }
    }

    if !sum.is_finite() || sum == 0.0 {
        return uniform_7(legal);
    }

    for p in &mut probs {
        *p /= sum;
    }
    probs
}

fn uniform_7(legal: &[usize]) -> [f32; 7] {
    let mut probs = [0.0f32; 7];
    let p = 1.0 / legal.len() as f32;
    for &col in legal {
        probs[col] = p;
    }
    probs
}

/// Sample an action from visit counts using the given temperature.
///
/// With temp → 0 this becomes greedy (argmax); with temp = 1 it's proportional.
fn sample_with_temperature(
    visit_counts: &[u32; 7],
    legal: &[usize],
    temp: f32,
    rng: &mut StdRng,
) -> usize {
    if temp < 1e-6 {
        return *legal.iter().max_by_key(|&&a| visit_counts[a]).unwrap();
    }

    let inv_temp = 1.0 / temp;
    let mut probs = [0.0f32; 7];
    let mut sum = 0.0f32;
    for &a in legal {
        let v = (visit_counts[a] as f32).powf(inv_temp);
        probs[a] = v;
        sum += v;
    }

    if sum <= 0.0 {
        return legal[rng.random_range(0..legal.len())];
    }

    for p in &mut probs {
        *p /= sum;
    }

    let r: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0f32;
    for &a in legal {
        cumulative += probs[a];
        if r < cumulative {
            return a;
        }
    }
    *legal.last().unwrap()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> AlphaZeroConfig {
        AlphaZeroConfig {
            num_simulations: 2,
            eval_simulations: 2,
            min_replay_size: 2,
            batch_size: 2,
            replay_capacity: 100,
            ..Default::default()
        }
    }

    #[test]
    fn test_az_agent_selects_legal_action() {
        let mut agent = AlphaZeroAgent::new(small_config());
        let state = GameState::initial();
        let legal = state.legal_actions();
        for _ in 0..5 {
            let action = agent.select_action(&state, false);
            assert!(legal.contains(&action), "action {} not legal", action);
        }
    }

    #[test]
    fn test_az_agent_training_mode_explores() {
        let mut agent = AlphaZeroAgent::new(small_config());
        let state = GameState::initial();

        let mut actions = std::collections::HashSet::new();
        for _ in 0..30 {
            let action = agent.select_action(&state, true);
            actions.insert(action);
            // Reset episode state between calls so buffer doesn't grow indefinitely.
            agent.episode_states.clear();
            agent.episode_policies.clear();
        }
        assert!(actions.len() > 1, "expected exploration, got {:?}", actions);
    }

    #[test]
    fn test_mcts_tree_expand() {
        let mut tree = MctsTree::new();
        let state = GameState::initial();
        tree.init(state.clone());

        let policy = [1.0 / 7.0f32; 7];
        tree.expand(0, &policy);

        assert!(tree.nodes[0].is_expanded);
        // All 7 columns legal from initial state → all children present.
        for action in 0..7 {
            let ci = tree.nodes[0].children[action];
            assert!(ci >= 0, "child {} missing", action);
            let child = &tree.nodes[ci as usize];
            assert!((child.prior - 1.0 / 7.0).abs() < 1e-6, "prior mismatch");
        }
    }

    #[test]
    fn test_mcts_backup_alternates_sign() {
        let mut tree = MctsTree::new();
        let state = GameState::initial();
        tree.init(state.clone());

        // Manually add two more nodes to form a 3-node path.
        let s1 = state.apply_move(0).unwrap();
        tree.nodes.push(MctsNode {
            state: s1.clone(),
            visit_count: 0,
            value_sum: 0.0,
            prior: 1.0,
            children: [-1i32; 7],
            is_expanded: false,
        });
        let s2 = s1.apply_move(1).unwrap();
        tree.nodes.push(MctsNode {
            state: s2,
            visit_count: 0,
            value_sum: 0.0,
            prior: 1.0,
            children: [-1i32; 7],
            is_expanded: false,
        });

        // Path: root(0) → node1(1) → node2(2), backup with leaf_value = 1.0
        tree.backup(&[0, 1, 2], 1.0);

        // node2 (leaf): value_sum = +1.0
        assert!((tree.nodes[2].value_sum - 1.0).abs() < 1e-6);
        assert_eq!(tree.nodes[2].visit_count, 1);
        // node1: value_sum = -1.0 (sign flipped)
        assert!((tree.nodes[1].value_sum - (-1.0)).abs() < 1e-6);
        // node0 (root): value_sum = +1.0 (sign flipped again)
        assert!((tree.nodes[0].value_sum - 1.0).abs() < 1e-6);
    }

    fn make_exp(reward: f32, done: bool, player: Player) -> Experience {
        let state = GameState::initial();
        let next = state.apply_move(0).unwrap();
        Experience { state, action: 0, reward, next_state: next, done, player }
    }

    #[test]
    fn test_extract_game_winner_red_wins() {
        let exps = vec![make_exp(1.0, true, Player::Red)];
        assert_eq!(extract_game_winner(&exps), Some(Player::Red));
    }

    #[test]
    fn test_extract_game_winner_yellow_wins() {
        let exps = vec![make_exp(1.0, true, Player::Yellow)];
        assert_eq!(extract_game_winner(&exps), Some(Player::Yellow));
    }

    #[test]
    fn test_extract_game_winner_draw() {
        let exps = vec![make_exp(0.0, true, Player::Red)];
        assert_eq!(extract_game_winner(&exps), None);
    }

    #[test]
    fn test_az_batch_update_fills_replay_below_min() {
        let mut agent = AlphaZeroAgent::new(small_config());
        let state = GameState::initial();
        let next = state.apply_move(0).unwrap();

        // Manually seed episode buffers (mimics select_action in training mode).
        agent.episode_states.push(state.clone());
        agent.episode_policies.push([1.0 / 7.0; 7]);

        let exps = vec![Experience {
            state,
            action: 0,
            reward: 1.0,
            next_state: next,
            done: true,
            player: Player::Red,
        }];

        // With min_replay_size=2, one episode worth of data → no update.
        let metrics = agent.batch_update(&exps);
        assert_eq!(agent.step_count(), 0, "no update expected below min_replay_size");
        assert_eq!(metrics.loss, 0.0);
        assert_eq!(agent.episode_count(), 1);
    }

    #[test]
    fn test_az_eval_agent_selects_legal_action() {
        let agent = AlphaZeroAgent::new(small_config());
        let mut eval_agent = agent.clone_for_eval();
        let state = GameState::initial();
        let legal = state.legal_actions();
        for _ in 0..5 {
            let action = eval_agent.select_action(&state, false);
            assert!(legal.contains(&action), "eval action {} not legal", action);
        }
    }
}
