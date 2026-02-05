use crate::game::{GameState, Player};

/// A single step of experience for RL training.
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: GameState,
    pub action: usize,
    pub reward: f32,
    pub next_state: GameState,
    pub done: bool,
    pub player: Player,
}

/// Metrics returned from a training update.
#[derive(Debug, Clone, Default)]
pub struct UpdateMetrics {
    pub loss: f32,
    pub gradient_norm: f32,
    pub policy_entropy: Option<f32>,
    pub value_estimate: Option<f32>,
}

/// Serializable agent state for checkpointing.
#[derive(Debug, Clone, Default)]
pub struct AgentState {
    pub data: Vec<u8>,
}

/// Current performance metrics for an agent.
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub total_games: u64,
}

/// Universal interface for all AI agents.
pub trait Agent {
    /// Select an action (column) given the current game state.
    /// When `training` is true, the agent may explore; otherwise it exploits.
    fn select_action(&mut self, state: &GameState, training: bool) -> usize;

    /// Return the agent's display name.
    fn name(&self) -> &str;

    /// Update the agent from a single experience. Returns training metrics.
    fn update(&mut self, _experience: Experience) -> UpdateMetrics {
        UpdateMetrics::default()
    }

    /// Update the agent from a batch of experiences. Returns training metrics.
    fn batch_update(&mut self, experiences: &[Experience]) -> UpdateMetrics {
        let mut metrics = UpdateMetrics::default();
        for exp in experiences {
            metrics = self.update(exp.clone());
        }
        metrics
    }

    /// Clone the agent into a boxed trait object.
    fn clone_agent(&self) -> Box<dyn Agent> {
        unimplemented!("clone_agent not implemented for this agent")
    }

    /// Save agent state for checkpointing.
    fn save_state(&self) -> AgentState {
        AgentState::default()
    }

    /// Load agent state from a checkpoint.
    fn load_state(&mut self, _state: AgentState) {}

    /// Get current performance metrics.
    fn current_metrics(&self) -> AgentMetrics {
        AgentMetrics::default()
    }
}
