//! AI agent infrastructure: trait definitions, algorithm implementations (DQN,
//! Policy Gradient), neural network architectures, and board-to-tensor encoding.

mod agent;
pub mod algorithms;
pub mod networks;
mod random;
pub mod state_encoding;

pub use agent::{Agent, AgentMetrics, AgentState, Experience, UpdateMetrics};
pub use algorithms::{DqnAgent, PgConfig, PolicyGradientAgent};
pub use networks::{DqnNetwork, DqnNetworkConfig, PolicyValueNetwork, PolicyValueNetworkConfig};
pub use random::RandomAgent;
