//! AI agent infrastructure: trait definitions, algorithm implementations (DQN,
//! Policy Gradient), neural network architectures, and board-to-tensor encoding.

mod agent;
pub mod algorithms;
mod negamax;
pub mod networks;
mod random;
pub mod state_encoding;

pub use agent::{Agent, AgentMetrics, AgentState, EvalState, Experience, TrainableAgent, UpdateMetrics};
pub use algorithms::{AlphaZeroAgent, AlphaZeroConfig, DqnAgent, PgConfig, PolicyGradientAgent};
pub use negamax::{ConnectFourHeuristic, Heuristic, NegamaxAgent};
pub use networks::{DqnNetwork, DqnNetworkConfig, PolicyValueNetwork, PolicyValueNetworkConfig};
pub use random::RandomAgent;
