use crate::game::GameState;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::agent::Agent;

/// An agent that selects uniformly at random from legal actions.
pub struct RandomAgent {
    rng: StdRng,
}

impl RandomAgent {
    pub fn new() -> Self {
        RandomAgent {
            rng: StdRng::from_os_rng(),
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for RandomAgent {
    fn select_action(&mut self, state: &GameState, _training: bool) -> usize {
        let actions = state.legal_actions();
        assert!(!actions.is_empty(), "No legal actions available");
        let idx = self.rng.random_range(0..actions.len());
        actions[idx]
    }

    fn name(&self) -> &str {
        "Random"
    }

    fn clone_agent(&self) -> Box<dyn Agent> {
        Box::new(RandomAgent::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;

    #[test]
    fn test_random_agent_selects_legal_action() {
        let mut agent = RandomAgent::new();
        let state = GameState::initial();
        let legal = state.legal_actions();

        for _ in 0..100 {
            let action = agent.select_action(&state, false);
            assert!(legal.contains(&action), "Action {} is not legal", action);
        }
    }

    #[test]
    fn test_random_agent_plays_full_game() {
        let mut agent1 = RandomAgent::new();
        let mut agent2 = RandomAgent::new();
        let mut state = GameState::initial();

        let mut turn = 0;
        while !state.is_terminal() {
            let action = if turn % 2 == 0 {
                agent1.select_action(&state, false)
            } else {
                agent2.select_action(&state, false)
            };
            state = state.apply_move(action).unwrap();
            turn += 1;
        }

        assert!(state.is_terminal());
        assert!(state.outcome().is_some());
    }

    #[test]
    fn test_random_agent_name() {
        let agent = RandomAgent::new();
        assert_eq!(agent.name(), "Random");
    }
}
