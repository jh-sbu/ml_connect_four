use crate::game::Player;

/// Result of a single episode.
pub struct EpisodeResult {
    pub winner: Option<Player>,
    pub game_length: usize,
}

/// Training metrics tracker with rolling window computations.
pub struct TrainingMetrics {
    episode_results: Vec<EpisodeResult>,
    update_losses: Vec<f32>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        TrainingMetrics {
            episode_results: Vec::new(),
            update_losses: Vec::new(),
        }
    }

    pub fn record_episode(&mut self, result: EpisodeResult) {
        self.episode_results.push(result);
    }

    pub fn record_update(&mut self, loss: f32) {
        self.update_losses.push(loss);
    }

    /// Win rate for Red in the last N episodes.
    pub fn win_rate(&self, last_n: usize) -> f32 {
        let results = self.last_n_results(last_n);
        if results.is_empty() {
            return 0.0;
        }
        let wins = results
            .iter()
            .filter(|r| r.winner == Some(Player::Red))
            .count();
        wins as f32 / results.len() as f32
    }

    /// Draw rate in the last N episodes.
    pub fn draw_rate(&self, last_n: usize) -> f32 {
        let results = self.last_n_results(last_n);
        if results.is_empty() {
            return 0.0;
        }
        let draws = results.iter().filter(|r| r.winner.is_none()).count();
        draws as f32 / results.len() as f32
    }

    /// Average loss over the last N updates.
    pub fn average_loss(&self, last_n: usize) -> f32 {
        if self.update_losses.is_empty() {
            return 0.0;
        }
        let start = self.update_losses.len().saturating_sub(last_n);
        let slice = &self.update_losses[start..];
        slice.iter().sum::<f32>() / slice.len() as f32
    }

    /// Average game length over the last N episodes.
    pub fn average_game_length(&self, last_n: usize) -> f32 {
        let results = self.last_n_results(last_n);
        if results.is_empty() {
            return 0.0;
        }
        let total: usize = results.iter().map(|r| r.game_length).sum();
        total as f32 / results.len() as f32
    }

    pub fn total_episodes(&self) -> usize {
        self.episode_results.len()
    }

    fn last_n_results(&self, last_n: usize) -> &[EpisodeResult] {
        let start = self.episode_results.len().saturating_sub(last_n);
        &self.episode_results[start..]
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_win_rate() {
        let mut m = TrainingMetrics::new();
        for _ in 0..7 {
            m.record_episode(EpisodeResult {
                winner: Some(Player::Red),
                game_length: 10,
            });
        }
        for _ in 0..3 {
            m.record_episode(EpisodeResult {
                winner: Some(Player::Yellow),
                game_length: 10,
            });
        }
        assert!((m.win_rate(10) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_draw_rate() {
        let mut m = TrainingMetrics::new();
        m.record_episode(EpisodeResult {
            winner: None,
            game_length: 42,
        });
        m.record_episode(EpisodeResult {
            winner: Some(Player::Red),
            game_length: 10,
        });
        assert!((m.draw_rate(10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_average_loss() {
        let mut m = TrainingMetrics::new();
        m.record_update(1.0);
        m.record_update(3.0);
        assert!((m.average_loss(10) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_game_length() {
        let mut m = TrainingMetrics::new();
        m.record_episode(EpisodeResult {
            winner: None,
            game_length: 20,
        });
        m.record_episode(EpisodeResult {
            winner: None,
            game_length: 30,
        });
        assert!((m.average_game_length(10) - 25.0).abs() < 1e-6);
    }
}
