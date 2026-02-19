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

/// Per-episode timing tracker for profiling the training loop.
pub struct TimingMetrics {
    episode_micros: Vec<u32>, // per-episode simulation µs
    update_micros: Vec<u32>,  // per-batch-update µs
    window_start: std::time::Instant,
    window_count: usize,
}

impl TimingMetrics {
    pub fn new() -> Self {
        TimingMetrics {
            episode_micros: Vec::new(),
            update_micros: Vec::new(),
            window_start: std::time::Instant::now(),
            window_count: 0,
        }
    }

    pub fn record_episode_time(&mut self, d: std::time::Duration) {
        self.episode_micros.push(d.as_micros() as u32);
        self.window_count += 1;
    }

    pub fn record_update_time(&mut self, d: std::time::Duration) {
        self.update_micros.push(d.as_micros() as u32);
    }

    /// Mean of the last `last_n` episode times in milliseconds.
    pub fn avg_episode_ms(&self, last_n: usize) -> f32 {
        if self.episode_micros.is_empty() {
            return 0.0;
        }
        let start = self.episode_micros.len().saturating_sub(last_n);
        let slice = &self.episode_micros[start..];
        let mean = slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
        (mean / 1000.0) as f32
    }

    /// Mean of the last `last_n` update times in milliseconds.
    pub fn avg_update_ms(&self, last_n: usize) -> f32 {
        if self.update_micros.is_empty() {
            return 0.0;
        }
        let start = self.update_micros.len().saturating_sub(last_n);
        let slice = &self.update_micros[start..];
        let mean = slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
        (mean / 1000.0) as f32
    }

    /// Episodes per second since the last `reset_window` call.
    pub fn episodes_per_sec(&self) -> f32 {
        let secs = self.window_start.elapsed().as_secs_f32();
        if secs == 0.0 {
            return 0.0;
        }
        self.window_count as f32 / secs
    }

    /// Reset the throughput window (call after each log interval).
    pub fn reset_window(&mut self) {
        self.window_start = std::time::Instant::now();
        self.window_count = 0;
    }
}

impl Default for TimingMetrics {
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

    #[test]
    fn test_timing_avg_episode_ms_full_slice() {
        let mut t = TimingMetrics::new();
        t.record_episode_time(std::time::Duration::from_micros(2000)); // 2ms
        t.record_episode_time(std::time::Duration::from_micros(4000)); // 4ms
        // avg = 3000µs = 3.0ms
        assert!((t.avg_episode_ms(100) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_timing_avg_episode_ms_last_n() {
        let mut t = TimingMetrics::new();
        t.record_episode_time(std::time::Duration::from_micros(1000));
        t.record_episode_time(std::time::Duration::from_micros(9000));
        t.record_episode_time(std::time::Duration::from_micros(5000)); // 5ms
        // last 1 → only the 5000µs value = 5.0ms
        assert!((t.avg_episode_ms(1) - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_timing_episodes_per_sec_positive() {
        let mut t = TimingMetrics::new();
        // record a few episodes so window_count > 0
        for _ in 0..10 {
            t.record_episode_time(std::time::Duration::from_micros(1000));
        }
        // Even with nearly zero elapsed time the result should be > 0
        assert!(t.episodes_per_sec() >= 0.0);
        // After some real elapsed time it should definitely be positive
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t.episodes_per_sec() > 0.0);
    }
}
