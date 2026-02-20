use std::collections::VecDeque;

use crate::game::Player;

/// Result of a single episode.
pub struct EpisodeResult {
    pub winner: Option<Player>,
    pub game_length: usize,
}

/// Training metrics tracker with rolling window computations.
pub struct TrainingMetrics {
    episode_results: VecDeque<EpisodeResult>,
    update_losses: VecDeque<f32>,
    capacity: usize,
    total_episodes: usize, // lifetime count, never capped
}

impl TrainingMetrics {
    pub fn with_capacity(capacity: usize) -> Self {
        TrainingMetrics {
            episode_results: VecDeque::with_capacity(capacity),
            update_losses: VecDeque::with_capacity(capacity),
            capacity,
            total_episodes: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    pub fn record_episode(&mut self, result: EpisodeResult) {
        self.total_episodes += 1;
        self.episode_results.push_back(result);
        if self.episode_results.len() > self.capacity {
            self.episode_results.pop_front();
        }
    }

    pub fn record_update(&mut self, loss: f32) {
        self.update_losses.push_back(loss);
        if self.update_losses.len() > self.capacity {
            self.update_losses.pop_front();
        }
    }

    /// Win rate for Red in the last N episodes.
    pub fn win_rate(&self, last_n: usize) -> f32 {
        let n = self.episode_results.len().min(last_n);
        if n == 0 {
            return 0.0;
        }
        let wins = self
            .episode_results
            .iter()
            .rev()
            .take(n)
            .filter(|r| r.winner == Some(Player::Red))
            .count();
        wins as f32 / n as f32
    }

    /// Draw rate in the last N episodes.
    pub fn draw_rate(&self, last_n: usize) -> f32 {
        let n = self.episode_results.len().min(last_n);
        if n == 0 {
            return 0.0;
        }
        let draws = self
            .episode_results
            .iter()
            .rev()
            .take(n)
            .filter(|r| r.winner.is_none())
            .count();
        draws as f32 / n as f32
    }

    /// Average loss over the last N updates.
    pub fn average_loss(&self, last_n: usize) -> f32 {
        let n = self.update_losses.len().min(last_n);
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = self.update_losses.iter().rev().take(n).sum();
        sum / n as f32
    }

    /// Average game length over the last N episodes.
    pub fn average_game_length(&self, last_n: usize) -> f32 {
        let n = self.episode_results.len().min(last_n);
        if n == 0 {
            return 0.0;
        }
        let total: usize = self
            .episode_results
            .iter()
            .rev()
            .take(n)
            .map(|r| r.game_length)
            .sum();
        total as f32 / n as f32
    }

    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-episode timing tracker for profiling the training loop.
pub struct TimingMetrics {
    episode_micros: VecDeque<u32>, // per-episode simulation µs
    update_micros: VecDeque<u32>,  // per-batch-update µs
    capacity: usize,
    window_start: std::time::Instant,
    window_count: usize,
    window_overhead_micros: u128, // eval/checkpoint time excluded from throughput
}

impl TimingMetrics {
    pub fn with_capacity(capacity: usize) -> Self {
        TimingMetrics {
            episode_micros: VecDeque::with_capacity(capacity),
            update_micros: VecDeque::with_capacity(capacity),
            capacity,
            window_start: std::time::Instant::now(),
            window_count: 0,
            window_overhead_micros: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    pub fn record_episode_time(&mut self, d: std::time::Duration) {
        self.episode_micros.push_back(d.as_micros() as u32);
        if self.episode_micros.len() > self.capacity {
            self.episode_micros.pop_front();
        }
        self.window_count += 1;
    }

    pub fn record_update_time(&mut self, d: std::time::Duration) {
        self.update_micros.push_back(d.as_micros() as u32);
        if self.update_micros.len() > self.capacity {
            self.update_micros.pop_front();
        }
    }

    /// Record time spent in eval or checkpoint saving so it is excluded from
    /// the throughput window.
    pub fn record_overhead(&mut self, d: std::time::Duration) {
        self.window_overhead_micros += d.as_micros();
    }

    /// Mean of the last `last_n` episode times in milliseconds.
    pub fn avg_episode_ms(&self, last_n: usize) -> f32 {
        let n = self.episode_micros.len().min(last_n);
        if n == 0 {
            return 0.0;
        }
        let mean = self
            .episode_micros
            .iter()
            .rev()
            .take(n)
            .map(|&v| v as f64)
            .sum::<f64>()
            / n as f64;
        (mean / 1000.0) as f32
    }

    /// Mean of the last `last_n` update times in milliseconds.
    pub fn avg_update_ms(&self, last_n: usize) -> f32 {
        let n = self.update_micros.len().min(last_n);
        if n == 0 {
            return 0.0;
        }
        let mean = self
            .update_micros
            .iter()
            .rev()
            .take(n)
            .map(|&v| v as f64)
            .sum::<f64>()
            / n as f64;
        (mean / 1000.0) as f32
    }

    /// Episodes per second since the last `reset_window` call, excluding time
    /// spent in eval/checkpoint overhead.
    pub fn episodes_per_sec(&self) -> f32 {
        let total_micros = self.window_start.elapsed().as_micros();
        let net_micros = total_micros.saturating_sub(self.window_overhead_micros);
        if net_micros == 0 {
            return 0.0;
        }
        self.window_count as f32 / (net_micros as f32 / 1_000_000.0)
    }

    /// Reset the throughput window (call after each log interval).
    pub fn reset_window(&mut self) {
        self.window_start = std::time::Instant::now();
        self.window_count = 0;
        self.window_overhead_micros = 0;
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

    #[test]
    fn test_overhead_excluded_from_eps_per_sec() {
        let mut t = TimingMetrics::new();
        for _ in 0..10 {
            t.record_episode_time(std::time::Duration::from_micros(1000));
        }
        // Overhead larger than any possible real elapsed time → net_micros = 0
        t.record_overhead(std::time::Duration::from_secs(9999));
        assert_eq!(t.episodes_per_sec(), 0.0);
    }

    #[test]
    fn test_reset_window_clears_overhead() {
        let mut t = TimingMetrics::new();
        for _ in 0..5 {
            t.record_episode_time(std::time::Duration::from_micros(1000));
        }
        t.record_overhead(std::time::Duration::from_secs(9999));
        assert_eq!(t.episodes_per_sec(), 0.0);

        t.reset_window();
        for _ in 0..5 {
            t.record_episode_time(std::time::Duration::from_micros(1000));
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t.episodes_per_sec() > 0.0, "overhead should be cleared after reset");
    }
}
