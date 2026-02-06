use std::collections::VecDeque;

use crate::game::GameState;
use crate::training::dashboard_msg::MetricsSnapshot;

const MAX_HISTORY: usize = 500;

/// Status of the training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingStatus {
    Running,
    Paused,
    Finished,
}

/// Dashboard state holding history buffers and current values.
pub struct DashboardState {
    // History buffers: (episode, value)
    pub win_rate_history: VecDeque<(f64, f64)>,
    pub loss_history: VecDeque<(f64, f64)>,
    pub eval_history: VecDeque<(f64, f64)>,
    pub game_length_history: VecDeque<u64>,

    // Current values
    pub episode: usize,
    pub total_episodes: usize,
    pub epsilon: f32,
    pub win_rate: f32,
    pub draw_rate: f32,
    pub loss: f32,
    pub avg_game_length: f32,
    pub step_count: usize,

    // Live game
    pub live_game: Option<GameState>,
    pub live_move_number: usize,

    // Status
    pub status: TrainingStatus,
    pub last_checkpoint: Option<String>,
    pub last_eval_win_rate: Option<f32>,
}

impl DashboardState {
    pub fn new(total_episodes: usize) -> Self {
        DashboardState {
            win_rate_history: VecDeque::new(),
            loss_history: VecDeque::new(),
            eval_history: VecDeque::new(),
            game_length_history: VecDeque::new(),

            episode: 0,
            total_episodes,
            epsilon: 1.0,
            win_rate: 0.0,
            draw_rate: 0.0,
            loss: 0.0,
            avg_game_length: 0.0,
            step_count: 0,

            live_game: None,
            live_move_number: 0,

            status: TrainingStatus::Running,
            last_checkpoint: None,
            last_eval_win_rate: None,
        }
    }

    /// Apply a metrics snapshot from the training thread.
    pub fn apply_metrics(&mut self, snap: &MetricsSnapshot) {
        self.episode = snap.episode;
        self.total_episodes = snap.total_episodes;
        self.epsilon = snap.epsilon;
        self.win_rate = snap.win_rate;
        self.draw_rate = snap.draw_rate;
        self.loss = snap.loss;
        self.avg_game_length = snap.avg_game_length;
        self.step_count = snap.step_count;

        // Append to histories
        let ep = snap.episode as f64;
        self.win_rate_history.push_back((ep, snap.win_rate as f64));
        if self.win_rate_history.len() > MAX_HISTORY {
            self.win_rate_history.pop_front();
        }

        self.loss_history.push_back((ep, snap.loss as f64));
        if self.loss_history.len() > MAX_HISTORY {
            self.loss_history.pop_front();
        }

        self.game_length_history
            .push_back(snap.avg_game_length as u64);
        if self.game_length_history.len() > MAX_HISTORY {
            self.game_length_history.pop_front();
        }
    }

    /// Progress ratio [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.total_episodes == 0 {
            return 0.0;
        }
        self.episode as f64 / self.total_episodes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_metrics_updates_fields() {
        let mut state = DashboardState::new(1000);
        let snap = MetricsSnapshot {
            episode: 100,
            total_episodes: 1000,
            epsilon: 0.8,
            win_rate: 0.65,
            draw_rate: 0.1,
            loss: 0.05,
            avg_game_length: 18.5,
            step_count: 500,
        };
        state.apply_metrics(&snap);

        assert_eq!(state.episode, 100);
        assert_eq!(state.total_episodes, 1000);
        assert!((state.epsilon - 0.8).abs() < 1e-6);
        assert!((state.win_rate - 0.65).abs() < 1e-6);
        assert!((state.draw_rate - 0.1).abs() < 1e-6);
        assert!((state.loss - 0.05).abs() < 1e-6);
        assert!((state.avg_game_length - 18.5).abs() < 1e-6);
        assert_eq!(state.step_count, 500);
        assert_eq!(state.win_rate_history.len(), 1);
        assert_eq!(state.loss_history.len(), 1);
        assert_eq!(state.game_length_history.len(), 1);
    }

    #[test]
    fn test_history_caps_at_500() {
        let mut state = DashboardState::new(10_000);
        for i in 0..600 {
            let snap = MetricsSnapshot {
                episode: i,
                total_episodes: 10_000,
                epsilon: 0.5,
                win_rate: 0.5,
                draw_rate: 0.1,
                loss: 0.1,
                avg_game_length: 20.0,
                step_count: i * 10,
            };
            state.apply_metrics(&snap);
        }

        assert_eq!(state.win_rate_history.len(), 500);
        assert_eq!(state.loss_history.len(), 500);
        assert_eq!(state.game_length_history.len(), 500);
    }

    #[test]
    fn test_progress() {
        let mut state = DashboardState::new(1000);
        assert!((state.progress() - 0.0).abs() < 1e-6);

        state.episode = 500;
        assert!((state.progress() - 0.5).abs() < 1e-6);

        state.episode = 1000;
        assert!((state.progress() - 1.0).abs() < 1e-6);
    }
}
