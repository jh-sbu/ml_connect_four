use rand::rngs::StdRng;
use rand::seq::index;
use rand::SeedableRng;

use crate::ai::Experience;

/// Fixed-capacity ring buffer for storing training experiences.
pub struct ReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    position: usize,
    len: usize,
    rng: StdRng,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
            len: 0,
            rng: StdRng::from_os_rng(),
        }
    }

    /// Add an experience to the buffer. Overwrites oldest when full.
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
        }
        self.position = (self.position + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Sample a random batch of experiences.
    pub fn sample(&mut self, batch_size: usize) -> Vec<Experience> {
        assert!(batch_size <= self.len, "Not enough experiences to sample");
        let indices = index::sample(&mut self.rng, self.len, batch_size);
        indices.iter().map(|i| self.buffer[i].clone()).collect()
    }

    /// Sample a random batch into a pre-allocated Vec. Clears `out` first.
    pub fn sample_into(&mut self, batch_size: usize, out: &mut Vec<Experience>) {
        assert!(batch_size <= self.len, "Not enough experiences to sample");
        let indices = index::sample(&mut self.rng, self.len, batch_size);
        out.clear();
        out.extend(indices.iter().map(|i| self.buffer[i].clone()));
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{GameState, Player};

    fn dummy_experience() -> Experience {
        let state = GameState::initial();
        let next_state = state.apply_move(0).unwrap();
        Experience {
            state,
            action: 0,
            reward: 0.0,
            next_state,
            done: false,
            player: Player::Red,
        }
    }

    #[test]
    fn test_push_and_len() {
        let mut buf = ReplayBuffer::new(10);
        assert_eq!(buf.len(), 0);

        buf.push(dummy_experience());
        assert_eq!(buf.len(), 1);

        for _ in 0..9 {
            buf.push(dummy_experience());
        }
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_ring_buffer_overwrites() {
        let mut buf = ReplayBuffer::new(5);
        for _ in 0..10 {
            buf.push(dummy_experience());
        }
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_sample() {
        let mut buf = ReplayBuffer::new(100);
        for _ in 0..50 {
            buf.push(dummy_experience());
        }
        let batch = buf.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    #[should_panic(expected = "Not enough experiences")]
    fn test_sample_too_many() {
        let mut buf = ReplayBuffer::new(10);
        buf.push(dummy_experience());
        buf.sample(5);
    }
}
