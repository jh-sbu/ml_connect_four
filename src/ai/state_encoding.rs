use burn::prelude::*;
use burn::tensor::TensorData;

use crate::game::{Board, Cell, GameState, COLS, ROWS};

/// Encode a game state as a tensor of shape [3, 6, 7] (player-relative).
///
/// Channel 0: Current player's pieces (1.0 where placed)
/// Channel 1: Opponent's pieces (1.0 where placed)
/// Channel 2: Valid move mask (1.0 at the landing row of each legal column)
pub fn encode_state<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 3> {
    let data = encode_state_flat(state);
    Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), device).reshape([3, 6, 7])
}

/// Encode multiple game states as a batched tensor of shape [batch, 3, 6, 7].
pub fn encode_states_batch<B: Backend>(states: &[GameState], device: &B::Device) -> Tensor<B, 4> {
    let batch_size = states.len();
    let mut flat = Vec::with_capacity(batch_size * 3 * ROWS * COLS);
    for state in states {
        flat.extend_from_slice(&encode_state_flat(state));
    }
    Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), device)
        .reshape([batch_size as i32, 3, 6, 7])
}

/// Produce the flat [126] f32 array for a single state encoding.
fn encode_state_flat(state: &GameState) -> [f32; 3 * ROWS * COLS] {
    let mut data = [0.0f32; 3 * ROWS * COLS];
    let board = state.board();
    let current = state.current_player();
    let current_cell = current.to_cell();
    let opponent_cell = current.other().to_cell();

    // Channel 0: current player's pieces
    // Channel 1: opponent's pieces
    for row in 0..ROWS {
        for col in 0..COLS {
            let cell = board.get(row, col);
            let idx = row * COLS + col;
            if cell == current_cell {
                data[idx] = 1.0; // channel 0
            } else if cell == opponent_cell {
                data[ROWS * COLS + idx] = 1.0; // channel 1
            }
        }
    }

    // Channel 2: valid move mask (1.0 at the landing row of each legal column)
    let ch2_offset = 2 * ROWS * COLS;
    for col in state.legal_actions() {
        // Find the lowest empty row in this column
        if let Some(landing_row) = find_landing_row(board, col) {
            data[ch2_offset + landing_row * COLS + col] = 1.0;
        }
    }

    data
}

/// Find the lowest empty row in a column (where a piece would land).
fn find_landing_row(board: &Board, col: usize) -> Option<usize> {
    for row in (0..ROWS).rev() {
        if board.get(row, col) == Cell::Empty {
            return Some(row);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu<f32, i32>;

    #[test]
    fn test_encode_initial_state_shape() {
        let state = GameState::initial();
        let device = Default::default();
        let tensor = encode_state::<TestBackend>(&state, &device);
        assert_eq!(tensor.shape().dims, [3, 6, 7]);
    }

    #[test]
    fn test_encode_initial_state_values() {
        let state = GameState::initial();
        let device = Default::default();
        let tensor = encode_state::<TestBackend>(&state, &device);
        let data: Vec<f32> = tensor.into_data().to_vec().unwrap();

        // Channels 0 and 1 should be all zeros (empty board)
        for i in 0..84 {
            assert_eq!(data[i], 0.0, "Channel 0/1 index {} should be 0", i);
        }

        // Channel 2 should have 1.0 at bottom row (row 5) for all 7 columns
        for col in 0..7 {
            let idx = 84 + 5 * 7 + col; // channel 2, row 5, col
            assert_eq!(data[idx], 1.0, "Channel 2, row 5, col {} should be 1.0", col);
        }
    }

    #[test]
    fn test_encode_after_one_move() {
        let state = GameState::initial().apply_move(3).unwrap();
        // Now it's Yellow's turn. Red placed at (5, 3).
        let device = Default::default();
        let tensor = encode_state::<TestBackend>(&state, &device);
        let data: Vec<f32> = tensor.into_data().to_vec().unwrap();

        // Channel 0 is Yellow's pieces (current player) - should be empty
        assert_eq!(data[5 * 7 + 3], 0.0);

        // Channel 1 is Red's pieces (opponent) - Red at (5, 3)
        assert_eq!(data[42 + 5 * 7 + 3], 1.0);

        // Channel 2: col 3 should now land at row 4
        assert_eq!(data[84 + 4 * 7 + 3], 1.0);
        // Row 5 col 3 should be 0 (occupied)
        assert_eq!(data[84 + 5 * 7 + 3], 0.0);
    }

    #[test]
    fn test_encode_batch() {
        let s1 = GameState::initial();
        let s2 = s1.apply_move(3).unwrap();
        let device = Default::default();
        let batch = encode_states_batch::<TestBackend>(&[s1, s2], &device);
        assert_eq!(batch.shape().dims, [2, 3, 6, 7]);
    }
}
