//! Core Connect Four game logic: board representation, player types, and game
//! state machine with immutable transitions.

mod board;
mod player;
mod state;

pub use board::{Board, Cell, COLS, ROWS};
pub use player::Player;
pub use state::{GameOutcome, GameState, MoveError};
