//! Terminal UI: game view for playing Connect Four, and a live training
//! dashboard with charts, stats, and a live game board.

mod app;
pub mod board_widget;
mod game_view;
pub mod training_dashboard;
pub mod training_view;

pub use app::App;
