//! # ML Connect Four
//!
//! A Connect Four game with reinforcement learning training capabilities.
//! Features a terminal UI built with Ratatui and supports DQN and Policy Gradient
//! (PPO) algorithms via the Burn ML framework.
//!
//! ## Modules
//!
//! - [`game`] — Core game logic: board, player, state machine
//! - [`ai`] — Agent trait, algorithms (DQN, PG), neural networks, state encoding
//! - [`training`] — Self-play trainer, replay buffer, metrics collection
//! - [`checkpoint`] — Model persistence and versioning
//! - [`ui`] — Terminal UI: game view, training dashboard
//! - [`config`] — TOML configuration loading and validation
//! - [`error`] — Structured error types

#![recursion_limit = "256"]

pub mod ai;
pub mod checkpoint;
pub mod config;
pub mod error;
pub mod game;
pub mod training;
pub mod ui;
