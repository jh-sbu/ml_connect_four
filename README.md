# ML Connect Four

A Connect Four game with reinforcement learning training, built in Rust. Features a rich terminal UI for both playing and monitoring training in real-time.

## Features

- **Playable TUI Game** — Human vs Human, Human vs AI, or AI vs AI
- **DQN Training** — Deep Q-Network with experience replay and target network
- **Policy Gradient Training** — REINFORCE with PPO clipping, GAE advantages, and rollout buffering
- **AlphaZero Training** — Monte Carlo Tree Search guided by a neural network, Dirichlet noise, temperature scheduling
- **Negamax AI** — Deterministic minimax search agent (depth 7) for gameplay
- **Live Training Dashboard** — Real-time win rate charts, loss curves, live game board, and stats
- **Checkpoint System** — Save/load models, resume training, automatic pruning
- **TOML Configuration** — All hyperparameters configurable via `config.toml`

## Quick Start

### Play

```sh
cargo run
```

Press `1` to open the player selection menu for Red, or `2` for Yellow.
Choose from: Human, Random AI, DQN AI, Policy Gradient AI, Negamax AI, or AlphaZero AI.

### Train

```sh
# Train DQN with live dashboard
cargo run --bin train

# Train DQN headless (stdout output)
cargo run --bin train -- --headless

# Train Policy Gradient
cargo run --bin train -- --algorithm pg

# Train AlphaZero
cargo run --bin train -- --algorithm az

# Resume from checkpoint
cargo run --bin train -- --resume

# Deterministic training with fixed seed
cargo run --bin train -- --seed 42

# Custom episodes and learning rate
cargo run --bin train -- --episodes 5000 --lr 0.001
```

## CLI Options

```
cargo run --bin train -- [OPTIONS]

Options:
  --algorithm <ALGORITHM>  Algorithm to train: dqn, pg, or az [default: dqn]
  --resume                 Resume training from the latest checkpoint
  --headless               Run in headless mode (stdout output, no TUI dashboard)
  --config <CONFIG>        Path to TOML configuration file [default: config.toml]
  --episodes <EPISODES>    Override number of training episodes
  --lr <LR>                Override learning rate
  --seed <SEED>            Fix the random seed for reproducible training
  -h, --help               Print help
```

## Configuration

Copy and edit `config.toml` to customize hyperparameters:

```toml
[dqn]
learning_rate = 0.0001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
batch_size = 64

[pg]
learning_rate = 0.0003
ppo_epsilon = 0.2
entropy_coeff = 0.01
rollout_episodes = 4   # Episodes to buffer before each PPO update

[az]
num_simulations = 100
c_puct = 1.5
dirichlet_alpha = 0.3

[training]
num_episodes = 10000
eval_interval = 500
checkpoint_interval = 1000
base_seed = 42          # Fix seed for reproducible training
num_eval_threads = 4    # Parallel evaluation workers

[checkpoint]
keep_last_n = 5
keep_best_n = 3
```

All fields are optional; omitted values use sensible defaults.

## Architecture

```
src/
  game/          Board, Player, GameState (immutable transitions)
  ai/
    algorithms/  DQN agent, Policy Gradient agent, AlphaZero agent, Negamax agent
    networks/    Conv2D neural networks (Burn framework)
    agent.rs     Agent and TrainableAgent traits
    random.rs    Random AI agent
    negamax.rs   Negamax (minimax) AI agent
  training/
    trainer.rs        Self-play trainer
    episode.rs        Episode logic and parallel evaluation
    replay_buffer.rs  Experience replay
    metrics.rs        Timing and performance metrics
    dashboard_msg.rs  Dashboard messaging types
  checkpoint/    Model persistence, symlink-based latest, pruning
  ui/            Ratatui game view + training dashboard
  config.rs      TOML config loading with validation
  error.rs       Structured error types
```

## Keyboard Shortcuts

### Game

| Key | Action |
|-----|--------|
| Left / Right | Move column selector |
| Enter / Space | Drop piece |
| `1` | Open player menu (Red) |
| `2` | Open player menu (Yellow) |
| Space | Toggle pause (AI vs AI mode) |
| `n` | Step one move (AI vs AI, when paused) |
| `r` | Restart game |
| `q` / Esc | Quit |

**Player menu** (after pressing `1` or `2`):

| Key | Action |
|-----|--------|
| Up / Down | Navigate player types |
| Enter | Confirm selection |
| Esc | Cancel |

Available player types: Human, Random AI, DQN AI, Policy Gradient AI, Negamax AI, AlphaZero AI

### Training Dashboard

| Key | Action |
|-----|--------|
| `p` | Pause / Resume |
| `s` | Save checkpoint |
| `q` | Quit |

## Requirements

- Rust 2024 edition
- GPU support via wgpu (Vulkan, Metal, or DX12)
