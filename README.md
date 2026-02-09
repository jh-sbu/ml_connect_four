# ML Connect Four

A Connect Four game with reinforcement learning training, built in Rust. Features a rich terminal UI for both playing and monitoring training in real-time.

## Features

- **Playable TUI Game** — Human vs Human, Human vs AI, or AI vs AI
- **DQN Training** — Deep Q-Network with experience replay and target network
- **Policy Gradient Training** — REINFORCE with PPO clipping and GAE advantages
- **Live Training Dashboard** — Real-time win rate charts, loss curves, live game board, and stats
- **Checkpoint System** — Save/load models, resume training, automatic pruning
- **TOML Configuration** — All hyperparameters configurable via `config.toml`

## Quick Start

### Play

```sh
cargo run
```

Use arrow keys to select a column, Enter/Space to drop a piece. Press `a` to toggle a random AI opponent, `d` for DQN AI, or `g` for Policy Gradient AI.

### Train

```sh
# Train DQN with live dashboard
cargo run --bin train

# Train DQN headless (stdout output)
cargo run --bin train -- --headless

# Train Policy Gradient
cargo run --bin train -- --algorithm pg

# Resume from checkpoint
cargo run --bin train -- --resume

# Custom episodes and learning rate
cargo run --bin train -- --episodes 5000 --lr 0.001
```

## CLI Options

```
cargo run --bin train -- [OPTIONS]

Options:
  --algorithm <ALGORITHM>  Algorithm to train: dqn or pg [default: dqn]
  --resume                 Resume training from the latest checkpoint
  --headless               Run in headless mode (stdout output, no TUI dashboard)
  --config <CONFIG>        Path to TOML configuration file [default: config.toml]
  --episodes <EPISODES>    Override number of training episodes
  --lr <LR>                Override learning rate
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

[training]
num_episodes = 10000
eval_interval = 500
checkpoint_interval = 1000
```

All fields are optional; omitted values use sensible defaults.

## Architecture

```
src/
  game/          Board, Player, GameState (immutable transitions)
  ai/
    algorithms/  DQN agent, Policy Gradient agent
    networks/    Conv2D neural networks (Burn framework)
    agent.rs     Agent trait
  training/      Self-play trainer, replay buffer, metrics
  checkpoint/    Model persistence, symlink-based latest, pruning
  ui/            Ratatui game view + training dashboard
  config.rs      TOML config loading with validation
  error.rs       Structured error types
```

## Keyboard Shortcuts

### Game

| Key | Action |
|-----|--------|
| Left/Right | Move column selector |
| Enter/Space | Drop piece |
| `a` | Toggle Random AI (yellow) |
| `d` | Toggle DQN AI (yellow) |
| `g` | Toggle PG AI (yellow) |
| `r` | Restart game |
| `q` / Esc | Quit |

### Training Dashboard

| Key | Action |
|-----|--------|
| `p` | Pause / Resume |
| `s` | Save checkpoint |
| `q` | Quit |

## Requirements

- Rust 2024 edition
- GPU support via wgpu (Vulkan, Metal, or DX12)
