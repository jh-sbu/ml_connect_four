#![recursion_limit = "256"]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use ml_connect_four::ai::algorithms::{AlphaZeroAgent, DqnAgent, PolicyGradientAgent};
use ml_connect_four::ai::TrainableAgent;
use ml_connect_four::checkpoint::{CheckpointManager, CheckpointManagerConfig};
use ml_connect_four::config::AppConfig;
use ml_connect_four::training::dashboard_msg::{TrainingCommand, TrainingUpdate};
use ml_connect_four::training::trainer::{Trainer, TrainerConfig};
use ml_connect_four::ui::training_dashboard::{DashboardState, TrainingStatus};
use ml_connect_four::ui::training_view;

/// Train a Connect Four RL agent via self-play.
#[derive(Parser)]
#[command(name = "train", about = "Train a Connect Four RL agent")]
struct Cli {
    /// Algorithm to train: dqn or pg
    #[arg(long, default_value = "dqn")]
    algorithm: String,

    /// Resume training from the latest checkpoint
    #[arg(long)]
    resume: bool,

    /// Run in headless mode (stdout output, no TUI dashboard)
    #[arg(long)]
    headless: bool,

    /// Path to TOML configuration file
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    /// Override number of training episodes
    #[arg(long)]
    episodes: Option<usize>,

    /// Override learning rate
    #[arg(long)]
    lr: Option<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate algorithm
    match cli.algorithm.as_str() {
        "dqn" | "pg" | "az" => {}
        other => bail!("unknown algorithm '{}' (expected 'dqn', 'pg', or 'az')", other),
    }

    // Load configuration
    let mut app_config = AppConfig::load_or_default(&cli.config)
        .with_context(|| format!("loading config from {}", cli.config.display()))?;

    // Apply CLI overrides
    if let Some(episodes) = cli.episodes {
        app_config.training.num_episodes = episodes;
    }
    if let Some(lr) = cli.lr {
        match cli.algorithm.as_str() {
            "dqn" => app_config.dqn.learning_rate = lr,
            "pg" => app_config.pg.learning_rate = lr,
            "az" => app_config.az.learning_rate = lr,
            _ => {}
        }
    }

    // Use algorithm-appropriate checkpoint directory.
    match cli.algorithm.as_str() {
        "pg" => app_config.training.checkpoint_dir = PathBuf::from("pg_checkpoints"),
        "az" => app_config.training.checkpoint_dir = PathBuf::from("az_checkpoints"),
        _ => {}
    }

    let trainer_config = app_config.training.clone();
    let total_episodes = trainer_config.num_episodes;

    match cli.algorithm.as_str() {
        "dqn" => {
            let mut agent = DqnAgent::new(app_config.dqn.clone());
            if cli.resume {
                resume_agent(&mut agent, &trainer_config, &app_config, cli.headless)?;
            }
            if cli.headless {
                let trainer = Trainer::new(trainer_config);
                trainer.train(&mut agent);
                Ok(())
            } else {
                run_dashboard(agent, trainer_config, total_episodes)
            }
        }
        "pg" => {
            let mut agent = PolicyGradientAgent::new(app_config.pg.clone());
            if cli.resume {
                resume_agent(&mut agent, &trainer_config, &app_config, cli.headless)?;
            }
            if cli.headless {
                let trainer = Trainer::new(trainer_config);
                trainer.train(&mut agent);
                Ok(())
            } else {
                run_dashboard(agent, trainer_config, total_episodes)
            }
        }
        "az" => {
            let mut agent = AlphaZeroAgent::new(app_config.az.clone());
            if cli.resume {
                resume_agent(&mut agent, &trainer_config, &app_config, cli.headless)?;
            }
            if cli.headless {
                let trainer = Trainer::new(trainer_config);
                trainer.train(&mut agent);
                Ok(())
            } else {
                run_dashboard(agent, trainer_config, total_episodes)
            }
        }
        _ => unreachable!(),
    }
}

/// Resume an agent from the latest checkpoint using trait-based loading.
fn resume_agent(
    agent: &mut dyn TrainableAgent,
    trainer_config: &TrainerConfig,
    config: &AppConfig,
    headless: bool,
) -> Result<()> {
    let manager = CheckpointManager::new(CheckpointManagerConfig {
        checkpoint_dir: trainer_config.checkpoint_dir.clone(),
        ..config.checkpoint.clone()
    });
    match manager.load_agent_latest() {
        Ok(data) => {
            agent
                .load_weights_from_dir(&data.path)
                .map_err(|e| anyhow::anyhow!("loading checkpoint weights: {e}"))?;
            agent
                .restore_training_state_json(&data.training_state_json)
                .map_err(|e| anyhow::anyhow!("restoring training state: {e}"))?;
            if headless {
                println!("Resumed from episode {}", data.metadata.episode);
            }
        }
        Err(e) => {
            if headless {
                println!("No checkpoint found ({}), starting fresh", e);
            }
        }
    }
    Ok(())
}

fn run_dashboard<A: TrainableAgent + Send + 'static>(
    agent: A,
    trainer_config: TrainerConfig,
    total_episodes: usize,
) -> Result<()> {
    let algorithm = agent.algorithm_name().to_string();

    let (update_tx, update_rx) = mpsc::channel::<TrainingUpdate>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<TrainingCommand>();

    let pause = Arc::new(AtomicBool::new(false));
    let quit = Arc::new(AtomicBool::new(false));

    let pause_clone = pause.clone();
    let quit_clone = quit.clone();

    let training_handle = std::thread::spawn(move || {
        let mut agent = agent;
        let trainer = Trainer::new(trainer_config);
        trainer.train_with_dashboard(&mut agent, update_tx, cmd_rx, pause_clone, quit_clone);
    });

    run_dashboard_ui(update_rx, cmd_tx, pause, quit, total_episodes, &algorithm)?;

    let _ = training_handle.join();
    Ok(())
}

fn run_dashboard_ui(
    update_rx: mpsc::Receiver<TrainingUpdate>,
    cmd_tx: mpsc::Sender<TrainingCommand>,
    pause: Arc<AtomicBool>,
    quit: Arc<AtomicBool>,
    total_episodes: usize,
    algorithm: &str,
) -> Result<()> {
    enable_raw_mode().context("enabling raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("entering alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("creating terminal")?;

    let mut dashboard = DashboardState::new(total_episodes);
    dashboard.algorithm = algorithm.to_string();

    let frame_duration = Duration::from_millis(100);

    loop {
        while let Ok(update) = update_rx.try_recv() {
            match update {
                TrainingUpdate::Metrics(snap) => {
                    dashboard.apply_metrics(&snap);
                }
                TrainingUpdate::LiveGame(live) => {
                    dashboard.live_game = Some(live.game_state);
                    dashboard.live_move_number = live.move_number;
                }
                TrainingUpdate::EvalResult {
                    episode: _,
                    win_rate,
                } => {
                    dashboard.last_eval_win_rate = Some(win_rate);
                    let ep = dashboard.episode as f64;
                    dashboard.eval_history.push_back((ep, win_rate as f64));
                    if dashboard.eval_history.len() > 500 {
                        dashboard.eval_history.pop_front();
                    }
                }
                TrainingUpdate::CheckpointSaved { episode, path } => {
                    dashboard.last_checkpoint =
                        Some(format!("ep {} ({})", episode, path.display()));
                }
                TrainingUpdate::Finished => {
                    dashboard.status = TrainingStatus::Finished;
                }
            }
        }

        terminal
            .draw(|f| training_view::render(f, &dashboard))
            .context("drawing dashboard")?;

        if event::poll(frame_duration).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => {
                        quit.store(true, Ordering::Relaxed);
                        break;
                    }
                    KeyCode::Char('p') | KeyCode::Char('P') => {
                        let was_paused = pause.load(Ordering::Relaxed);
                        pause.store(!was_paused, Ordering::Relaxed);
                        dashboard.status = if was_paused {
                            TrainingStatus::Running
                        } else {
                            TrainingStatus::Paused
                        };
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => {
                        let _ = cmd_tx.send(TrainingCommand::SaveCheckpoint);
                    }
                    _ => {}
                }
            }
        }

        if dashboard.status == TrainingStatus::Finished {
            // Give user time to see final state; they press q to exit
        }
    }

    // Terminal cleanup — use let _ = to avoid double-panic
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    Ok(())
}
