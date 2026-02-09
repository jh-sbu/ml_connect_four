#![recursion_limit = "256"]

use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use ml_connect_four::ai::algorithms::{DqnAgent, DqnConfig, PgConfig, PolicyGradientAgent};
use ml_connect_four::checkpoint::{CheckpointManager, CheckpointManagerConfig};
use ml_connect_four::training::dashboard_msg::{TrainingCommand, TrainingUpdate};
use ml_connect_four::training::trainer::{Trainer, TrainerConfig};
use ml_connect_four::ui::training_dashboard::{DashboardState, TrainingStatus};
use ml_connect_four::ui::training_view;

fn main() {
    let args: Vec<String> = env::args().collect();
    let resume = args.iter().any(|a| a == "--resume");
    let headless = args.iter().any(|a| a == "--headless");
    let algorithm = args
        .iter()
        .position(|a| a == "--algorithm")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("dqn");

    match algorithm {
        "dqn" => run_dqn(resume, headless),
        "pg" => run_pg(resume, headless),
        _ => {
            eprintln!("Unknown algorithm '{}'. Use 'dqn' or 'pg'.", algorithm);
            std::process::exit(1);
        }
    }
}

fn run_dqn(resume: bool, headless: bool) {
    let trainer_config = TrainerConfig::default();
    let total_episodes = trainer_config.num_episodes;
    let mut agent = DqnAgent::new(DqnConfig::default());

    if resume {
        let manager = CheckpointManager::new(CheckpointManagerConfig {
            checkpoint_dir: trainer_config.checkpoint_dir.clone(),
            ..Default::default()
        });
        match manager.load_latest() {
            Ok(data) => {
                agent
                    .load_from_dir(&data.path)
                    .expect("Failed to load checkpoint weights");
                agent.restore_training_state(&data.training_state);
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
    }

    if headless {
        let trainer = Trainer::new(trainer_config);
        trainer.train(&mut agent);
        return;
    }

    run_dashboard_dqn(agent, trainer_config, total_episodes);
}

fn run_pg(resume: bool, headless: bool) {
    let trainer_config = TrainerConfig {
        checkpoint_dir: PathBuf::from("pg_checkpoints"),
        ..Default::default()
    };
    let total_episodes = trainer_config.num_episodes;
    let mut agent = PolicyGradientAgent::new(PgConfig::default());

    if resume {
        let manager = CheckpointManager::new(CheckpointManagerConfig {
            checkpoint_dir: trainer_config.checkpoint_dir.clone(),
            ..Default::default()
        });
        match manager.load_pg_latest() {
            Ok(data) => {
                agent
                    .load_from_dir(&data.path)
                    .expect("Failed to load PG checkpoint weights");
                agent.restore_training_state(&data.training_state);
                if headless {
                    println!("Resumed from episode {}", data.metadata.episode);
                }
            }
            Err(e) => {
                if headless {
                    println!("No PG checkpoint found ({}), starting fresh", e);
                }
            }
        }
    }

    if headless {
        let trainer = Trainer::new(trainer_config);
        trainer.train_pg(&mut agent);
        return;
    }

    run_dashboard_pg(agent, trainer_config, total_episodes);
}

fn run_dashboard_dqn(agent: DqnAgent, trainer_config: TrainerConfig, total_episodes: usize) {
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

    run_dashboard_ui(update_rx, cmd_tx, pause, quit, total_episodes, "DQN");

    let _ = training_handle.join();
}

fn run_dashboard_pg(
    agent: PolicyGradientAgent,
    trainer_config: TrainerConfig,
    total_episodes: usize,
) {
    let (update_tx, update_rx) = mpsc::channel::<TrainingUpdate>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<TrainingCommand>();

    let pause = Arc::new(AtomicBool::new(false));
    let quit = Arc::new(AtomicBool::new(false));

    let pause_clone = pause.clone();
    let quit_clone = quit.clone();

    let training_handle = std::thread::spawn(move || {
        let mut agent = agent;
        let trainer = Trainer::new(trainer_config);
        trainer.train_pg_with_dashboard(&mut agent, update_tx, cmd_rx, pause_clone, quit_clone);
    });

    run_dashboard_ui(update_rx, cmd_tx, pause, quit, total_episodes, "PG");

    let _ = training_handle.join();
}

fn run_dashboard_ui(
    update_rx: mpsc::Receiver<TrainingUpdate>,
    cmd_tx: mpsc::Sender<TrainingCommand>,
    pause: Arc<AtomicBool>,
    quit: Arc<AtomicBool>,
    total_episodes: usize,
    algorithm: &str,
) {
    enable_raw_mode().expect("Failed to enable raw mode");
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).expect("Failed to enter alternate screen");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("Failed to create terminal");

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
            .expect("Failed to draw");

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

    disable_raw_mode().expect("Failed to disable raw mode");
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .expect("Failed to leave alternate screen");
    terminal.show_cursor().expect("Failed to show cursor");
}
