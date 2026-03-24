#![allow(clippy::module_name_repetitions)]

//! FerrumChem Phase 14 TUI — workflow shell.
//!
//! This file implements an interactive terminal shell that upgrades the initial
//! menu scaffold into concrete, navigable workflow screens:
//! - training wizard (real CSV/SMILES ingestion + AutoQsar training)
//! - batch prediction preview (using trained pipeline)
//! - chemical space summary (feature projection preview)
//! - descriptor profiling summary (feature stats preview)
//! - pipeline management (file-backed save/load with in-memory mirror)
//!
//! The goal here is to establish stable UX structure and plumbing to core ML
//! APIs with explicit state transitions and file-backed workflows.

use std::env;
use std::fs;
use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, ListState, Paragraph, Row, Table, Wrap,
    canvas::{Canvas, Points},
};

use ferrum_core::desc::batch::{BatchDescriptor, calculate_batch_flat_static};
use ferrum_core::io::csv_reader::CsvReader;
use ferrum_core::ml::dataset::QsarDataset;
use ferrum_core::ml::metrics;
use ferrum_core::ml::pca::Pca;
use ferrum_core::ml::pipeline::AutoQsar;
use ferrum_core::ml::pipeline::QsarPipeline;
use ferrum_core::ml::scaling::StandardScaler;

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const TRAINING_PREVIEW_ROWS: usize = 6;
const DEFAULT_DATASET_CSV_PATH: &str = "coconut_csv_lite-02-2026.csv";
const DEFAULT_DATASET_SMILES_COLUMN: &str = "canonical_smiles";
const DEFAULT_DATASET_TARGET_COLUMN: &str = "alogp";
const DEFAULT_DATASET_LIMIT: usize = 300;
const DEFAULT_PIPELINE_PATH: &str = "/tmp/ferrum_tui_pipeline.bin";

/// Descriptor bundle used for the TUI training workflow.
///
/// We intentionally reuse the same static-dispatch descriptor set used by
/// parity benchmarks so the TUI's model-training path exercises production
/// descriptor kernels rather than a reduced toy feature subset.
const TUI_TRAINING_DESCRIPTORS: [BatchDescriptor; 24] = [
    BatchDescriptor::MolecularWeight,
    BatchDescriptor::LogP,
    BatchDescriptor::MolarRefractivity,
    BatchDescriptor::Tpsa,
    BatchDescriptor::HBondDonor,
    BatchDescriptor::HBondAcceptor,
    BatchDescriptor::RotatableBondCount,
    BatchDescriptor::HeavyAtomCount,
    BatchDescriptor::AtomCount,
    BatchDescriptor::AromaticAtomCount,
    BatchDescriptor::HeteroatomCount,
    BatchDescriptor::AromaticRingCount,
    BatchDescriptor::FormalCharge,
    BatchDescriptor::FractionCSP3,
    BatchDescriptor::BertzCT,
    BatchDescriptor::LabuteAsa,
    BatchDescriptor::Qed,
    BatchDescriptor::NumAmideBonds,
    BatchDescriptor::HallKierAlpha,
    BatchDescriptor::BalabanIndex,
    BatchDescriptor::BondCount,
    BatchDescriptor::SlogpVsa,
    BatchDescriptor::SmrVsa,
    BatchDescriptor::PeoeVsa,
];

#[derive(Debug, Clone)]
struct AppConfig {
    dataset_csv_path: String,
    dataset_smiles_column: String,
    dataset_target_column: String,
    dataset_row_limit: usize,
    pipeline_path: String,
}

impl AppConfig {
    /// Reads TUI runtime configuration from environment variables.
    ///
    /// Supported variables:
    /// - `FERRUM_TUI_DATASET_CSV`
    /// - `FERRUM_TUI_SMILES_COLUMN`
    /// - `FERRUM_TUI_TARGET_COLUMN`
    /// - `FERRUM_TUI_ROW_LIMIT`
    /// - `FERRUM_TUI_PIPELINE_PATH`
    fn from_env() -> Self {
        Self {
            dataset_csv_path: env::var("FERRUM_TUI_DATASET_CSV")
                .unwrap_or_else(|_| DEFAULT_DATASET_CSV_PATH.to_string()),
            dataset_smiles_column: env::var("FERRUM_TUI_SMILES_COLUMN")
                .unwrap_or_else(|_| DEFAULT_DATASET_SMILES_COLUMN.to_string()),
            dataset_target_column: env::var("FERRUM_TUI_TARGET_COLUMN")
                .unwrap_or_else(|_| DEFAULT_DATASET_TARGET_COLUMN.to_string()),
            dataset_row_limit: env::var("FERRUM_TUI_ROW_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(DEFAULT_DATASET_LIMIT),
            pipeline_path: env::var("FERRUM_TUI_PIPELINE_PATH")
                .unwrap_or_else(|_| DEFAULT_PIPELINE_PATH.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
struct CsvDatasetSummary {
    csv_path: String,
    smiles_column: String,
    target_column: String,
    requested_rows: usize,
    parsed_rows: usize,
    parse_failures: usize,
    missing_targets: usize,
    loaded_rows: usize,
    feature_count: usize,
    target_min: f64,
    target_max: f64,
    target_mean: f64,
}

impl CsvDatasetSummary {
    fn short_status(&self) -> String {
        format!(
            "Loaded CSV dataset: {} rows ({} features), target '{}' mean={:.3}.",
            self.loaded_rows, self.feature_count, self.target_column, self.target_mean
        )
    }
}

fn dataset_feature_names(descriptors: &[BatchDescriptor]) -> Vec<String> {
    let mut names = Vec::new();
    for descriptor in descriptors {
        let len = descriptor.feature_len();
        if len == 1 {
            names.push(descriptor.name().to_string());
        } else {
            for idx in 0..len {
                names.push(format!("{}[{}]", descriptor.name(), idx));
            }
        }
    }
    names
}

fn parse_target_value(properties: &[(String, String)], target_column: &str) -> Option<f64> {
    let raw = properties
        .iter()
        .find(|(name, _)| name == target_column)
        .map(|(_, value)| value.trim())?;
    if raw.is_empty() {
        return None;
    }
    raw.parse::<f64>().ok()
}

fn load_training_dataset(config: &AppConfig) -> std::result::Result<(QsarDataset, CsvDatasetSummary), String> {
    let reader = CsvReader::from_path(&config.dataset_csv_path)
        .map_err(|error| format!("Failed to open dataset '{}': {}", config.dataset_csv_path, error))?
        .with_smiles_column(&config.dataset_smiles_column)
        .with_row_limit(config.dataset_row_limit);

    let records = reader.parse().map_err(|error| {
        format!(
            "Failed to parse dataset '{}' as CSV: {}",
            config.dataset_csv_path, error
        )
    })?;

    let mut parsed_rows = 0usize;
    let mut parse_failures = 0usize;
    let mut missing_targets = 0usize;

    let mut molecules = Vec::new();
    let mut targets = Vec::new();
    let mut identifiers = Vec::new();

    for record in records {
        parsed_rows += 1;
        let Some(molecule) = record.molecule else {
            parse_failures += 1;
            continue;
        };
        let Some(target) = parse_target_value(&record.properties, &config.dataset_target_column) else {
            missing_targets += 1;
            continue;
        };
        molecules.push(molecule);
        targets.push(target);
        identifiers.push(record.smiles);
    }

    if molecules.is_empty() {
        return Err(format!(
            "No trainable rows found in '{}'. Parsed={}, parse_failures={}, missing_target='{}':{}.",
            config.dataset_csv_path, parsed_rows, parse_failures, config.dataset_target_column, missing_targets
        ));
    }

    let feature_names = dataset_feature_names(&TUI_TRAINING_DESCRIPTORS);
    let feature_matrix = calculate_batch_flat_static(&mut molecules, &TUI_TRAINING_DESCRIPTORS);
    if feature_matrix.cols != feature_names.len() {
        return Err(format!(
            "Descriptor shape mismatch: cols={} but feature_names={}.",
            feature_matrix.cols,
            feature_names.len()
        ));
    }

    let mut rows = Vec::with_capacity(feature_matrix.rows);
    for row_idx in 0..feature_matrix.rows {
        rows.push(feature_matrix.row(row_idx).to_vec());
    }

    let target_min = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let target_max = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;

    let dataset = QsarDataset::from_f32_vectors(&rows, targets, feature_names).with_identifiers(identifiers);
    let summary = CsvDatasetSummary {
        csv_path: config.dataset_csv_path.clone(),
        smiles_column: config.dataset_smiles_column.clone(),
        target_column: config.dataset_target_column.clone(),
        requested_rows: config.dataset_row_limit,
        parsed_rows,
        parse_failures,
        missing_targets,
        loaded_rows: dataset.n_samples(),
        feature_count: dataset.n_features(),
        target_min,
        target_max,
        target_mean,
    };

    Ok((dataset, summary))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuAction {
    TrainingWizard,
    BatchPrediction,
    ChemicalSpace,
    DescriptorProfiling,
    PipelineManagement,
    Exit,
}

impl MenuAction {
    fn title(self) -> &'static str {
        match self {
            Self::TrainingWizard => "Interactive model training wizard",
            Self::BatchPrediction => "Batch prediction interface",
            Self::ChemicalSpace => "Chemical space visualization",
            Self::DescriptorProfiling => "Descriptor profiling dashboard",
            Self::PipelineManagement => "Pipeline management",
            Self::Exit => "Exit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Menu,
    TrainingWizard,
    BatchPrediction,
    ChemicalSpace,
    DescriptorProfiling,
    PipelineManagement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WizardStep {
    Dataset,
    Preprocess,
    Train,
    Evaluate,
    Done,
}

impl WizardStep {
    fn title(self) -> &'static str {
        match self {
            Self::Dataset => "Dataset setup",
            Self::Preprocess => "Preprocessing setup",
            Self::Train => "Model training",
            Self::Evaluate => "Evaluation",
            Self::Done => "Completed",
        }
    }
}

#[derive(Debug, Clone)]
struct TrainingSummary {
    pipeline_name: String,
    cv_q2: f64,
    train_r2: f64,
    preview_predictions: Vec<f64>,
}

/// One candidate model evaluated during AutoQsar training.
#[derive(Debug, Clone)]
struct ModelCandidate {
    name: String,
    cv_q2: f64,
    train_r2: f64,
    is_best: bool,
}

#[derive(Debug)]
struct App {
    items: Vec<MenuAction>,
    selected: usize,
    view: View,
    status_message: String,
    config: AppConfig,
    pipeline: Option<QsarPipeline>,
    pipeline_bytes: Option<Vec<u8>>,
    dataset: QsarDataset,
    dataset_summary: Option<CsvDatasetSummary>,
    dataset_load_error: Option<String>,
    wizard_step: WizardStep,
    wizard_summary: Option<TrainingSummary>,
    /// Scroll offset for views that need scrolling (e.g. descriptor profiling).
    scroll_offset: usize,
    /// All model candidates tried during training, for comparison table.
    model_candidates: Vec<ModelCandidate>,
}

impl App {
    fn new() -> Self {
        let items = vec![
            MenuAction::TrainingWizard,
            MenuAction::BatchPrediction,
            MenuAction::ChemicalSpace,
            MenuAction::DescriptorProfiling,
            MenuAction::PipelineManagement,
            MenuAction::Exit,
        ];

        let config = AppConfig::from_env();
        let (dataset, dataset_summary, dataset_load_error, status_message) =
            match load_training_dataset(&config) {
                Ok((dataset, summary)) => (
                    dataset,
                    Some(summary.clone()),
                    None,
                    format!("Phase 14 workflow shell ready. {}", summary.short_status()),
                ),
                Err(error) => (
                    build_mock_dataset(),
                    None,
                    Some(error.clone()),
                    format!(
                        "Phase 14 workflow shell ready with fallback mock dataset. CSV load failed: {}",
                        error
                    ),
                ),
            };

        Self {
            items,
            selected: 0,
            view: View::Menu,
            status_message,
            config,
            pipeline: None,
            pipeline_bytes: None,
            dataset,
            dataset_summary,
            dataset_load_error,
            wizard_step: WizardStep::Dataset,
            wizard_summary: None,
            scroll_offset: 0,
            model_candidates: Vec::new(),
        }
    }

    fn reload_dataset_from_csv(&mut self) -> std::result::Result<(), String> {
        let (dataset, summary) = load_training_dataset(&self.config)?;
        self.dataset = dataset;
        self.dataset_summary = Some(summary);
        self.dataset_load_error = None;
        self.wizard_summary = None;
        Ok(())
    }

    fn selected_action(&self) -> MenuAction {
        self.items[self.selected]
    }

    fn next(&mut self) {
        self.selected = (self.selected + 1) % self.items.len();
    }

    fn previous(&mut self) {
        if self.selected == 0 {
            self.selected = self.items.len() - 1;
        } else {
            self.selected -= 1;
        }
    }

    fn enter_selected_view_or_exit(&mut self) -> bool {
        match self.selected_action() {
            MenuAction::TrainingWizard => {
                self.view = View::TrainingWizard;
                self.wizard_step = WizardStep::Dataset;
                self.wizard_summary = None;
                self.status_message = format!(
                    "Training wizard opened. Enter loads '{}' (SMILES='{}', target='{}', limit={}).",
                    self.config.dataset_csv_path,
                    self.config.dataset_smiles_column,
                    self.config.dataset_target_column,
                    self.config.dataset_row_limit
                );
            }
            MenuAction::BatchPrediction => {
                self.view = View::BatchPrediction;
                self.status_message =
                    "Batch prediction opened. Press Enter to run preview predictions.".to_string();
            }
            MenuAction::ChemicalSpace => {
                self.view = View::ChemicalSpace;
                self.status_message = "Chemical space summary opened.".to_string();
            }
            MenuAction::DescriptorProfiling => {
                self.view = View::DescriptorProfiling;
                self.status_message = "Descriptor profiling summary opened.".to_string();
            }
            MenuAction::PipelineManagement => {
                self.view = View::PipelineManagement;
                self.status_message = format!(
                    "Pipeline management opened. s=save file, l=load file, m=load memory ('{}').",
                    self.config.pipeline_path
                );
            }
            MenuAction::Exit => return true,
        }
        false
    }

    fn back_to_menu(&mut self) {
        self.view = View::Menu;
        self.status_message = "Returned to main menu.".to_string();
    }

    fn handle_enter(&mut self) -> bool {
        match self.view {
            View::Menu => self.enter_selected_view_or_exit(),
            View::TrainingWizard => {
                self.advance_wizard();
                false
            }
            View::BatchPrediction => {
                self.run_batch_prediction_preview();
                false
            }
            View::ChemicalSpace | View::DescriptorProfiling | View::PipelineManagement => false,
        }
    }

    fn advance_wizard(&mut self) {
        match self.wizard_step {
            WizardStep::Dataset => {
                match self.reload_dataset_from_csv() {
                    Ok(()) => {
                        self.wizard_step = WizardStep::Preprocess;
                        if let Some(summary) = &self.dataset_summary {
                            self.status_message = format!(
                                "{} Parse summary: parsed={}, failures={}, missing_target={}.",
                                summary.short_status(),
                                summary.parsed_rows,
                                summary.parse_failures,
                                summary.missing_targets
                            );
                        } else {
                            self.status_message = format!(
                                "Dataset loaded: {} samples × {} features.",
                                self.dataset.n_samples(),
                                self.dataset.n_features()
                            );
                        }
                    }
                    Err(error) => {
                        self.dataset_load_error = Some(error.clone());
                        self.status_message = format!(
                            "Dataset reload failed from '{}': {}",
                            self.config.dataset_csv_path, error
                        );
                    }
                }
            }
            WizardStep::Preprocess => {
                self.wizard_step = WizardStep::Train;
                self.status_message =
                    "Preprocessing configured (StandardScaler in AutoQsar).".to_string();
            }
            WizardStep::Train => {
                let start = std::time::Instant::now();
                let pipeline = AutoQsar::regression(&self.dataset, 3, 42);
                let elapsed = start.elapsed();
                let preds = pipeline.predict(&self.dataset.features);
                let preds_vec = preds.to_vec();

                // Build model candidates list from the pipeline's training run.
                // AutoQsar internally tries: kNN-3/5/7, RF-50/100, PLS-2/3/5, EN-0.01/0.1/1.0
                // We reconstruct a summary here by re-running quick predictions for each class.
                let best_name = pipeline.name.clone();
                let candidate_names = ["kNN-3", "kNN-5", "kNN-7", "RF-50", "RF-100",
                    "PLS-2", "PLS-3", "PLS-5", "EN-a0.01", "EN-a0.1", "EN-a1"];
                self.model_candidates.clear();
                // Just record the best model since AutoQsar doesn't expose intermediates
                self.model_candidates.push(ModelCandidate {
                    name: pipeline.name.clone(),
                    cv_q2: pipeline.cv_q2,
                    train_r2: pipeline.train_r2,
                    is_best: true,
                });

                let summary = TrainingSummary {
                    pipeline_name: pipeline.name.clone(),
                    cv_q2: pipeline.cv_q2,
                    train_r2: pipeline.train_r2,
                    preview_predictions: preds_vec
                        .iter()
                        .take(TRAINING_PREVIEW_ROWS)
                        .copied()
                        .collect(),
                };

                self.pipeline_bytes = match bincode::serialize(&pipeline) {
                    Ok(bytes) => Some(bytes),
                    Err(error) => {
                        self.status_message = format!(
                            "Training succeeded but serialization failed: {}",
                            error
                        );
                        None
                    }
                };
                self.pipeline = Some(pipeline);
                self.wizard_summary = Some(summary);
                self.wizard_step = WizardStep::Evaluate;
                self.status_message = format!(
                    "Training complete in {:.1}s — best model: '{}' (Q²={:.4}). Press Enter to finalize.",
                    elapsed.as_secs_f64(),
                    best_name,
                    self.wizard_summary.as_ref().unwrap().cv_q2
                );
            }
            WizardStep::Evaluate => {
                self.wizard_step = WizardStep::Done;
                self.status_message =
                    "Wizard complete. Model is now available to batch prediction and management."
                        .to_string();
            }
            WizardStep::Done => {
                self.status_message =
                    "Wizard already completed. Press Esc to return to menu.".to_string();
            }
        }
    }

    fn run_batch_prediction_preview(&mut self) {
        let Some(pipeline) = &self.pipeline else {
            self.status_message =
                "No trained pipeline found. Run Training Wizard first (menu item 1).".to_string();
            return;
        };

        let preds = pipeline.predict(&self.dataset.features);
        let r2 = metrics::r_squared(self.dataset.targets.as_slice().unwrap(), &preds.to_vec());
        let rmse = metrics::rmse(self.dataset.targets.as_slice().unwrap(), &preds.to_vec());

        self.status_message = format!(
            "Batch preview complete using '{}': R²={:.3}, RMSE={:.3} on {} samples.",
            pipeline.name,
            r2,
            rmse,
            self.dataset.n_samples()
        );
    }

    fn save_pipeline_to_file(&mut self) {
        let Some(pipeline) = &self.pipeline else {
            self.status_message = "No pipeline to save. Train one first.".to_string();
            return;
        };

        let bytes = match bincode::serialize(pipeline) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.status_message = format!("Failed to serialize pipeline: {}", error);
                return;
            }
        };

        match fs::write(&self.config.pipeline_path, &bytes) {
            Ok(()) => {
                self.pipeline_bytes = Some(bytes);
                self.status_message = format!(
                    "Pipeline '{}' saved to '{}' ({} bytes).",
                    pipeline.name,
                    self.config.pipeline_path,
                    self.pipeline_bytes.as_ref().map_or(0, Vec::len)
                );
            }
            Err(error) => {
                self.status_message = format!(
                    "Failed to save pipeline to '{}': {}",
                    self.config.pipeline_path, error
                );
            }
        }
    }

    fn load_pipeline_from_file(&mut self) {
        let bytes = match fs::read(&self.config.pipeline_path) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.status_message = format!(
                    "Failed to read pipeline file '{}': {}",
                    self.config.pipeline_path, error
                );
                return;
            }
        };

        let loaded = match bincode::deserialize::<QsarPipeline>(&bytes) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.status_message = format!(
                    "Failed to deserialize pipeline file '{}': {}",
                    self.config.pipeline_path, error
                );
                return;
            }
        };

        let loaded_name = loaded.name.clone();
        self.pipeline_bytes = Some(bytes);
        self.pipeline = Some(loaded);
        self.status_message = format!(
            "Loaded pipeline '{}' from '{}'.",
            loaded_name, self.config.pipeline_path
        );
    }

    fn load_pipeline_from_memory(&mut self) {
        let Some(bytes) = &self.pipeline_bytes else {
            self.status_message = "No serialized pipeline found in in-memory mirror.".to_string();
            return;
        };

        let loaded = match bincode::deserialize::<QsarPipeline>(bytes) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.status_message = format!("Failed to deserialize in-memory pipeline: {}", error);
                return;
            }
        };
        let loaded_name = loaded.name.clone();
        self.pipeline = Some(loaded);
        self.status_message = format!("Loaded pipeline '{}' from in-memory mirror.", loaded_name);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = setup_terminal()?;
    let result = run_app(&mut terminal);
    restore_terminal(&mut terminal)?;
    result?;
    Ok(())
}

fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let mut app = App::new();

    loop {
        terminal.draw(|frame| render(frame, &app))?;

        if !event::poll(POLL_INTERVAL)? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };

        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Char('q') => return Ok(()),
            KeyCode::Esc => {
                if app.view == View::Menu {
                    return Ok(());
                }
                app.back_to_menu();
            }
            KeyCode::Up | KeyCode::Char('k') if app.view == View::Menu => app.previous(),
            KeyCode::Down | KeyCode::Char('j') if app.view == View::Menu => app.next(),
            KeyCode::Enter => {
                if app.handle_enter() {
                    return Ok(());
                }
            }
            KeyCode::Char('s') if app.view == View::PipelineManagement => {
                app.save_pipeline_to_file();
            }
            KeyCode::Char('l') if app.view == View::PipelineManagement => {
                app.load_pipeline_from_file();
            }
            KeyCode::Char('m') if app.view == View::PipelineManagement => {
                app.load_pipeline_from_memory();
            }
            KeyCode::Up | KeyCode::Char('k') if app.view == View::DescriptorProfiling => {
                if app.scroll_offset > 0 {
                    app.scroll_offset -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') if app.view == View::DescriptorProfiling => {
                app.scroll_offset += 1;
            }
            KeyCode::Char('?') => {
                app.status_message = "Keys: q=quit, Esc=back, Enter=activate, ↑↓=scroll, s/l/m=pipeline ops, ?=help".to_string();
            }
            _ => {}
        }
    }
}

fn render(frame: &mut Frame, app: &App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(12),
        Constraint::Length(5),
    ])
    .areas(frame.area());

    let header_text = Paragraph::new("FerrumChem — Phase 14 TUI Workflow Shell")
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::ALL).title("FerrumChem"));
    frame.render_widget(header_text, header);

    match app.view {
        View::Menu => render_menu_view(frame, body, app),
        View::TrainingWizard => render_training_wizard(frame, body, app),
        View::BatchPrediction => render_batch_prediction(frame, body, app),
        View::ChemicalSpace => render_chemical_space(frame, body, app),
        View::DescriptorProfiling => render_descriptor_profile(frame, body, app),
        View::PipelineManagement => render_pipeline_management(frame, body, app),
    }

    let footer_text = Paragraph::new(format!(
        "Status: {}\nGlobal keys: q quit • Esc back/menu • Enter activate • (Pipeline view) s save-file, l load-file, m load-memory",
        app.status_message
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Status / Help"),
    );
    frame.render_widget(footer_text, footer);
}

fn render_menu_view(frame: &mut Frame, area: Rect, app: &App) {
    let [menu_area, details_area] =
        Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(area);

    let menu_items: Vec<ListItem> = app
        .items
        .iter()
        .map(|action| ListItem::new(action.title()))
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(app.selected));

    let list = List::new(menu_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Phase 14 Modules"),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");
    frame.render_stateful_widget(list, menu_area, &mut list_state);

    let details = match app.selected_action() {
        MenuAction::TrainingWizard => {
            "CSV-driven wizard flow:\n1) load/validate dataset from CSV\n2) preprocessing setup\n3) train via AutoQsar\n4) evaluate\n\nPress Enter to start."
        }
        MenuAction::BatchPrediction => {
            "Run batch predictions with the currently trained pipeline.\n\nPress Enter to open."
        }
        MenuAction::ChemicalSpace => {
            "Lightweight chemical-space projection summary from the currently loaded dataset."
        }
        MenuAction::DescriptorProfiling => {
            "Descriptor profile summary: feature means, ranges, and basic shape diagnostics."
        }
        MenuAction::PipelineManagement => {
            "File-backed persistence for serialized QsarPipeline bytes.\n\nInside this view: s=save file, l=load file, m=load memory mirror."
        }
        MenuAction::Exit => "Exit the TUI shell.",
    };

    let details_widget = Paragraph::new(details).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Selected module"),
    );
    frame.render_widget(details_widget, details_area);
}

fn render_training_wizard(frame: &mut Frame, area: Rect, app: &App) {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(area);

    let steps = [
        WizardStep::Dataset,
        WizardStep::Preprocess,
        WizardStep::Train,
        WizardStep::Evaluate,
        WizardStep::Done,
    ];
    let step_lines: Vec<String> = steps
        .iter()
        .map(|step| {
            if *step == app.wizard_step {
                format!(">> {}", step.title())
            } else {
                format!("   {}", step.title())
            }
        })
        .collect();

    let step_widget = Paragraph::new(step_lines.join("\n"))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Training Wizard Steps"),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(step_widget, left);

    let mut details = format!(
        "Current dataset: {} samples × {} features\nTargets: {} values\nCSV path: {}\nSMILES column: {}\nTarget column: {}\nRow limit: {}\n\nPress Enter to advance.",
        app.dataset.n_samples(),
        app.dataset.n_features(),
        app.dataset.targets.len(),
        app.config.dataset_csv_path,
        app.config.dataset_smiles_column,
        app.config.dataset_target_column,
        app.config.dataset_row_limit
    );

    if let Some(dataset_summary) = &app.dataset_summary {
        details.push_str(&format!(
            "\n\nLast CSV load summary:\n- requested rows: {}\n- parsed rows: {}\n- parse failures: {}\n- missing targets: {}\n- loaded rows: {}\n- feature count: {}\n- smiles column: {}\n- target range: [{:.3}, {:.3}] mean={:.3}\n- source: {}",
            dataset_summary.requested_rows,
            dataset_summary.parsed_rows,
            dataset_summary.parse_failures,
            dataset_summary.missing_targets,
            dataset_summary.loaded_rows,
            dataset_summary.feature_count,
            dataset_summary.smiles_column,
            dataset_summary.target_min,
            dataset_summary.target_max,
            dataset_summary.target_mean,
            dataset_summary.csv_path
        ));
    }

    if let Some(error) = &app.dataset_load_error {
        details.push_str(&format!(
            "\n\nDataset load warning:\n{}\n(Currently using fallback mock dataset until reload succeeds.)",
            error
        ));
    }

    if let Some(summary) = &app.wizard_summary {
        let preview: Vec<String> = summary
            .preview_predictions
            .iter()
            .enumerate()
            .map(|(i, v)| format!("  #{:02}: {:.4}", i + 1, v))
            .collect();
        details.push_str(&format!(
            "\n\nLatest training summary:\n- pipeline: {}\n- cv_q2: {:.4}\n- train_r2: {:.4}\n- preview predictions:\n{}",
            summary.pipeline_name,
            summary.cv_q2,
            summary.train_r2,
            preview.join("\n")
        ));
    }

    let detail_widget = Paragraph::new(details)
        .block(Block::default().borders(Borders::ALL).title("Wizard state"))
        .wrap(Wrap { trim: true });
    frame.render_widget(detail_widget, right);
}

fn render_batch_prediction(frame: &mut Frame, area: Rect, app: &App) {
    let message = if let Some(pipeline) = &app.pipeline {
        let preds = pipeline.predict(&app.dataset.features);
        let preview_lines: Vec<String> = preds
            .iter()
            .take(TRAINING_PREVIEW_ROWS)
            .enumerate()
            .map(|(i, p)| format!("  sample #{:02}: {:.4}", i + 1, p))
            .collect();
        format!(
            "Pipeline loaded: {}\nDataset rows: {}\n\nPress Enter to recompute batch preview.\nCurrent preview:\n{}",
            pipeline.name,
            app.dataset.n_samples(),
            preview_lines.join("\n")
        )
    } else {
        "No trained pipeline available.\nRun the Training Wizard first.".to_string()
    };

    let widget = Paragraph::new(message)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Batch prediction"),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(widget, area);
}

fn render_chemical_space(frame: &mut Frame, area: Rect, app: &App) {
    if app.dataset.n_features() < 2 || app.dataset.n_samples() < 3 {
        let widget = Paragraph::new(
            "Chemical-space scatter requires at least 3 samples and 2 features.\nReload a dataset with more data.",
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Chemical space"),
        )
        .wrap(Wrap { trim: true });
        frame.render_widget(widget, area);
        return;
    }

    // PCA(2) projection: scale → project → scatter
    let scaler = StandardScaler::fit(&app.dataset.features);
    let scaled = scaler.transform(&app.dataset.features);
    let (pca, projected) = Pca::fit_transform(&scaled, 2);

    let pc1 = projected.column(0);
    let pc2 = projected.column(1);
    let x_min = pc1.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = pc1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = pc2.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = pc2.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Add a small padding so edge points are visible
    let x_pad = (x_max - x_min).max(0.01) * 0.05;
    let y_pad = (y_max - y_min).max(0.01) * 0.05;

    let [chart_area, info_area] =
        Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(area);

    // Collect points for the braille canvas
    let points: Vec<(f64, f64)> = pc1.iter().zip(pc2.iter()).map(|(&x, &y)| (x, y)).collect();

    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("PCA Scatter (PC1 vs PC2)"),
        )
        .x_bounds([x_min - x_pad, x_max + x_pad])
        .y_bounds([y_min - y_pad, y_max + y_pad])
        .paint(move |ctx| {
            ctx.draw(&Points {
                coords: &points,
                color: Color::Cyan,
            });
        });
    frame.render_widget(canvas, chart_area);

    // Info panel with PCA summary
    let var_ratio_1 = pca.explained_variance_ratio[0] * 100.0;
    let var_ratio_2 = pca.explained_variance_ratio[1] * 100.0;
    let total_var = var_ratio_1 + var_ratio_2;

    let info = format!(
        "PCA Chemical Space Summary\n\n\
         Samples: {}\n\
         Features: {}\n\n\
         PC1 variance: {:.1}%\n\
         PC2 variance: {:.1}%\n\
         Total explained: {:.1}%\n\n\
         PC1 range: [{:.2}, {:.2}]\n\
         PC2 range: [{:.2}, {:.2}]\n\n\
         Projection uses StandardScaler\n\
         + Jacobi PCA(2).",
        app.dataset.n_samples(),
        app.dataset.n_features(),
        var_ratio_1,
        var_ratio_2,
        total_var,
        x_min,
        x_max,
        y_min,
        y_max,
    );

    let info_widget = Paragraph::new(info)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("PCA Info"),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(info_widget, info_area);
}

fn render_descriptor_profile(frame: &mut Frame, area: Rect, app: &App) {
    let feature_count = app.dataset.n_features();
    let n = app.dataset.n_samples() as f64;

    // Build rows with stats: name, mean, std_dev, min, max, [flag]
    let mut table_rows: Vec<Row> = Vec::with_capacity(feature_count);
    let mut zero_var_count = 0usize;

    for j in 0..feature_count {
        let col = app.dataset.features.column(j);
        let mean = col.iter().sum::<f64>() / n;
        let min_val = col.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let flag = if variance < 1e-12 {
            zero_var_count += 1;
            " ⚠"
        } else {
            ""
        };

        let style = if variance < 1e-12 {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
        };

        let row = Row::new(vec![
            format!("{}", app.dataset.feature_names[j]),
            format!("{:.4}", mean),
            format!("{:.4}", std_dev),
            format!("{:.4}", min_val),
            format!("{:.4}", max_val),
            flag.to_string(),
        ])
        .style(style);
        table_rows.push(row);
    }

    // Apply scroll offset — clamp to valid range
    let visible_rows = (area.height as usize).saturating_sub(6); // header + borders
    let max_offset = feature_count.saturating_sub(visible_rows);
    let offset = app.scroll_offset.min(max_offset);
    let visible: Vec<Row> = table_rows.into_iter().skip(offset).collect();

    let header = Row::new(vec!["Feature", "Mean", "StdDev", "Min", "Max", "Flag"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    let title = format!(
        "Descriptor Profile ({} features, {} zero-var, showing {}-{}/{}  ↑↓=scroll)",
        feature_count,
        zero_var_count,
        offset + 1,
        (offset + visible_rows).min(feature_count),
        feature_count
    );

    let table = Table::new(
        visible,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(14),
            Constraint::Percentage(14),
            Constraint::Percentage(14),
            Constraint::Percentage(14),
            Constraint::Percentage(8),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(title));

    frame.render_widget(table, area);
}

fn render_pipeline_management(frame: &mut Frame, area: Rect, app: &App) {
    let loaded = app
        .pipeline
        .as_ref()
        .map(|p| p.name.as_str())
        .unwrap_or("none");
    let bytes = app.pipeline_bytes.as_ref().map_or(0, Vec::len);

    let text = format!(
        "Pipeline management\n\nLoaded pipeline: {}\nIn-memory mirror bytes: {}\nPipeline file path: {}\n\nCommands in this view:\n- s: serialize current pipeline and write file\n- l: read pipeline file and load it\n- m: reload from in-memory mirror only\n\nThis validates file-backed persistence workflows for Phase 14.",
        loaded, bytes, app.config.pipeline_path
    );

    let widget = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Pipeline management"),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(widget, area);

    if app.pipeline.is_none() && app.pipeline_bytes.is_none() {
        render_notice_popup(
            frame,
            "No pipeline state yet.\nRun Training Wizard first, then return here.",
            Color::Yellow,
        );
    }
}

fn render_notice_popup(frame: &mut Frame, message: &str, color: Color) {
    let popup_area = centered_rect(66, 28, frame.area());
    frame.render_widget(Clear, popup_area);
    let popup = Paragraph::new(message).wrap(Wrap { trim: true }).block(
        Block::default()
            .title("Notice")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(color)),
    );
    frame.render_widget(popup, popup_area);
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);

    let horizontal = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(vertical[1]);

    horizontal[1]
}

fn build_mock_dataset() -> QsarDataset {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut targets = Vec::new();

    for i in 0..48usize {
        let x0 = i as f64 / 10.0;
        let x1 = ((i * 7) % 13) as f64 / 3.0;
        let x2 = ((i * 5 + 3) % 17) as f64 / 4.0;
        let x3 = (i % 9) as f64;
        let x4 = (x0 * 0.7 + x1 * 0.4 - x2 * 0.2 + x3 * 0.05).sin();
        let x5 = (x0 * 1.3 + x2 * 0.3).cos();

        let y = 1.2 * x0 + 0.8 * x1 - 0.5 * x2 + 0.2 * x3 + 0.4 * x4 + 0.1 * x5;

        rows.push(vec![
            x0 as f32, x1 as f32, x2 as f32, x3 as f32, x4 as f32, x5 as f32,
        ]);
        targets.push(y);
    }

    let feature_names = vec![
        "feat_mw_like".to_string(),
        "feat_logp_like".to_string(),
        "feat_tpsa_like".to_string(),
        "feat_ring_like".to_string(),
        "feat_charge_like".to_string(),
        "feat_shape_like".to_string(),
    ];

    QsarDataset::from_f32_vectors(&rows, targets, feature_names)
}
