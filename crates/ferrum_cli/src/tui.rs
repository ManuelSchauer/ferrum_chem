use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::desc::table::{format_csv_rows_par, write_csv_header, DescriptorTable, FingerprintFormat};
use ferrum_core::io::stream::MoleculeReader;
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, ListState, Paragraph, Row, Table, Tabs, Wrap,
};
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const SPINNER_FRAMES: [char; 4] = ['|', '/', '-', '\\'];
const PREVIEW_ROWS: usize = 10;
const PREVIEW_COLS: usize = 6;
const DEFAULT_EXPORT_PATH: &str = "/tmp/ferrum_descriptors.csv";

/// Records shown in the dataset preview.
const DATASET_PREVIEW: usize = 15;

/// `(identifier, heavy atoms)` of the first records of a dataset.
type Preview = Vec<(String, String)>;

/// The full results of the last calculation (streamed to disk, so memory does
/// not grow with the dataset); export copies this file.
fn results_path() -> PathBuf {
    std::env::temp_dir().join(format!("ferrum_tui_{}.csv", std::process::id()))
}

enum WorkerRequest {
    /// Count the records of a file (streamed) and keep a short preview.
    LoadDataset(PathBuf),
    /// Stream the file through a calculator into the results file.
    CalculateDescriptors {
        path: PathBuf,
        descriptors: Vec<BatchDescriptor>,
        total: usize,
    },
    ExportMatrix {
        path: PathBuf,
    },
    Shutdown,
}

#[allow(dead_code)]
enum WorkerResponse {
    Progress(f32),
    DatasetLoaded {
        path: PathBuf,
        records: usize,
        failed_rows: usize,
        preview: Vec<(String, String)>,
    },
    DescriptorsCalculated {
        /// The first chunk of results (for the preview).
        matrix: DescriptorTable,
        feature_names: Vec<String>,
        rows: usize,
    },
    MatrixExported(PathBuf),
    Error(String),
    Canceled,
}

#[derive(PartialEq, Eq)]
enum PopupMode {
    None,
    FileBrowser,
    ExportPrompt,
    Computing,
    Error(String),
}

struct App {
    tab_index: usize,
    popup: PopupMode,
    status_message: String,
    spinner_idx: usize,

    // File browser
    browser_path: PathBuf,
    browser_entries: Vec<PathBuf>,
    browser_selected: usize,

    // Export input
    export_path_input: String,

    // Loaded dataset
    dataset_path: Option<PathBuf>,
    records: usize,
    failed_rows: usize,
    /// `(identifier, heavy atoms)` of the first records.
    preview: Vec<(String, String)>,

    // Descriptor selection
    descriptors: Vec<(BatchDescriptor, bool)>,
    desc_list_state: ListState,

    // Calculated results (preview of the first chunk; all rows are in `results_path()`)
    matrix: Option<DescriptorTable>,
    /// Scalar column names of `matrix` (fingerprints are not previewed).
    feature_names: Vec<String>,
    /// Number of result rows written to the results file.
    result_rows: usize,
    result_row_offset: usize,
    result_col_offset: usize,

    // Worker comms
    req_tx: Sender<WorkerRequest>,
    resp_rx: Receiver<WorkerResponse>,
    cancel_flag: Arc<AtomicBool>,
}

impl App {
    fn new(req_tx: Sender<WorkerRequest>, resp_rx: Receiver<WorkerResponse>, cancel_flag: Arc<AtomicBool>) -> Self {
        let all = BatchDescriptor::all();
        let descriptors = all.iter().map(|&d| (d, true)).collect();
        let mut list_state = ListState::default();
        list_state.select(Some(0));

        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut app = Self {
            tab_index: 0,
            popup: PopupMode::None,
            status_message: "Welcome to FerrumChem TUI. Press 'f' to browse and load a file.".to_string(),
            spinner_idx: 0,

            browser_path: current_dir.clone(),
            browser_entries: Vec::new(),
            browser_selected: 0,

            export_path_input: DEFAULT_EXPORT_PATH.to_string(),

            dataset_path: None,
            records: 0,
            failed_rows: 0,
            preview: Vec::new(),

            descriptors,
            desc_list_state: list_state,

            matrix: None,
            feature_names: Vec::new(),
            result_rows: 0,
            result_row_offset: 0,
            result_col_offset: 0,

            req_tx,
            resp_rx,
            cancel_flag,
        };
        app.refresh_browser_entries();
        app
    }

    fn refresh_browser_entries(&mut self) {
        self.browser_entries.clear();
        self.browser_selected = 0;
        if let Some(parent) = self.browser_path.parent() {
            self.browser_entries.push(parent.to_path_buf());
        }
        if let Ok(entries) = fs::read_dir(&self.browser_path) {
            let mut list: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    if p.is_dir() {
                        let name = p.file_name().unwrap_or_default().to_string_lossy();
                        !name.starts_with('.') && name != "target"
                    } else {
                        let ext = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
                        matches!(ext.as_str(), "smi" | "smiles" | "csv" | "sdf")
                    }
                })
                .collect();
            list.sort_by(|a, b| {
                let a_dir = a.is_dir();
                let b_dir = b.is_dir();
                if a_dir && !b_dir {
                    std::cmp::Ordering::Less
                } else if !a_dir && b_dir {
                    std::cmp::Ordering::Greater
                } else {
                    a.cmp(b)
                }
            });
            self.browser_entries.extend(list);
        }
    }

    fn selected_descriptors(&self) -> Vec<BatchDescriptor> {
        self.descriptors
            .iter()
            .filter_map(|(d, on)| if *on { Some(*d) } else { None })
            .collect()
    }

    fn handle_tick(&mut self) {
        self.spinner_idx = (self.spinner_idx + 1) % SPINNER_FRAMES.len();
        while let Ok(resp) = self.resp_rx.try_recv() {
            match resp {
                WorkerResponse::Progress(p) => {
                    if p > 0.0 && matches!(self.popup, PopupMode::Computing) {
                        self.status_message = format!("Calculating descriptors... {:.0}%", (p * 100.0).min(100.0));
                    }
                }
                WorkerResponse::DatasetLoaded { path, records, failed_rows, preview } => {
                    self.dataset_path = Some(path);
                    self.records = records;
                    self.failed_rows = failed_rows;
                    self.preview = preview;
                    self.popup = PopupMode::None;
                    self.matrix = None;
                    self.status_message = format!(
                        "Found {} records ({} cannot be parsed). Ready to calculate descriptors.",
                        records, failed_rows
                    );
                }
                WorkerResponse::DescriptorsCalculated { matrix, feature_names, rows } => {
                    let r = rows;
                    let c = matrix.n_scalars() + matrix.fingerprints.len();
                    self.matrix = Some(matrix);
                    self.feature_names = feature_names;
                    self.result_rows = rows;
                    self.popup = PopupMode::None;
                    self.tab_index = 1;
                    self.status_message = format!("Successfully calculated descriptors ({} molecules x {} features). Press 'e' to export.", r, c);
                }
                WorkerResponse::MatrixExported(path) => {
                    self.popup = PopupMode::None;
                    self.status_message = format!("Exported feature matrix to '{}'", path.display());
                }
                WorkerResponse::Error(err) => {
                    self.popup = PopupMode::Error(err);
                }
                WorkerResponse::Canceled => {
                    self.popup = PopupMode::None;
                    self.status_message = "Operation canceled by user.".to_string();
                }
            }
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }

        match &self.popup {
            PopupMode::Error(_) => {
                if key.code == KeyCode::Esc || key.code == KeyCode::Enter {
                    self.popup = PopupMode::None;
                }
                return false;
            }
            PopupMode::Computing => {
                if key.code == KeyCode::Esc {
                    self.cancel_flag.store(true, Ordering::Relaxed);
                    self.status_message = "Canceling calculation...".to_string();
                }
                return false;
            }
            PopupMode::FileBrowser => {
                match key.code {
                    KeyCode::Esc => self.popup = PopupMode::None,
                    KeyCode::Up => {
                        if self.browser_selected > 0 {
                            self.browser_selected -= 1;
                        }
                    }
                    KeyCode::Down => {
                        if !self.browser_entries.is_empty() && self.browser_selected + 1 < self.browser_entries.len() {
                            self.browser_selected += 1;
                        }
                    }
                    KeyCode::Enter => {
                        if let Some(target) = self.browser_entries.get(self.browser_selected).cloned() {
                            if target.is_dir() {
                                self.browser_path = target;
                                self.refresh_browser_entries();
                            } else {
                                self.popup = PopupMode::Computing;
                                self.status_message = format!("Loading '{}'...", target.display());
                                let _ = self.req_tx.send(WorkerRequest::LoadDataset(target));
                            }
                        }
                    }
                    _ => {}
                }
                return false;
            }
            PopupMode::ExportPrompt => {
                match key.code {
                    KeyCode::Esc => self.popup = PopupMode::None,
                    KeyCode::Backspace => {
                        self.export_path_input.pop();
                    }
                    KeyCode::Char(c) => {
                        self.export_path_input.push(c);
                    }
                    KeyCode::Enter
                        if self.matrix.is_some() => {
                            let path = PathBuf::from(&self.export_path_input);
                            self.popup = PopupMode::Computing;
                            self.status_message = format!("Exporting to '{}'...", path.display());
                            let _ = self.req_tx.send(WorkerRequest::ExportMatrix { path });
                        }
                    _ => {}
                }
                return false;
            }
            PopupMode::None => {}
        }

        // Global keys
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Tab => {
                self.tab_index = (self.tab_index + 1) % 2;
                return false;
            }
            KeyCode::Char('1') => {
                self.tab_index = 0;
                return false;
            }
            KeyCode::Char('2') => {
                self.tab_index = 1;
                return false;
            }
            KeyCode::Char('f') => {
                self.popup = PopupMode::FileBrowser;
                self.refresh_browser_entries();
                return false;
            }
            _ => {}
        }

        if self.tab_index == 0 {
            // Import tab keys
            if key.code == KeyCode::Char('l') || key.code == KeyCode::Enter {
                self.popup = PopupMode::FileBrowser;
                self.refresh_browser_entries();
            }
        } else if self.tab_index == 1 {
            // Descriptors tab keys
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    let sel = self.desc_list_state.selected().unwrap_or(0);
                    if sel > 0 {
                        self.desc_list_state.select(Some(sel - 1));
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let sel = self.desc_list_state.selected().unwrap_or(0);
                    if sel + 1 < self.descriptors.len() {
                        self.desc_list_state.select(Some(sel + 1));
                    }
                }
                KeyCode::Char(' ') => {
                    if let Some(sel) = self.desc_list_state.selected()
                        && let Some((_, state)) = self.descriptors.get_mut(sel) {
                            *state = !*state;
                        }
                }
                KeyCode::Char('a') => {
                    for (_, state) in self.descriptors.iter_mut() {
                        *state = true;
                    }
                    self.status_message = "Selected all descriptors.".to_string();
                }
                KeyCode::Char('n') => {
                    for (_, state) in self.descriptors.iter_mut() {
                        *state = false;
                    }
                    self.status_message = "Cleared descriptor selection.".to_string();
                }
                KeyCode::Char('r') => {
                    let preset = BatchDescriptor::fast_ml_preset();
                    for (d, state) in self.descriptors.iter_mut() {
                        *state = preset.contains(d);
                    }
                    self.status_message = "Selected Fast ML preset descriptors.".to_string();
                }
                KeyCode::Enter | KeyCode::Char('c') => {
                    let selected = self.selected_descriptors();
                    if self.records == 0 || self.dataset_path.is_none() {
                        self.popup = PopupMode::Error("Please load a dataset first (Tab 1 or press 'f').".to_string());
                    } else if selected.is_empty() {
                        self.popup = PopupMode::Error("No descriptors selected. Press 'a' or toggle with Space.".to_string());
                    } else {
                        self.popup = PopupMode::Computing;
                        self.cancel_flag.store(false, Ordering::Relaxed);
                        self.status_message = format!("Calculating {} descriptor families on {} records...", selected.len(), self.records);
                        let _ = self.req_tx.send(WorkerRequest::CalculateDescriptors {
                            path: self.dataset_path.clone().unwrap_or_default(),
                            descriptors: selected,
                            total: self.records,
                        });
                    }
                }
                KeyCode::Char('e') => {
                    if self.matrix.is_some() {
                        self.popup = PopupMode::ExportPrompt;
                    } else {
                        self.popup = PopupMode::Error("No calculated descriptors yet. Calculate descriptors first (press Enter).".to_string());
                    }
                }
                KeyCode::Right | KeyCode::Char('l') => {
                    self.result_col_offset = self.result_col_offset.saturating_add(1);
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    self.result_col_offset = self.result_col_offset.saturating_sub(1);
                }
                KeyCode::Char('d') => {
                    self.result_row_offset = self.result_row_offset.saturating_add(5);
                }
                KeyCode::Char('u') => {
                    self.result_row_offset = self.result_row_offset.saturating_sub(5);
                }
                _ => {}
            }
        }

        false
    }
}

pub fn run_tui() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (req_tx, req_rx) = mpsc::channel::<WorkerRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<WorkerResponse>();
    let cancel_flag = Arc::new(AtomicBool::new(false));

    let worker_cancel = Arc::clone(&cancel_flag);
    let _worker_handle = thread::spawn(move || {
        worker_thread(req_rx, resp_tx, worker_cancel);
    });

    let mut app = App::new(req_tx.clone(), resp_rx, cancel_flag);

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
                && app.handle_key(key) {
                    break;
                }
        app.handle_tick();
    }

    let _ = req_tx.send(WorkerRequest::Shutdown);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn worker_thread(
    req_rx: Receiver<WorkerRequest>,
    resp_tx: Sender<WorkerResponse>,
    cancel_flag: Arc<AtomicBool>,
) {
    while let Ok(req) = req_rx.recv() {
        match req {
            WorkerRequest::Shutdown => {
                let _ = fs::remove_file(results_path());
                break;
            }
            WorkerRequest::LoadDataset(path) => {
                let _ = resp_tx.send(WorkerResponse::Progress(0.1));
                match count_records(&path) {
                    Ok((records, failed_rows, preview)) => {
                        let _ = resp_tx.send(WorkerResponse::DatasetLoaded { path, records, failed_rows, preview });
                    }
                    Err(e) => {
                        let _ = resp_tx.send(WorkerResponse::Error(e));
                    }
                }
            }
            WorkerRequest::CalculateDescriptors { path, descriptors, total } => {
                let _ = resp_tx.send(WorkerResponse::Progress(0.0));
                let flag = Arc::clone(&cancel_flag);
                match calculate_to_file(&path, &descriptors, total, &resp_tx, &move || flag.load(Ordering::Relaxed)) {
                    Ok(Some((matrix, rows))) => {
                        let feature_names = matrix.scalar_names.clone();
                        let _ = resp_tx.send(WorkerResponse::DescriptorsCalculated { matrix, feature_names, rows });
                    }
                    Ok(None) => {
                        let _ = resp_tx.send(WorkerResponse::Canceled);
                    }
                    Err(e) => {
                        let _ = resp_tx.send(WorkerResponse::Error(e));
                    }
                }
            }
            WorkerRequest::ExportMatrix { path } => match fs::copy(results_path(), &path) {
                Ok(_) => {
                    let _ = resp_tx.send(WorkerResponse::MatrixExported(path));
                }
                Err(e) => {
                    let _ = resp_tx.send(WorkerResponse::Error(format!("Failed to write CSV: {}", e)));
                }
            },
        }
    }
}

/// Streams a dataset once: number of records, unparseable records and a preview.
fn count_records(path: &std::path::Path) -> Result<(usize, usize, Preview), String> {
    let mut reader = MoleculeReader::open(path).map_err(|e| format!("Failed to open '{}': {}", path.display(), e))?;
    let (mut records, mut failed, mut preview) = (0usize, 0usize, Vec::new());
    while let Some(chunk) = reader.next_chunk().map_err(|e| format!("Failed to read '{}': {}", path.display(), e))? {
        for record in chunk {
            records += 1;
            let atoms = match &record.molecule {
                Ok(m) => m.atoms.iter().filter(|a| a.atomic_number > 1).count().to_string(),
                Err(_) => {
                    failed += 1;
                    "parse error".to_string()
                }
            };
            if preview.len() < DATASET_PREVIEW {
                preview.push((record.id, atoms));
            }
        }
    }
    Ok((records, failed, preview))
}

/// Streams `path` through one calculator into `results_path()`. Returns the
/// first chunk (preview) and the number of rows, or `None` when cancelled.
fn calculate_to_file(
    path: &std::path::Path,
    descriptors: &[BatchDescriptor],
    total: usize,
    resp_tx: &Sender<WorkerResponse>,
    should_cancel: &(dyn Fn() -> bool + Sync),
) -> Result<Option<(DescriptorTable, usize)>, String> {
    let calc = DescriptorCalculator::new(descriptors);
    let mut reader = MoleculeReader::open(path).map_err(|e| format!("Failed to open '{}': {}", path.display(), e))?;
    let out = results_path();
    let mut writer = BufWriter::new(File::create(&out).map_err(|e| format!("Failed to create '{}': {}", out.display(), e))?);
    let format = FingerprintFormat::Hex;
    let write_err = |e: io::Error| format!("Failed to write '{}': {}", out.display(), e);
    write_csv_header(&mut writer, reader.id_column(), calc.layout(), format).map_err(write_err)?;
    let (mut first, mut rows) = (None, 0usize);
    while let Some(chunk) = reader.next_chunk().map_err(|e| format!("Failed to read '{}': {}", path.display(), e))? {
        let (ids, molecules): (Vec<_>, Vec<Option<_>>) = chunk.into_iter().map(|r| (r.id, r.molecule.ok())).unzip();
        let Ok(table) = calc.calculate_cancellable(&molecules, &should_cancel) else {
            return Ok(None);
        };
        io::Write::write_all(&mut writer, &format_csv_rows_par(&ids, &table, format)).map_err(write_err)?;
        rows += table.rows;
        let _ = resp_tx.send(WorkerResponse::Progress(rows as f32 / total.max(1) as f32));
        if first.is_none() {
            first = Some(table);
        }
    }
    io::Write::flush(&mut writer).map_err(write_err)?;
    Ok(first.map(|t| (t, rows)))
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(f.area());

    // Header & Tabs
    let tab_titles = vec!["1: Import", "2: Descriptors & Calculate"];
    let tabs = Tabs::new(tab_titles)
        .block(Block::default().borders(Borders::ALL).title("FerrumChem - Descriptor Engine"))
        .select(app.tab_index)
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, chunks[0]);

    // Body
    match app.tab_index {
        0 => render_import_tab(f, chunks[1], app),
        1 => render_descriptor_tab(f, chunks[1], app),
        _ => {}
    }

    // Footer
    let footer_text = format!(
        " [q] Quit | [Tab] Switch Tab | [f] File Browser | Status: {}",
        app.status_message
    );
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::ALL).title("Controls & Status"))
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(footer, chunks[2]);

    // Popups
    match &app.popup {
        PopupMode::None => {}
        PopupMode::Computing => {
            render_popup(f, "Processing...", |f, area| {
                let text = format!("Please wait... Working in background {}", SPINNER_FRAMES[app.spinner_idx]);
                let p = Paragraph::new(text).alignment(Alignment::Center);
                f.render_widget(p, area);
            }, 50, 20);
        }
        PopupMode::Error(err) => {
            render_popup(f, "Error", |f, area| {
                let p = Paragraph::new(format!("{}\n\nPress [Enter] or [Esc] to dismiss", err))
                    .style(Style::default().fg(Color::Red))
                    .wrap(Wrap { trim: true });
                f.render_widget(p, area);
            }, 60, 30);
        }
        PopupMode::FileBrowser => {
            render_popup(f, &format!("Browse: {}", app.browser_path.display()), |f, area| {
                let items: Vec<ListItem> = app
                    .browser_entries
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "..".to_string());
                        let display = if p.is_dir() { format!("📁 {}/", name) } else { format!("📄 {}", name) };
                        let style = if i == app.browser_selected {
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        };
                        ListItem::new(display).style(style)
                    })
                    .collect();
                let list = List::new(items).block(Block::default().borders(Borders::NONE));
                f.render_widget(list, area);
            }, 70, 70);
        }
        PopupMode::ExportPrompt => {
            render_popup(f, "Export Descriptors to CSV", |f, area| {
                let text = format!("File Path:\n{}\n\n[Enter] Export | [Esc] Cancel", app.export_path_input);
                let p = Paragraph::new(text).wrap(Wrap { trim: true });
                f.render_widget(p, area);
            }, 60, 25);
        }
    }
}

fn render_import_tab(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Min(5),
        ])
        .split(area);

    let source = app
        .dataset_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(no file loaded yet)".to_string());
    let info = Paragraph::new(format!(
        "Active File: {}\nPress [f] to open the file browser and load a .smi, .csv, or .sdf dataset.",
        source
    ))
    .block(Block::default().borders(Borders::ALL).title("Data Source"));
    f.render_widget(info, chunks[0]);

    let summary = Paragraph::new(format!(
        "Records:          {}\nCannot parse:     {}\nStatus:           {}",
        app.records,
        app.failed_rows,
        if app.records == 0 { "Empty" } else { "Ready" }
    ))
    .block(Block::default().borders(Borders::ALL).title("Dataset Summary"));
    f.render_widget(summary, chunks[1]);

    let rows: Vec<Row> = app
        .preview
        .iter()
        .enumerate()
        .map(|(i, (id, atoms))| Row::new(vec![format!("#{}", i + 1), id.clone(), atoms.clone()]))
        .collect();

    let header = Row::new(vec!["Index", "SMILES", "Heavy Atoms"])
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Percentage(70),
            Constraint::Length(14),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Molecules Preview (First 15)"));
    f.render_widget(table, chunks[2]);
}

fn render_descriptor_tab(f: &mut Frame, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    // Left panel: list of descriptors with checkboxes
    let items: Vec<ListItem> = app
        .descriptors
        .iter()
        .map(|(d, on)| {
            let check = if *on { "[X]" } else { "[ ]" };
            ListItem::new(format!("{} {} ({} feat)", check, d.name(), d.feature_len()))
        })
        .collect();

    let total_feats: usize = app.descriptors.iter().filter(|(_, on)| *on).map(|(d, _)| d.feature_len()).sum();
    let left_title = format!("Descriptors ({}/{} on, {} total features)", app.selected_descriptors().len(), app.descriptors.len(), total_feats);
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(left_title))
        .highlight_symbol("▶ ")
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_stateful_widget(list, chunks[0], &mut app.desc_list_state);

    // Right panel: Actions summary & Matrix preview
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(8)])
        .split(chunks[1]);

    let actions_text = "Keyboard Controls:\n  [Space] Toggle descriptor  |  [a] Select All  |  [n] Deselect All\n  [r] Fast ML Preset         |  [Enter/c] Calculate\n  [e] Export CSV             |  [h/l] Scroll cols  |  [u/d] Scroll rows";
    let actions = Paragraph::new(actions_text)
        .block(Block::default().borders(Borders::ALL).title("Actions"));
    f.render_widget(actions, right_chunks[0]);

    if let Some(matrix) = &app.matrix {
        let max_r = matrix.rows.min(PREVIEW_ROWS);
        let n_cols = matrix.n_scalars();
        let max_c = n_cols.min(PREVIEW_COLS);

        let row_start = app.result_row_offset.min(matrix.rows.saturating_sub(max_r));
        let col_start = app.result_col_offset.min(n_cols.saturating_sub(max_c));

        let mut header_cells = vec!["Row".to_string()];
        for c in col_start..(col_start + max_c).min(n_cols) {
            let name = app.feature_names.get(c).map(|s| s.as_str()).unwrap_or("?");
            header_cells.push(if name.len() > 14 { format!("{}...", &name[..11]) } else { name.to_string() });
        }

        let header = Row::new(header_cells)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));

        let mut rows = Vec::new();
        for r in row_start..(row_start + max_r).min(matrix.rows) {
            let mut cells = vec![format!("#{}", r + 1)];
            for c in col_start..(col_start + max_c).min(n_cols) {
                let val = matrix.scalars.get(r * n_cols + c).copied().unwrap_or(f64::NAN);
                cells.push(format!("{:.3}", val));
            }
            rows.push(Row::new(cells));
        }

        let mut constraints = vec![Constraint::Length(7)];
        for _ in 0..max_c {
            constraints.push(Constraint::Min(12));
        }

        let title = format!(
            "Preview (rows {}..{} of the first {}; {} rows in total, all exported; cols {}..{}/{})",
            row_start,
            row_start + max_r,
            matrix.rows,
            app.result_rows,
            col_start,
            col_start + max_c,
            n_cols
        );
        let table = Table::new(rows, constraints)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title(title));
        f.render_widget(table, right_chunks[1]);
    } else {
        let placeholder = Paragraph::new("\n\n  No descriptors calculated yet.\n  Press [Enter] or [c] to start calculation.")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title("Calculated Matrix Preview"));
        f.render_widget(placeholder, right_chunks[1]);
    }
}

fn render_popup<F>(f: &mut Frame, title: &str, render_inner: F, pct_x: u16, pct_y: u16)
where
    F: FnOnce(&mut Frame, Rect),
{
    let area = centered_rect(pct_x, pct_y, f.area());
    f.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .style(Style::default().bg(Color::Black));
    let inner = block.inner(area);
    f.render_widget(block, area);
    render_inner(f, inner);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smi_file(content: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("ferrum_tui_test_{}_{}.smi", std::process::id(), content.len()));
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn count_records_streams_and_previews() {
        let path = smi_file("CCO\nnot_a_smiles\nc1ccccc1\n");
        let (records, failed, preview) = count_records(&path).unwrap();
        assert_eq!((records, failed), (3, 1));
        assert_eq!(preview[0], ("CCO".to_string(), "3".to_string()));
        assert_eq!(preview[1].1, "parse error");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn calculate_to_file_writes_every_row_and_honours_cancel() {
        let path = smi_file("CCO\nnot_a_smiles\nc1ccccc1\nCCN\n");
        let (tx, rx) = mpsc::channel();
        let descriptors = [BatchDescriptor::MolecularWeight, BatchDescriptor::MaccsKeys];
        let (preview, rows) = calculate_to_file(&path, &descriptors, 4, &tx, &|| false).unwrap().unwrap();
        assert_eq!((rows, preview.rows), (4, 4));
        let csv = fs::read_to_string(results_path()).unwrap();
        assert_eq!(csv.lines().count(), 5);
        assert!(csv.starts_with("SMILES,MolWt,MACCS\n"));
        assert!(rx.try_iter().any(|r| matches!(r, WorkerResponse::Progress(p) if p >= 1.0)));
        assert!(calculate_to_file(&path, &descriptors, 4, &tx, &|| true).unwrap().is_none());
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(results_path());
    }
}
