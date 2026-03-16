#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// --- CONFIGURATION ---
const LM_STUDIO_BASE: &str = "http://127.0.0.1:1234";
const DEFAULT_MODEL: &str = "";
const DEFAULT_CONTEXT: u32 = 30_000;
const DELETE_INTERVAL_MS: u64 = 10;
const TYPE_INTERVAL_MS: u64 = 10;
const TRANSITION_MS: u64 = 500; // file-switch diffusion effect duration
const LINE_HEIGHT_PX: f32 = 14.0; // matches monospace size 11.5 + line gap
const SCROLL_LERP: f32 = 0.18;
const SCROLL_SNAP_PX: f32 = 1.0;
const MAX_FIX_ROUNDS: u32 = 3;
const MAX_FILE_RETRIES: u8 = 2; // per-file retry attempts before skipping
// Network timeouts (seconds)
const TIMEOUT_MODEL_LIST_SECS: u64 = 10;  // GET /api/v1/models
const TIMEOUT_MODEL_UNLOAD_SECS: u64 = 60; // POST /api/v1/models/unload
const TIMEOUT_MODEL_LOAD_SECS: u64 = 300;  // POST /api/v1/models/load (large models take time)
const TIMEOUT_LLM_CONNECT_SECS: u64 = 30;  // initial connection to LM Studio for chat completions
const TIMEOUT_LLM_IDLE_SECS: u64 = 300;    // max silence between stream chunks before giving up
const TIMEOUT_ANALYZING_SECS: u64 = 720;   // force Error if a file stays Analyzing this long (task crash guard)
const MAX_LOG_LINES: usize = 500;
/// Internal marker prefix for ASCII-art log lines (renders differently in the terminal).
const ART_LINE: char = '\x01';

// --- PALETTE (GitHub Dark + React hacker) ---
const C_BASE: egui::Color32 = egui::Color32::from_rgb(13, 17, 23);
const C_SURFACE: egui::Color32 = egui::Color32::from_rgb(22, 27, 34);
const C_RAISED: egui::Color32 = egui::Color32::from_rgb(33, 38, 45);
const C_BORDER: egui::Color32 = egui::Color32::from_rgb(48, 54, 61);
const C_ACCENT: egui::Color32 = egui::Color32::from_rgb(97, 218, 251);
const C_GREEN: egui::Color32 = egui::Color32::from_rgb(63, 185, 80);
const C_RED: egui::Color32 = egui::Color32::from_rgb(248, 81, 73);
const C_ORANGE: egui::Color32 = egui::Color32::from_rgb(210, 153, 34);
const C_PURPLE: egui::Color32 = egui::Color32::from_rgb(188, 140, 255);
const C_TEXT: egui::Color32 = egui::Color32::from_rgb(230, 237, 243);
const C_TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(139, 148, 158);
const C_TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(72, 79, 88);

// --- DATA MODELS ---
#[derive(Debug, Clone)]
struct ModelInfo {
    key: String,
    display_name: String,
    is_loaded: bool,
    loaded_context: Option<u32>,
    max_context_length: u32,
    instance_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fix {
    explanation: String,
    old: String,
    new: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeChange {
    fixes: Vec<Fix>,
}

#[derive(Debug, Clone, PartialEq)]
enum AnimPhase {
    Deleting(usize), // chars of `old` still visible (counts down)
    Typing(usize),   // chars of `new` typed so far (counts up)
}

#[derive(Debug, Clone, PartialEq)]
enum FileStatus {
    Pending,
    Analyzing,
    ReadyToAnimate,
    Animating { fix_index: usize, phase: AnimPhase },
    Completed,
    Error(String),
    PendingErrorFix(Vec<String>), // cargo error lines for this file
}

#[derive(Debug, Clone)]
struct FileTask {
    path: PathBuf,
    original_content: String,
    working_content: String,
    fixes: Vec<Fix>,
    status: FileStatus,
    change_reason: String,
    retry_count: u8,
    analyzing_since: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
enum AppState {
    Idle,
    LoadingModel,
    Scanning,
    Paused,
    Processing,
    CargoChecking,
    FixingErrors(u32),
    Finished,
}

// --- SHARED STATE ---
struct SharedState {
    app_state: AppState,
    root_path: Option<PathBuf>,
    files: Vec<FileTask>,
    current_file_index: usize,
    logs: Vec<String>,
    cargo_errors: Vec<String>,
    cargo_check_round: u32,
}

impl SharedState {
    fn new() -> Self {
        Self {
            app_state: AppState::Idle,
            root_path: None,
            files: Vec::new(),
            current_file_index: 0,
            logs: Vec::new(),
            cargo_errors: Vec::new(),
            cargo_check_round: 0,
        }
    }

    fn push_log(&mut self, msg: impl Into<String>) {
        self.logs.push(msg.into());
        if self.logs.len() > MAX_LOG_LINES {
            self.logs.drain(0..self.logs.len() - MAX_LOG_LINES);
        }
    }
}

// --- HELPERS ---
fn strip_markdown_json(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim().to_string()
}

fn char_byte_at(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len())
}

/// Extract absolute file paths from cargo error output (lines like `  --> src/foo.rs:4:5`).
fn parse_error_files(errors: &[String], root: &Path) -> Vec<PathBuf> {
    let mut paths: HashSet<PathBuf> = HashSet::new();
    for line in errors {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("-->") {
            let path_part = rest.trim().split(':').next().unwrap_or("").trim();
            if !path_part.is_empty() && !path_part.starts_with('<') {
                let normalized = path_part.replace('/', std::path::MAIN_SEPARATOR_STR);
                paths.insert(root.join(normalized));
            }
        }
    }
    paths.into_iter().collect()
}

// Snapshot of file state for rendering — built inside a scoped lock and passed
// to render_editor so the mutex is never held while egui lays out widgets.
struct EditorRenderData {
    display_content: String,
    is_animating: bool,
    header_text: String,
    header_color: egui::Color32,
}

fn build_editor_data(file: &FileTask) -> EditorRenderData {
    let is_animating = matches!(&file.status, FileStatus::Animating { .. });

    let (header_text, header_color) = match &file.status {
        FileStatus::Animating { fix_index, phase } => {
            if let Some(fix) = file.fixes.get(*fix_index) {
                let (icon, verb, color) = match phase {
                    AnimPhase::Deleting(_) => ("-", "Removing", C_RED),
                    AnimPhase::Typing(_) => ("+", "Writing", C_GREEN),
                };
                (
                    format!(
                        "{} {} [{}/{}]: {}",
                        icon,
                        verb,
                        fix_index + 1,
                        file.fixes.len(),
                        fix.explanation
                    ),
                    color,
                )
            } else {
                ("Animating...".into(), C_TEXT_DIM)
            }
        }
        _ => (format!("  {}", file.change_reason), C_ACCENT),
    };

    let display_content = match &file.status {
        FileStatus::Animating { fix_index, phase } => {
            if let Some(fix) = file.fixes.get(*fix_index) {
                if let Some(old_start) = file.working_content.find(fix.old.as_str()) {
                    let old_end = old_start + fix.old.len();
                    let before = &file.working_content[..old_start];
                    let after = &file.working_content[old_end..];
                    let mid = match phase {
                        AnimPhase::Deleting(chars_left) => {
                            let end = char_byte_at(&fix.old, *chars_left);
                            format!("{}█", &fix.old[..end])
                        }
                        AnimPhase::Typing(chars_typed) => {
                            let end = char_byte_at(&fix.new, *chars_typed);
                            format!("{}█", &fix.new[..end])
                        }
                    };
                    format!("{}{}{}", before, mid, after)
                } else {
                    file.working_content.clone()
                }
            } else {
                file.working_content.clone()
            }
        }
        FileStatus::Completed => file.working_content.clone(),
        _ => file.original_content.clone(),
    };

    EditorRenderData {
        display_content,
        is_animating,
        header_text,
        header_color,
    }
}

// --- EASTER EGG ASCII ART ---
fn pick_ascii_art() -> &'static [&'static str] {
    const ARTS: &[&[&str]] = &[
        &[
            r"    _~^~^~_     ",
            r" \) /  o o  \ (/",
            r"   '_   -   _'  ",
            r"   / '-----' \  ",
            r"    rustacean   ",
        ],
        &[
            r"  .---.   .--.  ",
            r" /     \ /     \",
            r"|  (.)  |  (.)  |",
            r" \      X      / ",
            r"  '----' '----'  ",
            r"   for science   ",
        ],
        &[
            r"  .----------.  ",
            r"  | LGTM  ✅ |  ",
            r"  '----------'  ",
            r"      \  /      ",
            r"      /  \      ",
        ],
        &[
            r"  ___           ",
            r" /   \          ",
            r"| o o |  beep   ",
            r" \___/   boop   ",
            r"   |            ",
            r" __|__          ",
        ],
        &[
            r"  \ | /   ",
            r"  -[*]-   ",
            r"  / | \   ",
            r"          ",
            r"  fixed!  ",
        ],
        &[
            r"  ( (      ",
            r"   ) )     ",
            r" .-----.   ",
            r" | | | |   ",
            r" '-----'   ",
            r"  coffee   ",
        ],
        &[
            r" +----------+  ",
            r" | fn main()|  ",
            r" |  ok !!!  |  ",
            r" +----------+  ",
        ],
        &[
            r"  *  .  *  .  * ",
            r" .  *  .  *  .  ",
            r"  *  .  *  .  * ",
            r"   optimized!   ",
        ],
    ];
    let idx = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_micros() as usize
        % ARTS.len();
    ARTS[idx]
}

// --- DIFFUSION TRANSITION ---

/// Cheap deterministic pseudo-random 0.0..1.0 from a character position.
fn diffuse_hash(pos: usize) -> f32 {
    let h = pos.wrapping_mul(2654435761_usize) ^ pos.wrapping_shr(16);
    (h & 0xFFFF) as f32 / 65535.0
}

const DIFFUSE_NOISE: &[u8] = b"!@#$%^&*[]{}|<>/\\~+=_?;:.,'\"0123456789";

fn diffuse_noise_char(pos: usize, t: f32) -> char {
    let idx = pos.wrapping_add((t * 14.0) as usize) % DIFFUSE_NOISE.len();
    DIFFUSE_NOISE[idx] as char
}

/// Renders a single frame of the file-switch diffusion effect.
/// progress 0.0–0.45 : old content scatters to noise (low-hash positions go first)
/// progress 0.45–1.0 : new content resolves from noise (low-hash positions appear first)
fn compute_diffusion_frame(from: &str, to: &str, progress: f32) -> String {
    if progress <= 0.0 { return from.to_string(); }
    if progress >= 1.0 { return to.to_string(); }

    if progress < 0.45 {
        let p = progress / 0.45;
        from.chars().enumerate().map(|(i, ch)| {
            if ch == '\n' || ch.is_whitespace() { return ch; }
            if diffuse_hash(i) < p { diffuse_noise_char(i, p) } else { ch }
        }).collect()
    } else {
        let p = (progress - 0.45) / 0.55;
        to.chars().enumerate().map(|(i, ch)| {
            if ch == '\n' || ch.is_whitespace() { return ch; }
            if diffuse_hash(i) < p { ch } else { diffuse_noise_char(i, p) }
        }).collect()
    }
}

// --- APP STRUCTURE ---
struct RefactorApp {
    state: Arc<Mutex<SharedState>>,
    tx: mpsc::UnboundedSender<Message>,
    rx: mpsc::UnboundedReceiver<Message>,
    egui_ctx: egui::Context,
    last_anim_tick: Instant,
    editor_scroll_y: f32,
    scroll_target_y: f32,
    // UI-only state — lives on the main thread, no mutex needed
    available_models: Vec<ModelInfo>,
    selected_model_idx: usize,
    context_length: u32,
    analysis_prompt: String,
    // Cancels any in-flight LLM stream when set to true (pause / close folder).
    llm_cancel: Arc<AtomicBool>,
    // Backup directory for undo — created on folder open, deleted on folder close.
    backup_dir: Option<PathBuf>,
    // File-switch diffusion effect
    transition_start: Option<Instant>,
    transition_from_content: String,
    last_shown_file_idx: Option<usize>,
}

enum Message {
    Log(String),
    SetState(AppState),
    AddFile(PathBuf, String),
    UpdateFixes(usize, Vec<Fix>),
    SetFileStatus(usize, FileStatus),
    CargoCheckResult(Vec<String>),
    ModelListReceived(Vec<ModelInfo>),
    ModelLoaded(String),
    ModelUnloaded(String),
}

impl RefactorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self::apply_modern_theme(&cc.egui_ctx);
        let app = Self {
            state: Arc::new(Mutex::new(SharedState::new())),
            tx,
            rx,
            egui_ctx: cc.egui_ctx.clone(),
            last_anim_tick: Instant::now(),
            editor_scroll_y: 0.0,
            scroll_target_y: 0.0,
            available_models: Vec::new(),
            selected_model_idx: 0,
            context_length: DEFAULT_CONTEXT,
            analysis_prompt: "Analyze the following file and identify specific improvements.".into(),
            llm_cancel: Arc::new(AtomicBool::new(false)),
            backup_dir: None,
            transition_start: None,
            transition_from_content: String::new(),
            last_shown_file_idx: None,
        };
        app.spawn_fetch_models();
        app
    }

    fn apply_modern_theme(ctx: &egui::Context) {
        // Register NotoEmoji-Regular and ensure it is a fallback for *both*
        // Proportional and Monospace — egui only adds it to Proportional by default.
        let mut font_defs = egui::FontDefinitions::default();
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            let list = font_defs.families.entry(family).or_default();
            if !list.iter().any(|f| f == "NotoEmoji-Regular") {
                list.push("NotoEmoji-Regular".to_owned());
            }
        }
        ctx.set_fonts(font_defs);

        let mut visuals = egui::Visuals::dark();

        // Background surfaces
        visuals.panel_fill = C_BASE;
        visuals.window_fill = C_SURFACE;
        visuals.window_stroke = egui::Stroke::new(1.0, C_BORDER);
        visuals.window_corner_radius = egui::CornerRadius::same(6);

        // Code editor / text input background
        visuals.extreme_bg_color = egui::Color32::from_rgb(10, 14, 20);

        // Scrollbar
        visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };

        // Widget rounding — tight/consistent
        let r4 = egui::CornerRadius::same(4);
        visuals.widgets.noninteractive.corner_radius = r4;
        visuals.widgets.inactive.corner_radius = r4;
        visuals.widgets.hovered.corner_radius = r4;
        visuals.widgets.active.corner_radius = r4;
        visuals.widgets.open.corner_radius = r4;

        // Noninteractive (separators, static text backgrounds)
        visuals.widgets.noninteractive.bg_fill = C_RAISED;
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, C_BORDER);
        visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, C_TEXT_DIM);

        // Buttons / inputs at rest
        visuals.widgets.inactive.bg_fill = C_RAISED;
        visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, C_BORDER);
        visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, C_TEXT);

        // Hovered
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(56, 62, 70);
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, C_ACCENT);
        visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, C_TEXT);

        // Active / pressed
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(64, 70, 80);
        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, C_ACCENT);
        visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, C_ACCENT);

        // Open (dropdown menus)
        visuals.widgets.open.bg_fill = C_RAISED;
        visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, C_BORDER);
        visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, C_TEXT);

        // Selection highlight (cyan glow)
        visuals.selection.bg_fill =
            egui::Color32::from_rgba_unmultiplied(97, 218, 251, 45);
        visuals.selection.stroke = egui::Stroke::new(1.0, C_ACCENT);

        ctx.set_visuals(visuals);
    }

    fn current_model_key(&self) -> String {
        self.available_models
            .get(self.selected_model_idx)
            .map(|m| m.key.clone())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    // --- ASYNC TASKS ---

    fn spawn_fetch_models(&self) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(TIMEOUT_MODEL_LIST_SECS))
                .build()
            {
                Ok(c) => c,
                Err(_) => return,
            };
            match client
                .get(format!("{}/api/v1/models", LM_STUDIO_BASE))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let body = resp.text().await.unwrap_or_default();
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                        // /api/v1 returns { "models": [...] }; /api/v0 returns { "data": [...] }
                        let arr = json["models"]
                            .as_array()
                            .or_else(|| json["data"].as_array());
                        if let Some(arr) = arr {
                            let mut models = Vec::new();
                            for m in arr {
                                let key = m["id"]
                                    .as_str()
                                    .or_else(|| m["key"].as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if key.is_empty() {
                                    continue;
                                }
                                let display_name = m["display_name"]
                                    .as_str()
                                    .unwrap_or(&key)
                                    .to_string();
                                let max_ctx =
                                    m["max_context_length"].as_u64().unwrap_or(8192) as u32;
                                let instances = m["loaded_instances"].as_array();
                                let is_loaded = instances
                                    .map_or(false, |v| !v.is_empty())
                                    || m["state"].as_str() == Some("loaded");
                                let first_instance =
                                    instances.and_then(|v| v.first());
                                let loaded_context = first_instance
                                    .and_then(|i| i["config"]["context_length"].as_u64())
                                    .map(|v| v as u32);
                                let instance_id = first_instance
                                    .and_then(|i| i["identifier"].as_str()
                                        .or_else(|| i["id"].as_str()))
                                    .map(|s| s.to_string());
                                models.push(ModelInfo {
                                    key,
                                    display_name,
                                    is_loaded,
                                    loaded_context,
                                    max_context_length: max_ctx,
                                    instance_id,
                                });
                            }
                            let _ = tx.send(Message::ModelListReceived(models));
                        }
                    }
                }
                Ok(resp) => {
                    let _ = tx.send(Message::Log(format!(
                        "  ⚠️ Model list: HTTP {}",
                        resp.status()
                    )));
                }
                Err(e) => {
                    let _ = tx.send(Message::Log(format!(
                        "  ⚠️ LM Studio unreachable: {}",
                        e
                    )));
                }
            }
        });
    }

    fn spawn_unload_model(&self, model_key: String, instance_id: Option<String>) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let _ = tx.send(Message::Log(format!("⏏  Ejecting {}...", model_key)));
            let client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(TIMEOUT_MODEL_UNLOAD_SECS))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Message::Log(format!("  ❌ Client error: {}", e)));
                    return;
                }
            };
            let body = if let Some(id) = &instance_id {
                serde_json::json!({ "instance_id": id })
            } else {
                // Fallback: use model key as instance_id (best-effort)
                serde_json::json!({ "instance_id": model_key })
            };
            match client
                .post(format!("{}/api/v1/models/unload", LM_STUDIO_BASE))
                .json(&body)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let _ = tx.send(Message::ModelUnloaded(model_key));
                }
                Ok(resp) => {
                    let _ = tx.send(Message::Log(format!(
                        "  ⚠️ Eject HTTP {}",
                        resp.status()
                    )));
                }
                Err(e) => {
                    let _ = tx.send(Message::Log(format!("  ❌ Eject error: {}", e)));
                }
            }
        });
    }

    fn spawn_load_model(
        &self,
        model_key: String,
        context_length: u32,
        eject_first: Option<(String, Option<String>)>, // (model_key, instance_id)
    ) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let _ = tx.send(Message::SetState(AppState::LoadingModel));

            let client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(TIMEOUT_MODEL_LOAD_SECS))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Message::Log(format!("  ❌ Client error: {}", e)));
                    let _ = tx.send(Message::SetState(AppState::Idle));
                    return;
                }
            };

            // Eject any currently loaded model first
            if let Some((old_key, old_instance_id)) = eject_first {
                let _ = tx.send(Message::Log(format!("⏏  Ejecting {}...", old_key)));
                let eject_body = if let Some(id) = &old_instance_id {
                    serde_json::json!({ "instance_id": id })
                } else {
                    serde_json::json!({ "instance_id": old_key })
                };
                match client
                    .post(format!("{}/api/v1/models/unload", LM_STUDIO_BASE))
                    .json(&eject_body)
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        let _ = tx.send(Message::ModelUnloaded(old_key));
                    }
                    Ok(resp) => {
                        let _ = tx.send(Message::Log(format!(
                            "  ⚠️ Eject HTTP {} — continuing anyway",
                            resp.status()
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(Message::Log(format!(
                            "  ⚠️ Eject error: {} — continuing anyway",
                            e
                        )));
                    }
                }
            }

            let _ = tx.send(Message::Log(format!(
                "⚡ Loading {}  (ctx: {} tokens)...",
                model_key, context_length
            )));

            match client
                .post(format!("{}/api/v1/models/load", LM_STUDIO_BASE))
                .json(&serde_json::json!({
                    "model": model_key,
                    "context_length": context_length,
                    "echo_load_config": true,
                }))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let body = resp.text().await.unwrap_or_default();
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(secs) = json["load_time_seconds"].as_f64() {
                            let _ = tx.send(Message::Log(format!(
                                "  ✅ Loaded in {:.1}s",
                                secs
                            )));
                        }
                        if let Some(cfg) = json["load_config"].as_object() {
                            if let Some(ctx) =
                                cfg.get("context_length").and_then(|v| v.as_u64())
                            {
                                let _ = tx.send(Message::Log(format!(
                                    "  📐 context_length: {} tokens",
                                    ctx
                                )));
                            }
                        }
                    }
                    let _ = tx.send(Message::ModelLoaded(model_key));
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    let snippet = body.chars().take(200).collect::<String>();
                    let _ = tx.send(Message::Log(format!(
                        "  ❌ Load failed HTTP {}: {}",
                        status, snippet
                    )));
                    let _ = tx.send(Message::SetState(AppState::Idle));
                }
                Err(e) => {
                    let _ = tx.send(Message::Log(format!("  ❌ Load error: {}", e)));
                    let _ = tx.send(Message::SetState(AppState::Idle));
                }
            }
        });
    }

    fn spawn_llm_analysis(&self, file_index: usize, content: String, path: String) {
        let tx = self.tx.clone();
        let ctx = self.egui_ctx.clone();
        let model_key = self.current_model_key();
        let cancel = self.llm_cancel.clone();
        let instruction = self.analysis_prompt.clone();
        tokio::spawn(async move {
            let _ = tx.send(Message::Log(format!("⏳ Analyzing {}...", path)));
            let prompt = format!(
                r#"{instruction} Answer ONLY in JSON matching this exact schema:

{{
  "fixes": [
    {{
      "explanation": "brief one-line reason for this change",
      "old": "exact verbatim substring to replace (copy it from the source)",
      "new": "the replacement code"
    }}
  ]
}}

SOURCE ({path}):
{content}"#,
                path = path,
                content = content
            );
            if Self::do_llm_request(tx.clone(), ctx.clone(), file_index, model_key, prompt, cancel).await {
                // Cancelled by pause — reset to Pending so it re-runs on resume.
                let _ = tx.send(Message::Log("  ⏸ analysis cancelled (paused)".into()));
                let _ = tx.send(Message::SetFileStatus(file_index, FileStatus::Pending));
                ctx.request_repaint();
            }
        });
    }

    fn spawn_llm_error_fix(
        &self,
        file_index: usize,
        content: String,
        path: String,
        errors: Vec<String>,
    ) {
        let tx = self.tx.clone();
        let ctx = self.egui_ctx.clone();
        let model_key = self.current_model_key();
        let cancel = self.llm_cancel.clone();
        tokio::spawn(async move {
            let _ = tx.send(Message::Log(format!(
                "🔧 Fixing compiler errors in {}...",
                path
            )));
            let error_text = errors.join("\n");
            let prompt = format!(
                r#"Answer ONLY in JSON matching this exact schema. Fix the errors in this file:

{{
  "fixes": [
    {{
      "explanation": "what this fixes",
      "old": "exact verbatim substring from the source to replace",
      "new": "the corrected code"
    }}
  ]
}}

COMPILER ERRORS:
{error_text}

SOURCE ({path}):
{content}"#,
                error_text = error_text,
                path = path,
                content = content
            );
            if Self::do_llm_request(tx.clone(), ctx.clone(), file_index, model_key, prompt, cancel).await {
                // Cancelled by pause — restore PendingErrorFix so it re-runs on resume.
                let _ = tx.send(Message::Log("  ⏸ error-fix cancelled (paused)".into()));
                let _ = tx.send(Message::SetFileStatus(
                    file_index,
                    FileStatus::PendingErrorFix(errors),
                ));
                ctx.request_repaint();
            }
        });
    }

    /// Shared HTTP logic — uses SSE streaming so the terminal shows live progress.
    /// Returns `true` if the stream was cancelled (pause), `false` if it completed normally.
    async fn do_llm_request(
        tx: mpsc::UnboundedSender<Message>,
        ctx: egui::Context,
        file_index: usize,
        model_key: String,
        prompt: String,
        cancel: Arc<AtomicBool>,
    ) -> bool {
        let _ = tx.send(Message::SetFileStatus(file_index, FileStatus::Analyzing));
        let request_start = Instant::now();

        // Fresh client per request — avoids stale keep-alive connections from
        // LM Studio's short 5-second idle timeout between files.
        let client = match reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(TIMEOUT_LLM_CONNECT_SECS))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Message::SetFileStatus(
                    file_index,
                    FileStatus::Error(e.to_string()),
                ));
                ctx.request_repaint();
                return false;
            }
        };

        let resp = match client
            .post(format!("{}/v1/chat/completions", LM_STUDIO_BASE))
            .json(&serde_json::json!({
                "model": model_key,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON API. Return only raw JSON — no markdown, no commentary."
                    },
                    { "role": "user", "content": prompt }
                ],
                "temperature": 0.2,
                "stream": true,
            }))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                let snippet = body.chars().take(300).collect::<String>();
                let _ = tx.send(Message::SetFileStatus(
                    file_index,
                    FileStatus::Error(format!("HTTP {}: {}", status, snippet)),
                ));
                ctx.request_repaint();
                return false;
            }
            Err(e) => {
                let _ = tx.send(Message::SetFileStatus(
                    file_index,
                    FileStatus::Error(e.to_string()),
                ));
                ctx.request_repaint();
                return false;
            }
        };

        let _ = tx.send(Message::Log("  processing prompt...".into()));
        ctx.request_repaint();

        // Consume the SSE stream chunk by chunk
        let mut resp = resp;
        let mut line_buf = String::new();
        let mut full_content = String::new();
        let mut completion_tokens: u64 = 0;
        let mut prompt_tokens: u64 = 0;
        let mut first_token_logged = false;
        let mut last_reported_tokens: u64 = 0;

        let idle_timeout = Duration::from_secs(TIMEOUT_LLM_IDLE_SECS);
        'stream: loop {
            // Check cancel flag before every chunk — set by pause button.
            if cancel.load(Ordering::Relaxed) {
                return true;
            }

            let chunk = tokio::time::timeout(idle_timeout, resp.chunk()).await;
            let bytes = match chunk {
                Ok(Ok(Some(b))) => b,
                Ok(Ok(None)) => break 'stream, // stream finished normally
                Ok(Err(e)) => {
                    let _ = tx.send(Message::SetFileStatus(
                        file_index,
                        FileStatus::Error(format!("Stream error: {}", e)),
                    ));
                    ctx.request_repaint();
                    return false;
                }
                Err(_) => {
                    // No data received for TIMEOUT_LLM_IDLE_SECS — model stalled
                    let _ = tx.send(Message::SetFileStatus(
                        file_index,
                        FileStatus::Error(format!(
                            "Stream idle for {}s — model may have stalled",
                            TIMEOUT_LLM_IDLE_SECS
                        )),
                    ));
                    ctx.request_repaint();
                    return false;
                }
            };

            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            // Process every complete line in the buffer
            loop {
                let newline = match line_buf.find('\n') {
                    Some(p) => p,
                    None => break,
                };
                let raw = line_buf[..newline].trim().to_string();
                line_buf = line_buf[newline + 1..].to_string();

                let data = match raw.strip_prefix("data: ") {
                    Some(d) => d.to_string(),
                    None => continue,
                };

                if data == "[DONE]" {
                    break 'stream;
                }

                let json = match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(j) => j,
                    Err(_) => continue,
                };

                // Accumulate delta content
                if let Some(piece) = json["choices"][0]["delta"]["content"].as_str() {
                    if !piece.is_empty() {
                        if !first_token_logged {
                            first_token_logged = true;
                            let t = request_start.elapsed().as_secs_f32();
                            let _ = tx.send(Message::Log(format!(
                                "  first token in {:.1}s — generating...",
                                t
                            )));
                            ctx.request_repaint();
                        }
                        full_content.push_str(piece);
                        completion_tokens += 1;
                    }
                }

                // Some models stream usage in the final chunk
                if let Some(pt) = json["usage"]["prompt_tokens"].as_u64() {
                    prompt_tokens = pt;
                }
                if let Some(ct) = json["usage"]["completion_tokens"].as_u64() {
                    // Use exact count if provided
                    completion_tokens = ct;
                }

                // Live progress every 30 tokens
                if completion_tokens > 0
                    && completion_tokens % 30 == 0
                    && completion_tokens != last_reported_tokens
                {
                    last_reported_tokens = completion_tokens;
                    let elapsed = request_start.elapsed().as_secs_f32();
                    let tps = if elapsed > 0.0 {
                        completion_tokens as f32 / elapsed
                    } else {
                        0.0
                    };
                    let _ = tx.send(Message::Log(format!(
                        "  ~ {} tok  {:.1}s  {:.0} tok/s",
                        completion_tokens, elapsed, tps
                    )));
                    ctx.request_repaint();
                }
            }
        }

        // Final stats
        let elapsed = request_start.elapsed().as_secs_f32();
        let tps = if elapsed > 0.0 {
            completion_tokens as f32 / elapsed
        } else {
            0.0
        };
        if prompt_tokens > 0 {
            let total = prompt_tokens + completion_tokens;
            let _ = tx.send(Message::Log(format!(
                "  📊 {total} tok  ({prompt_tokens} prompt + {completion_tokens} completion)  {elapsed:.1}s  {tps:.0} tok/s"
            )));
        } else {
            let _ = tx.send(Message::Log(format!(
                "  📊 ~{completion_tokens} tok  {elapsed:.1}s  {tps:.0} tok/s"
            )));
        }

        // Parse accumulated JSON
        let clean = strip_markdown_json(&full_content);
        match serde_json::from_str::<CodeChange>(&clean) {
            Ok(mut change) => {
                // Drop fixes with empty `old` — applying them would corrupt the file
                // (Rust's str::find("") always returns Some(0), prepending content).
                let before = change.fixes.len();
                change.fixes.retain(|f| !f.old.is_empty() && !f.new.is_empty() && f.old != f.new);
                let skipped = before - change.fixes.len();
                if skipped > 0 {
                    let _ = tx.send(Message::Log(format!(
                        "  ! {} fix(es) skipped (empty or no-op old string)",
                        skipped
                    )));
                }
                let count = change.fixes.len();
                let _ = tx.send(Message::Log(format!("  ✅ {} fix(es) queued", count)));
                let _ = tx.send(Message::UpdateFixes(file_index, change.fixes));
                let _ = tx.send(Message::SetFileStatus(
                    file_index,
                    FileStatus::ReadyToAnimate,
                ));
            }
            Err(e) => {
                let snippet = clean.chars().take(200).collect::<String>();
                let _ = tx.send(Message::SetFileStatus(
                    file_index,
                    FileStatus::Error(format!("Parse error: {} — got: {}", e, snippet)),
                ));
            }
        }

        ctx.request_repaint();
        false
    }

    fn spawn_cargo_check(&self) {
        let tx = self.tx.clone();
        let ctx = self.egui_ctx.clone();
        let state = self.state.clone();
        tokio::spawn(async move {
            let _ = tx.send(Message::Log("🔎 Running cargo check...".into()));
            ctx.request_repaint();
            let root = state.lock().unwrap_or_else(|e| e.into_inner()).root_path.clone();
            if let Some(path) = root {
                // Use tokio's async process so we don't block a worker thread.
                match tokio::process::Command::new("cargo")
                    .arg("check")
                    .current_dir(&path)
                    .output()
                    .await
                {
                    Ok(out) => {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        // Capture error lines, warning lines, and --> file references
                        let errors: Vec<String> = stderr
                            .lines()
                            .filter(|l| {
                                let t = l.trim();
                                t.starts_with("error")
                                    || t.starts_with("warning")
                                    || t.starts_with("-->")
                            })
                            .map(|l| l.to_string())
                            .collect();

                        // Use the exit code as source of truth — parsing alone can
                        // miss errors that don't follow the error[Exxxx] pattern.
                        let failed = !out.status.success();
                        let err_count = errors
                            .iter()
                            .filter(|l| l.trim_start().starts_with("error"))
                            .count();
                        let warn_count = errors
                            .iter()
                            .filter(|l| l.trim_start().starts_with("warning"))
                            .count();

                        if failed {
                            let _ = tx.send(Message::Log(format!(
                                "  ⚠️  {} error(s), {} warning(s)",
                                err_count, warn_count
                            )));
                        } else {
                            let _ = tx.send(Message::Log("  ✅ cargo check passed!".into()));
                        }
                        // Only send errors vec when the build actually failed so
                        // CargoCheckResult doesn't trigger a spurious fix round.
                        let _ = tx.send(Message::CargoCheckResult(
                            if failed { errors } else { vec![] }
                        ));
                        ctx.request_repaint();
                    }
                    Err(e) => {
                        let _ = tx.send(Message::Log(format!(
                            "  ❌ cargo check failed to run: {}",
                            e
                        )));
                        let _ = tx.send(Message::SetState(AppState::Finished));
                        ctx.request_repaint();
                    }
                }
            }
        });
    }

    // --- UI COMPONENTS ---

    fn render_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.label(egui::RichText::new("⚡").color(C_ACCENT).size(15.0));
            ui.add_space(3.0);
            ui.label(
                egui::RichText::new("OPTIMIZER")
                    .color(C_TEXT)
                    .size(13.0)
                    .strong()
                    .family(egui::FontFamily::Monospace),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let (dot_color, status_text): (egui::Color32, String) = match &state.app_state {
                    AppState::Idle => (C_TEXT_MUTED, "idle".into()),
                    AppState::LoadingModel => (C_ACCENT, "loading model".into()),
                    AppState::Scanning => (C_ORANGE, "scanning".into()),
                    AppState::Paused => (C_ORANGE, "paused".into()),
                    AppState::Processing => (C_GREEN, "processing".into()),
                    AppState::CargoChecking => (C_ACCENT, "cargo check".into()),
                    AppState::FixingErrors(n) => {
                        (C_ORANGE, format!("fixing errors  {}/{}", n, MAX_FIX_ROUNDS))
                    }
                    AppState::Finished => (C_GREEN, "done".into()),
                };
                ui.label(
                    egui::RichText::new(&status_text)
                        .color(C_TEXT_DIM)
                        .size(11.0)
                        .family(egui::FontFamily::Monospace),
                );
                ui.add_space(6.0);
                let (resp, painter) =
                    ui.allocate_painter(egui::vec2(8.0, 8.0), egui::Sense::hover());
                painter.circle_filled(resp.rect.center(), 4.0, dot_color);
                ui.add_space(2.0);
            });
        });
    }

    fn render_model_panel(&mut self, ui: &mut egui::Ui) {
        let app_state = self.state.lock().unwrap_or_else(|e| e.into_inner()).app_state.clone();
        let is_loading = app_state == AppState::LoadingModel;

        // Section label + refresh button
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("MODEL")
                    .color(C_TEXT_MUTED)
                    .size(10.0)
                    .strong()
                    .family(egui::FontFamily::Monospace),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(
                        egui::Button::new(
                            egui::RichText::new("↺").color(C_TEXT_DIM).size(14.0),
                        )
                        .frame(false),
                    )
                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                    .on_hover_text("Refresh model list from LM Studio")
                    .clicked()
                {
                    self.spawn_fetch_models();
                }
            });
        });
        ui.add_space(5.0);

        if self.available_models.is_empty() {
            ui.label(
                egui::RichText::new("No models — is LM Studio running?")
                    .color(C_TEXT_MUTED)
                    .size(11.0)
                    .italics(),
            );
        } else {
            if self.selected_model_idx >= self.available_models.len() {
                self.selected_model_idx = 0;
            }

            let current_label = {
                let m = &self.available_models[self.selected_model_idx];
                if m.is_loaded {
                    format!("* {}", m.display_name)
                } else {
                    format!("- {}", m.display_name)
                }
            };

            egui::ComboBox::from_id_salt("model_selector")
                .selected_text(
                    egui::RichText::new(current_label)
                        .color(C_TEXT)
                        .size(12.0)
                        .family(egui::FontFamily::Monospace),
                )
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    for (i, m) in self.available_models.iter().enumerate() {
                        let (dot, dot_color) = if m.is_loaded {
                            ("*", C_GREEN)
                        } else {
                            ("-", C_TEXT_MUTED)
                        };
                        let ctx_k = m.max_context_length / 1000;
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(dot).color(dot_color).size(10.0));
                            ui.selectable_value(
                                &mut self.selected_model_idx,
                                i,
                                egui::RichText::new(format!("{}  {}K", m.display_name, ctx_k))
                                    .color(C_TEXT)
                                    .size(12.0),
                            );
                        });
                    }
                });

            ui.add_space(6.0);

            // Context length control
            egui::Frame::NONE
                .fill(C_RAISED)
                .corner_radius(4)
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("CTX")
                                .color(C_TEXT_MUTED)
                                .size(10.0)
                                .family(egui::FontFamily::Monospace),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.context_length)
                                    .range(1024..=128_000)
                                    .speed(512.0)
                                    .suffix(" tok"),
                            );
                        });
                    });
                });

            ui.add_space(8.0);

            if is_loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new("Loading model...")
                            .color(C_ACCENT)
                            .size(12.0),
                    );
                });
            } else {
                let selected_key = self
                    .available_models
                    .get(self.selected_model_idx)
                    .map(|m| m.key.clone())
                    .unwrap_or_default();
                let selected_loaded = self
                    .available_models
                    .get(self.selected_model_idx)
                    .map_or(false, |m| m.is_loaded);
                let selected_instance_id = self
                    .available_models
                    .get(self.selected_model_idx)
                    .and_then(|m| m.instance_id.clone());

                // Find any OTHER model that's currently loaded (to auto-eject)
                let other_loaded = self
                    .available_models
                    .iter()
                    .find(|m| m.is_loaded && m.key != selected_key)
                    .map(|m| (m.key.clone(), m.instance_id.clone()));

                ui.horizontal(|ui| {
                    // Load / Reload button
                    let btn_text = if selected_loaded { "↺  Reload" } else { "⚡  Load" };
                    let load_btn = egui::Button::new(
                        egui::RichText::new(btn_text)
                            .color(egui::Color32::from_rgb(10, 14, 20))
                            .size(12.0)
                            .strong()
                            .family(egui::FontFamily::Monospace),
                    )
                    .fill(C_ACCENT)
                    .corner_radius(4);
                    let load_width = if selected_loaded {
                        ui.available_width() - 38.0
                    } else {
                        ui.available_width()
                    };
                    if ui.add_sized([load_width, 26.0], load_btn)
                        .on_hover_cursor(egui::CursorIcon::PointingHand)
                        .clicked() {
                        let ctx = self.context_length;
                        // Always eject before loading: either the same model (reload)
                        // or a different model that happens to be loaded.
                        let eject = if selected_loaded {
                            Some((selected_key.clone(), selected_instance_id.clone()))
                        } else {
                            other_loaded
                        };
                        self.spawn_load_model(selected_key.clone(), ctx, eject);
                    }

                    // Eject button — only shown when selected model is currently loaded
                    if selected_loaded {
                        let eject_btn = egui::Button::new(
                            egui::RichText::new("⏏")
                                .color(C_TEXT_DIM)
                                .size(13.0),
                        )
                        .fill(C_RAISED)
                        .corner_radius(4)
                        .stroke(egui::Stroke::new(1.0, C_BORDER));
                        if ui
                            .add_sized([30.0, 26.0], eject_btn)
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .on_hover_text("Eject model from memory")
                            .clicked()
                        {
                            self.spawn_unload_model(selected_key, selected_instance_id);
                        }
                    }
                });
            }
        }
    }

    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        egui::Frame::NONE
            .inner_margin(egui::Margin { left: 10, right: 8, top: 10, bottom: 10 })
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                    ui.set_width(ui.available_width());
                self.render_model_panel(ui);

                ui.add_space(8.0);
                ui.add(egui::Separator::default().spacing(0.0));
                ui.add_space(6.0);

                // Prompt section
                ui.label(
                    egui::RichText::new("PROMPT")
                        .color(C_TEXT_MUTED)
                        .size(10.0)
                        .strong()
                        .family(egui::FontFamily::Monospace),
                );
                ui.add_space(4.0);
                egui::Frame::NONE
                    .fill(C_RAISED)
                    .corner_radius(4)
                    .stroke(egui::Stroke::new(1.0, C_BORDER))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.add(
                            egui::TextEdit::multiline(&mut self.analysis_prompt)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(3)
                                .text_color(C_TEXT)
                                .frame(false),
                        );
                    });

                ui.add_space(8.0);
                ui.add(egui::Separator::default().spacing(0.0));
                ui.add_space(6.0);

                // Source folder controls
                ui.label(
                    egui::RichText::new("SOURCE")
                        .color(C_TEXT_MUTED)
                        .size(10.0)
                        .strong()
                        .family(egui::FontFamily::Monospace),
                );
                ui.add_space(4.0);

                let folder_is_open = self.state.lock().unwrap_or_else(|e| e.into_inner()).root_path.is_some();

                ui.horizontal(|ui| {
                    if folder_is_open {
                        let btn = egui::Button::new(
                            egui::RichText::new("📂  Close Folder")
                                .color(C_TEXT)
                                .size(12.0),
                        )
                        .fill(C_RAISED)
                        .corner_radius(4);
                        if ui.add_sized([ui.available_width() * 0.55, 26.0], btn)
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .clicked() {
                            self.close_folder();
                        }
                    } else {
                        let btn = egui::Button::new(
                            egui::RichText::new("📁  Open Folder")
                                .color(C_TEXT)
                                .size(12.0),
                        )
                        .fill(C_RAISED)
                        .corner_radius(4);
                        if ui.add_sized([ui.available_width() * 0.55, 26.0], btn)
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .clicked() {
                            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                self.scan_folder(&path);
                            }
                        }
                    }

                    let (app_state, has_files) = {
                        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
                        (s.app_state.clone(), !s.files.is_empty())
                    };

                    if has_files
                        && (app_state == AppState::Paused
                            || app_state == AppState::Processing
                            || app_state == AppState::CargoChecking
                            || matches!(app_state, AppState::FixingErrors(_)))
                    {
                        let is_processing = app_state == AppState::Processing
                            || app_state == AppState::CargoChecking
                            || matches!(app_state, AppState::FixingErrors(_));
                        let (btn_text, bg, fg) = if is_processing {
                            ("⏸  Pause", C_RED, C_TEXT)
                        } else {
                            ("▶  Start", C_ACCENT, egui::Color32::from_rgb(10, 14, 20))
                        };
                        let btn = egui::Button::new(
                            egui::RichText::new(btn_text)
                                .color(fg)
                                .size(12.0)
                                .strong(),
                        )
                        .fill(bg)
                        .corner_radius(4);
                        if ui.add_sized([ui.available_width(), 26.0], btn)
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .clicked() {
                            let next = if is_processing {
                                // Cancel any active LLM stream immediately.
                                self.llm_cancel.store(true, Ordering::Relaxed);
                                AppState::Paused
                            } else {
                                // Allow new LLM streams to start.
                                self.llm_cancel.store(false, Ordering::Relaxed);
                                AppState::Processing
                            };
                            let _ = self.tx.send(Message::SetState(next));
                        }
                    }
                });

                ui.add_space(8.0);
                ui.add(egui::Separator::default().spacing(0.0));
                ui.add_space(6.0);

                // Queue section header with file count
                let file_count = self.state.lock().unwrap_or_else(|e| e.into_inner()).files.len();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("QUEUE")
                            .color(C_TEXT_MUTED)
                            .size(10.0)
                            .strong()
                            .family(egui::FontFamily::Monospace),
                    );
                    if file_count > 0 {
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{}", file_count))
                                        .color(C_TEXT_MUTED)
                                        .size(10.0)
                                        .family(egui::FontFamily::Monospace),
                                );
                            },
                        );
                    }
                });
                ui.add_space(4.0);

                // Snapshot row data so the lock isn't held during rendering.
                struct QueueRow {
                    name: String,         // full filename (shown on hover)
                    display_name: String, // truncated to fit the column
                    badge: &'static str,
                    badge_color: egui::Color32,
                    name_color: egui::Color32,
                    removable: bool,
                    undoable: bool,
                    hover: Option<String>,
                }
                let rows: Vec<QueueRow> = {
                    let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    state.files.iter().enumerate().map(|(i, file)| {
                        let (badge, badge_color) = match &file.status {
                            FileStatus::Pending           => (".", C_TEXT_MUTED),
                            FileStatus::Analyzing         => ("~", C_PURPLE),
                            FileStatus::ReadyToAnimate    => ("-", C_ORANGE),
                            FileStatus::Animating { .. }  => (">", C_ACCENT),
                            FileStatus::Completed         => ("+", C_GREEN),
                            FileStatus::Error(_)          => ("x", C_RED),
                            FileStatus::PendingErrorFix(_)=> ("!", C_ORANGE),
                        };
                        let is_current = i == state.current_file_index
                            && matches!(state.app_state,
                                AppState::Processing | AppState::FixingErrors(_));
                        let removable = matches!(file.status,
                            FileStatus::Pending | FileStatus::Completed | FileStatus::Error(_));
                        let undoable = matches!(file.status, FileStatus::Completed);
                        let hover = match &file.status {
                            FileStatus::Error(msg) => Some(format!("Error: {}", msg)),
                            FileStatus::PendingErrorFix(errs) => Some(format!(
                                "{} error(s) to fix",
                                errs.iter().filter(|e| e.trim_start().starts_with("error")).count()
                            )),
                            _ => None,
                        };
                        let name = file.path.file_name().unwrap_or_default()
                            .to_string_lossy().into_owned();
                        // Truncate long names to keep the grid column tight
                        let display_name = if name.chars().count() > 23 {
                            let mut s: String = name.chars().take(16).collect();
                            s.push('…');
                            s
                        } else {
                            name.clone()
                        };
                        QueueRow {
                            name,
                            display_name,
                            badge,
                            badge_color,
                            name_color: if is_current { C_TEXT } else { C_TEXT_DIM },
                            removable,
                            undoable,
                            hover,
                        }
                    }).collect()
                };

                let mut remove_idx: Option<usize> = None;
                let mut undo_idx: Option<usize> = None;

                const BADGE_W: f32 = 10.0;
                const BTN_W:   f32 = 14.0;
                const ROW_H:   f32 = 14.0;
                const COL_GAP: f32 = 4.0;
                let col_layout = egui::Layout::left_to_right(egui::Align::Center);
                let btn_layout = egui::Layout::centered_and_justified(egui::Direction::LeftToRight);
                // Non-floating scrollbar allocates its own lane — never overlaps content.
                ui.style_mut().spacing.scroll.floating = false;

                egui::ScrollArea::vertical()
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                            // Compute name width here — inside the scroll area where
                            // available_width() already excludes the scrollbar.
                            let name_w = (ui.available_width()
                                - BADGE_W
                                - BTN_W * 2.0
                                - COL_GAP * 3.0)
                                .max(40.0);
                            egui::Grid::new("queue_grid")
                                .num_columns(4)
                                .spacing([COL_GAP, 1.0])
                                .min_col_width(0.0)
                                .show(ui, |ui| {
                                    for (i, row) in rows.iter().enumerate() {
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(BADGE_W, ROW_H), col_layout,
                                            |ui| { ui.label(egui::RichText::new(row.badge)
                                                .color(row.badge_color).size(11.0).strong()
                                                .family(egui::FontFamily::Monospace)); },
                                        );
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(name_w, ROW_H), col_layout,
                                            |ui| {
                                                let hover_text = match &row.hover {
                                                    Some(tip) => format!("{}\n{}", row.name, tip),
                                                    None => row.name.clone(),
                                                };
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new(&row.display_name)
                                                        .color(row.name_color).size(12.0)
                                                        .family(egui::FontFamily::Monospace),
                                                ).truncate()).on_hover_text(hover_text);
                                            },
                                        );
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(BTN_W, ROW_H), btn_layout,
                                            |ui| {
                                                if row.undoable {
                                                    if ui.add(egui::Button::new(
                                                        egui::RichText::new("↩").color(C_ACCENT).size(11.0),
                                                    ).frame(false).min_size(egui::vec2(0.0, 0.0)))
                                                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                                                    .on_hover_text("Restore original file")
                                                    .clicked() { undo_idx = Some(i); }
                                                }
                                            },
                                        );
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(BTN_W, ROW_H), btn_layout,
                                            |ui| {
                                                if row.removable {
                                                    if ui.add(egui::Button::new(
                                                        egui::RichText::new("×").color(C_TEXT_MUTED).size(11.0),
                                                    ).frame(false).min_size(egui::vec2(0.0, 0.0)))
                                                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                                                    .on_hover_text("Remove from queue")
                                                    .clicked() { remove_idx = Some(i); }
                                                }
                                            },
                                        );
                                        ui.end_row();
                                    }
                                });
                        });

                // Apply removal.
                if let Some(idx) = remove_idx {
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    if idx < state.files.len() {
                        state.files.remove(idx);
                        if idx < state.current_file_index && state.current_file_index > 0 {
                            state.current_file_index -= 1;
                        }
                    }
                }

                // Apply undo — restore from backup file, or fall back to original_content.
                if let Some(idx) = undo_idx {
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    if idx < state.files.len() {
                        // Compute backup path before borrowing files mutably.
                        let backup_path = self.backup_dir.as_ref()
                            .zip(state.root_path.as_ref())
                            .and_then(|(bdir, root)| {
                                state.files[idx].path.strip_prefix(root).ok()
                                    .map(|rel| bdir.join(rel))
                            });
                        let file_name = state.files[idx].path
                            .file_name().unwrap_or_default()
                            .to_string_lossy().into_owned();

                        let file = &mut state.files[idx];
                        let restored = backup_path
                            .and_then(|bk| std::fs::read_to_string(&bk).ok())
                            .unwrap_or_else(|| file.original_content.clone());

                        std::fs::write(&file.path, &restored).ok();
                        file.working_content = restored.clone();
                        file.original_content = restored;
                        file.fixes.clear();
                        file.retry_count = 0;
                        file.status = FileStatus::Pending;
                        state.push_log(format!("↩ {} restored to original", file_name));
                    }
                }
            }); // with_layout
        }); // Frame
    }

    fn render_editor(&mut self, ui: &mut egui::Ui) {
        egui::Frame::NONE
            .fill(C_SURFACE)
            .stroke(egui::Stroke::new(1.0, C_BORDER))
            .corner_radius(6)
            .show(ui, |ui| {
                // Compute display data inside a scoped lock — release before any &mut self use.
                let render_data: Option<(EditorRenderData, usize)> = {
                    let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    // Fall back to the last file if the index has advanced past the end
                    // (e.g. while cargo check is running after the final file).
                    let shown_idx = if state.files.is_empty() {
                        None
                    } else if state.current_file_index < state.files.len() {
                        Some(state.current_file_index)
                    } else {
                        Some(state.files.len() - 1)
                    };
                    shown_idx.map(|i| (build_editor_data(&state.files[i]), i))
                };

                if let Some((mut data, shown_idx)) = render_data {
                    // --- Diffusion transition between files ---
                    let file_changed = self.last_shown_file_idx
                        .map_or(false, |prev| prev != shown_idx);
                    if file_changed {
                        // Lerp scroll to top over the diffusion window so the
                        // transition feels smooth rather than snapping mid-edit.
                        self.scroll_target_y = 0.0;
                        self.transition_start = Some(Instant::now());
                    }
                    self.last_shown_file_idx = Some(shown_idx);

                    // Override display_content while the transition is running.
                    let transitioning = if let Some(start) = self.transition_start {
                        let progress = (start.elapsed().as_millis() as f32
                            / TRANSITION_MS as f32)
                            .min(1.0);
                        if progress < 1.0 {
                            data.display_content = compute_diffusion_frame(
                                &self.transition_from_content,
                                &data.display_content,
                                progress,
                            );
                            ui.ctx().request_repaint();
                            true
                        } else {
                            self.transition_start = None;
                            false
                        }
                    } else {
                        false
                    };
                    // Keep from_content up-to-date when nothing is active
                    // so the next switch dissolves from the correct frame.
                    if !transitioning {
                        self.transition_from_content = data.display_content.clone();
                    }
                    egui::Frame::NONE
                        .fill(C_RAISED)
                        .stroke(egui::Stroke::new(1.0, C_BORDER))
                        .corner_radius(egui::CornerRadius {
                            nw: 6,
                            ne: 6,
                            sw: 0,
                            se: 0,
                        })
                        .inner_margin(egui::Margin {
                            left: 12,
                            right: 12,
                            top: 6,
                            bottom: 6,
                        })
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.label(
                                egui::RichText::new(&data.header_text)
                                    .color(data.header_color)
                                    .size(11.5)
                                    .family(egui::FontFamily::Monospace),
                            );
                        });

                    // Code area — stretch dark bg to fill the remaining panel height
                    let code_area_height = ui.available_height();
                    egui::Frame::NONE
                        .fill(egui::Color32::from_rgb(10, 14, 20))
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.set_min_height(code_area_height - 24.0);
                            // Scroll-to-cursor during animation.
                            if data.is_animating {
                                let cursor_line = data
                                    .display_content
                                    .find('█')
                                    .map(|pos| {
                                        data.display_content[..pos].matches('\n').count()
                                    })
                                    .unwrap_or(0) as f32;
                                self.scroll_target_y =
                                    (cursor_line * LINE_HEIGHT_PX - 200.0).max(0.0);
                            }

                            // Lerp toward target every frame.
                            let delta = self.scroll_target_y - self.editor_scroll_y;
                            if delta.abs() > SCROLL_SNAP_PX {
                                self.editor_scroll_y += delta * SCROLL_LERP;
                                ui.ctx().request_repaint();
                            } else {
                                self.editor_scroll_y = self.scroll_target_y;
                            }

                            // Always pass our scroll position so egui's internal state
                            // never diverges from ours. Use a stable id_salt so egui
                            // doesn't lose its state between frames.
                            let mut content = data.display_content;
                            let scroll_out = egui::ScrollArea::vertical()
                                .id_salt("editor")
                                .scroll_offset(egui::Vec2::new(0.0, self.editor_scroll_y))
                                .show(ui, |ui| {
                                    ui.add(
                                        egui::TextEdit::multiline(&mut content)
                                            .font(egui::TextStyle::Monospace)
                                            .desired_width(f32::INFINITY)
                                            .code_editor()
                                            .interactive(false)
                                            .text_color(C_TEXT)
                                            .frame(false),
                                    );
                                });

                            // When not animating, follow the user's scroll so we stay
                            // in sync and don't snap when animation resumes.
                            if !data.is_animating {
                                self.editor_scroll_y = scroll_out.state.offset.y;
                                self.scroll_target_y = self.editor_scroll_y;
                            }
                        });
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new("Open a folder to begin...")
                                .color(C_TEXT_MUTED)
                                .size(13.0),
                        );
                    });
                }
            });
    }

    fn render_terminal(&mut self, ui: &mut egui::Ui) {
        egui::Frame::NONE
            .fill(C_RAISED)
            .stroke(egui::Stroke::new(1.0, C_BORDER))
            .inner_margin(egui::Margin {
                left: 12,
                right: 12,
                top: 5,
                bottom: 5,
            })
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                // Build clipboard text up front (owned) so the lock stays immutable.
                let copy_text: String = state
                    .logs
                    .iter()
                    .map(|l| {
                        if l.starts_with(ART_LINE) {
                            &l[ART_LINE.len_utf8()..]
                        } else {
                            l.as_str()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("TERMINAL")
                            .color(C_TEXT_MUTED)
                            .size(10.0)
                            .strong()
                            .family(egui::FontFamily::Monospace),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            // Copy button — rightmost item in the header bar.
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("copy")
                                            .color(C_ACCENT)
                                            .size(10.0)
                                            .family(egui::FontFamily::Monospace),
                                    )
                                    .frame(false),
                                )
                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                .on_hover_text("Copy terminal output to clipboard")
                                .clicked()
                            {
                                ui.ctx().copy_text(copy_text);
                            }
                            ui.add_space(8.0);
                            match &state.app_state {
                                AppState::Finished if !state.cargo_errors.is_empty() => {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "x  {} issue(s) remain",
                                            state
                                                .cargo_errors
                                                .iter()
                                                .filter(|l| l.trim_start().starts_with("error["))
                                                .count()
                                        ))
                                        .color(C_RED)
                                        .size(11.0)
                                        .family(egui::FontFamily::Monospace),
                                    );
                                }
                                AppState::Finished => {
                                    ui.label(
                                        egui::RichText::new("+  done")
                                            .color(C_GREEN)
                                            .size(11.0)
                                            .family(egui::FontFamily::Monospace),
                                    );
                                }
                                AppState::FixingErrors(n) => {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "~  fix round {}/{}",
                                            n, MAX_FIX_ROUNDS
                                        ))
                                        .color(C_ORANGE)
                                        .size(11.0)
                                        .family(egui::FontFamily::Monospace),
                                    );
                                }
                                _ => {}
                            }
                        },
                    );
                });
            });

        // Log body — panel fill is already the dark terminal bg
        ui.add_space(4.0);
        egui::Frame::NONE
            .inner_margin(egui::Margin {
                left: 12,
                right: 8,
                top: 0,
                bottom: 6,
            })
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        for log in &state.logs {
                            // ASCII art lines — render full-width in accent cyan, no gutter
                            if log.starts_with(ART_LINE) {
                                ui.label(
                                    egui::RichText::new(&log[ART_LINE.len_utf8()..])
                                        .color(C_ACCENT)
                                        .size(11.5)
                                        .family(egui::FontFamily::Monospace),
                                );
                                continue;
                            }

                            let (prefix, color) = if log.contains('❌') {
                                ("x ", C_RED)
                            } else if log.contains('✅') {
                                ("+ ", C_GREEN)
                            } else if log.contains('⚠') {
                                ("! ", C_ORANGE)
                            } else if log.contains('🔧') {
                                ("~ ", C_ORANGE)
                            } else if log.contains("📊") {
                                ("# ", egui::Color32::from_rgb(140, 190, 255))
                            } else if log.contains('⚡') || log.contains('📐') {
                                ("> ", C_ACCENT)
                            } else if log.starts_with("  ") {
                                ("  ", C_TEXT_DIM)
                            } else {
                                ("> ", C_TEXT_DIM)
                            };
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(prefix)
                                        .color(color.linear_multiply(0.6))
                                        .size(11.5)
                                        .family(egui::FontFamily::Monospace),
                                );
                                ui.label(
                                    egui::RichText::new(log.trim_start())
                                        .color(color)
                                        .size(11.5)
                                        .family(egui::FontFamily::Monospace),
                                );
                            });
                        }
                        if state.app_state == AppState::Finished {
                            for err in &state.cargo_errors {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("x ")
                                            .color(C_RED.linear_multiply(0.6))
                                            .size(11.5)
                                            .family(egui::FontFamily::Monospace),
                                    );
                                    ui.label(
                                        egui::RichText::new(err.trim_start())
                                            .color(egui::Color32::from_rgb(255, 130, 120))
                                            .size(11.5)
                                            .family(egui::FontFamily::Monospace),
                                    );
                                });
                            }
                        }
                    });
            });
    }

    // --- LOGIC ---

    fn close_folder(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.root_path = None;
        state.files.clear();
        state.current_file_index = 0;
        state.app_state = AppState::Idle;
        state.cargo_errors.clear();
        state.cargo_check_round = 0;
        state.push_log("📁 Folder closed.");
        drop(state);
        self.llm_cancel.store(false, Ordering::Relaxed);
        self.editor_scroll_y = 0.0;
        self.scroll_target_y = 0.0;
        self.transition_start = None;
        self.transition_from_content = String::new();
        self.last_shown_file_idx = None;
        // Wipe the backup directory for this session.
        if let Some(dir) = self.backup_dir.take() {
            std::fs::remove_dir_all(dir).ok();
        }
    }

    fn scan_folder(&mut self, path: &Path) {
        // Create a timestamped backup directory in the system temp folder.
        let backup_dir = std::env::temp_dir().join(format!(
            "optimizer_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ));
        std::fs::create_dir_all(&backup_dir).ok();
        self.backup_dir = Some(backup_dir.clone());

        let tx = self.tx.clone();
        let path = path.to_path_buf();
        let _ = tx.send(Message::SetState(AppState::Scanning));
        let _ = tx.send(Message::Log(format!("📁 Scanning {}...", path.display())));
        let _ = tx.send(Message::Log(format!("💾 Backups: {}", backup_dir.display())));
        // Use a plain OS thread — walkdir + fs::read_to_string are blocking and
        // running them inside tokio::spawn would starve the async executor.
        std::thread::spawn(move || {
            if let Ok(entries) = walkdir::WalkDir::new(&path)
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
            {
                // Accept any file that reads as valid UTF-8 text.
                let files: Vec<_> = entries
                    .iter()
                    .filter(|e| e.path().is_file())
                    .collect();
                let count = files.len();
                for entry in files {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        // Write original to backup dir before queuing.
                        if let Ok(rel) = entry.path().strip_prefix(&path) {
                            let bk = backup_dir.join(rel);
                            if let Some(parent) = bk.parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            std::fs::write(&bk, &content).ok();
                        }
                        let _ = tx.send(Message::AddFile(
                            entry.path().to_path_buf(),
                            content,
                        ));
                    }
                }
                let _ = tx.send(Message::SetState(AppState::Paused));
                let _ = tx.send(Message::Log(format!(
                    "  {} file(s) found. Press > Start.",
                    count
                )));
            }
        });
    }

    fn process_message(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                // These update self directly — no mutex needed
                Message::ModelListReceived(models) => {
                    let count = models.len();
                    let loaded = models.iter().filter(|m| m.is_loaded).count();
                    // Auto-select the currently loaded model
                    if let Some(idx) = models.iter().position(|m| m.is_loaded) {
                        self.selected_model_idx = idx;
                        if let Some(ctx) = models[idx].loaded_context {
                            self.context_length = ctx;
                        }
                    }
                    self.available_models = models;
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    state.push_log(format!(
                        "📋 {} model(s) available, {} loaded",
                        count, loaded
                    ));
                    for m in &self.available_models {
                        let detail = if m.is_loaded {
                            format!(
                                "  + {} — {}K ctx loaded",
                                m.display_name,
                                m.loaded_context.unwrap_or(0) / 1000
                            )
                        } else {
                            format!(
                                "    {} — {}K ctx max",
                                m.display_name,
                                m.max_context_length / 1000
                            )
                        };
                        state.push_log(detail);
                    }
                }

                Message::ModelLoaded(key) => {
                    for m in &mut self.available_models {
                        m.is_loaded = m.key == key;
                        if m.is_loaded {
                            m.loaded_context = Some(self.context_length);
                        }
                    }
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    // Clear the loading spinner — go to Paused if a folder is open
                    // (so Start is visible), otherwise Idle.
                    state.app_state = if state.root_path.is_some() {
                        AppState::Paused
                    } else {
                        AppState::Idle
                    };
                    state.push_log(format!("✅ Model '{}' ready", key));
                    drop(state);
                    // Re-fetch model list to get fresh instance_id for the loaded model
                    self.spawn_fetch_models();
                }

                Message::ModelUnloaded(key) => {
                    for m in &mut self.available_models {
                        if m.key == key {
                            m.is_loaded = false;
                            m.loaded_context = None;
                            m.instance_id = None;
                        }
                    }
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    state.push_log(format!("⏏  '{}' ejected", key));
                }

                // All others go through shared state
                msg => {
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    match msg {
                        Message::Log(log) => state.push_log(log),
                        Message::SetState(s) => {
                            // Ignore file-pipeline states from stale tasks if the
                            // folder has been closed since they were spawned.
                            let is_pipeline_state = matches!(
                                s,
                                AppState::Processing
                                    | AppState::FixingErrors(_)
                                    | AppState::CargoChecking
                                    | AppState::Scanning
                            );
                            if !is_pipeline_state || state.root_path.is_some() {
                                // When resuming from Paused to Processing, check if we
                                // actually have error-fix files queued — if so, restore
                                // the correct FixingErrors(round) state instead.
                                let resolved = if s == AppState::Processing
                                    && state.app_state == AppState::Paused
                                    && state.files.iter().any(|f| {
                                        matches!(f.status, FileStatus::PendingErrorFix(_))
                                    })
                                {
                                    AppState::FixingErrors(state.cargo_check_round)
                                } else {
                                    s
                                };
                                state.app_state = resolved;
                            }
                        }
                        Message::AddFile(path, content) => {
                            if state.root_path.is_none() {
                                state.root_path =
                                    path.parent().map(|p| p.to_path_buf());
                            }
                            // Normalize CRLF → LF so that LLM-returned `old` substrings
                            // (which are always LF) match the working content on Windows.
                            let content = content.replace("\r\n", "\n");
                            state.files.push(FileTask {
                                path,
                                original_content: content.clone(),
                                working_content: content,
                                fixes: Vec::new(),
                                status: FileStatus::Pending,
                                change_reason: "Awaiting analysis...".into(),
                                retry_count: 0,
                                analyzing_since: None,
                            });
                        }
                        Message::UpdateFixes(idx, mut fixes) => {
                            if let Some(file) = state.files.get_mut(idx) {
                                // Normalize line endings in LLM-returned fix strings.
                                for f in &mut fixes {
                                    f.old = f.old.replace("\r\n", "\n");
                                    f.new = f.new.replace("\r\n", "\n");
                                }
                                file.change_reason = fixes
                                    .first()
                                    .map(|f| f.explanation.clone())
                                    .unwrap_or_else(|| "No changes needed".into());
                                file.fixes = fixes;
                            }
                        }
                        Message::SetFileStatus(idx, status) => {
                            if let Some(file) = state.files.get_mut(idx) {
                                file.analyzing_since = None;
                                file.status = status.clone();
                            }
                            // As soon as a response is queued, immediately start
                            // prompting the next pending file — no waiting for
                            // animation to finish.
                            if matches!(status, FileStatus::ReadyToAnimate) {
                                let next = (idx + 1..state.files.len()).find(|&i| {
                                    matches!(
                                        state.files[i].status,
                                        FileStatus::Pending | FileStatus::PendingErrorFix(_)
                                    )
                                });
                                if let Some(next_idx) = next {
                                    let errors =
                                        if let FileStatus::PendingErrorFix(e) =
                                            &state.files[next_idx].status
                                        {
                                            Some(e.clone())
                                        } else {
                                            None
                                        };
                                    let next_content = if errors.is_some() {
                                        state.files[next_idx].working_content.clone()
                                    } else {
                                        state.files[next_idx].original_content.clone()
                                    };
                                    let next_path =
                                        state.files[next_idx].path.display().to_string();
                                    state.files[next_idx].status = FileStatus::Analyzing;
                                    state.files[next_idx].analyzing_since =
                                        Some(Instant::now());
                                    drop(state);
                                    if let Some(errs) = errors {
                                        self.spawn_llm_error_fix(
                                            next_idx, next_content, next_path, errs,
                                        );
                                    } else {
                                        self.spawn_llm_analysis(
                                            next_idx, next_content, next_path,
                                        );
                                    }
                                    continue;
                                }
                            }
                        }
                        Message::CargoCheckResult(errors) => {
                            let has_errors = errors
                                .iter()
                                .any(|l| l.trim_start().starts_with("error["));

                            if !has_errors {
                                state.push_log("✅ All cargo checks passed!");
                                state.app_state = AppState::Finished;
                            } else if state.cargo_check_round >= MAX_FIX_ROUNDS {
                                let rounds = state.cargo_check_round;
                                state.push_log(format!(
                                    "⚠️ Errors remain after {} round(s). Giving up.",
                                    rounds
                                ));
                                state.cargo_errors = errors;
                                state.app_state = AppState::Finished;
                            } else {
                                // Match errors to files and re-queue them
                                let root = state
                                    .root_path
                                    .clone()
                                    .unwrap_or_default();
                                let error_files =
                                    parse_error_files(&errors, &root);
                                let round = state.cargo_check_round + 1;
                                state.cargo_check_round = round;

                                let mut requeued = 0usize;
                                for file in &mut state.files {
                                    let is_errored =
                                        error_files.iter().any(|ep| ep == &file.path)
                                        || {
                                            // Fallback: check if relative path string appears in any error
                                            file.path
                                                .strip_prefix(&root)
                                                .ok()
                                                .and_then(|r| r.to_str())
                                                .map_or(false, |rel| {
                                                    errors.iter().any(|e| e.contains(rel))
                                                })
                                        };
                                    if is_errored {
                                        file.status =
                                            FileStatus::PendingErrorFix(errors.clone());
                                        file.retry_count = 0;
                                        requeued += 1;
                                    }
                                }

                                if requeued > 0 {
                                    state.current_file_index = 0;
                                    let currently_paused = state.app_state == AppState::Paused;
                                    if !currently_paused {
                                        state.app_state = AppState::FixingErrors(round);
                                    }
                                    state.push_log(format!(
                                        "🔧 Fix round {}/{}: re-analyzing {} file(s){}...",
                                        round, MAX_FIX_ROUNDS, requeued,
                                        if currently_paused { " (paused)" } else { "" }
                                    ));
                                } else {
                                    // Couldn't match errors to files
                                    state.cargo_errors = errors;
                                    state.app_state = AppState::Finished;
                                    state
                                        .logs
                                        .push("⚠️ Could not match errors to source files.".into());
                                }
                            }
                        }
                        // Already handled above
                        Message::ModelListReceived(_)
                        | Message::ModelLoaded(_)
                        | Message::ModelUnloaded(_) => {}
                    }
                }
            }
        }
    }

    fn run_animation_loop(&mut self, ctx: &egui::Context) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        // FixingErrors is treated as Processing for animation purposes
        let is_active = matches!(
            state.app_state,
            AppState::Processing | AppState::FixingErrors(_)
        );
        if !is_active {
            return;
        }

        // Skip already-completed files (important for the error-fix re-pass)
        while state.current_file_index < state.files.len()
            && matches!(state.files[state.current_file_index].status, FileStatus::Completed)
        {
            state.current_file_index += 1;
        }

        let idx = state.current_file_index;

        if idx >= state.files.len() {
            // Only run cargo check when Rust files were part of the scan.
            let has_rust = state.files.iter().any(|f| {
                f.path.extension().map_or(false, |x| x.eq_ignore_ascii_case("rs"))
            });
            if has_rust {
                state.app_state = AppState::CargoChecking;
                drop(state);
                self.spawn_cargo_check();
            } else {
                state.app_state = AppState::Finished;
            }
            return;
        }

        let status = state.files[idx].status.clone();

        match status {
            // --- Kick off analysis ---
            FileStatus::Pending => {
                state.files[idx].status = FileStatus::Analyzing;
                state.files[idx].analyzing_since = Some(Instant::now());
                let content = state.files[idx].original_content.clone();
                let path = state.files[idx].path.display().to_string();
                drop(state);
                self.spawn_llm_analysis(idx, content, path);
            }

            FileStatus::PendingErrorFix(errors) => {
                let errors = errors.clone();
                state.files[idx].status = FileStatus::Analyzing;
                state.files[idx].analyzing_since = Some(Instant::now());
                let content = state.files[idx].working_content.clone();
                let path = state.files[idx].path.display().to_string();
                drop(state);
                self.spawn_llm_error_fix(idx, content, path, errors);
            }

            // --- Start animating ---
            // (Next-file prompting is triggered in the SetFileStatus message handler
            // the moment this status is set, so it's already in flight by now.)
            FileStatus::ReadyToAnimate => {
                // Hold here until the diffusion transition has finished so the
                // animation always begins on a clean, fully-resolved frame.
                // Safety: force-clear if stuck beyond the transition budget.
                if let Some(ts) = self.transition_start {
                    if ts.elapsed() > Duration::from_millis(TRANSITION_MS + 500) {
                        self.transition_start = None;
                    }
                }
                if self.transition_start.is_some() {
                    ctx.request_repaint_after(Duration::from_millis(16));
                    return;
                }

                // Drop any fixes whose old text no longer exists in the file.
                let content = state.files[idx].working_content.clone();
                let fixes_before = state.files[idx].fixes.len();
                state.files[idx].fixes.retain(|f| content.contains(f.old.as_str()));
                let fixes_dropped = fixes_before - state.files[idx].fixes.len();
                if fixes_dropped > 0 {
                    state.push_log(format!(
                        "  ! {} fix(es) skipped — old text not found in file (LLM drift?)",
                        fixes_dropped
                    ));
                }

                let has_fixes = !state.files[idx].fixes.is_empty();
                if has_fixes {
                    let chars = state.files[idx].fixes[0].old.chars().count();
                    state.files[idx].status = FileStatus::Animating {
                        fix_index: 0,
                        phase: AnimPhase::Deleting(chars),
                    };
                    self.last_anim_tick = Instant::now();
                    self.scroll_target_y = 0.0;
                    self.editor_scroll_y = 0.0;
                } else {
                    state.files[idx].status = FileStatus::Completed;
                    state.current_file_index += 1;
                }

            }

            // --- Advance animation one char per tick ---
            FileStatus::Animating { fix_index, phase } => {
                let interval = match &phase {
                    AnimPhase::Deleting(_) => Duration::from_millis(DELETE_INTERVAL_MS),
                    AnimPhase::Typing(_) => Duration::from_millis(TYPE_INTERVAL_MS),
                };
                if self.last_anim_tick.elapsed() < interval {
                    ctx.request_repaint_after(interval);
                    return;
                }
                self.last_anim_tick = Instant::now();
                ctx.request_repaint_after(interval);

                let fix_count = state.files[idx].fixes.len();

                match phase {
                    AnimPhase::Deleting(left) => {
                        state.files[idx].status = FileStatus::Animating {
                            fix_index,
                            phase: if left == 0 {
                                AnimPhase::Typing(0)
                            } else {
                                AnimPhase::Deleting(left - 1)
                            },
                        };
                    }
                    AnimPhase::Typing(typed) => {
                        let total = state.files[idx]
                            .fixes
                            .get(fix_index)
                            .map_or(0, |f| f.new.chars().count());
                        if typed >= total {
                            // Commit this fix — but only if old text is still present.
                            // A prior fix in this batch may have shifted/removed it.
                            let fix = state.files[idx].fixes[fix_index].clone();
                            if state.files[idx].working_content.contains(fix.old.as_str()) {
                                let updated = state.files[idx]
                                    .working_content
                                    .replacen(&fix.old, &fix.new, 1);
                                state.files[idx].working_content = updated;
                            } else {
                                state.push_log(format!(
                                    "  ! fix {} skipped at apply time — old text no longer in file (prior fix collision?)",
                                    fix_index + 1
                                ));
                            }

                            let next_fix = fix_index + 1;
                            if next_fix < fix_count {
                                let chars =
                                    state.files[idx].fixes[next_fix].old.chars().count();
                                state.files[idx].status = FileStatus::Animating {
                                    fix_index: next_fix,
                                    phase: AnimPhase::Deleting(chars),
                                };
                            } else {
                                // All fixes done — write to disk
                                let path = state.files[idx].path.clone();
                                let final_content =
                                    state.files[idx].working_content.clone();
                                state.files[idx].status =
                                    match std::fs::write(&path, &final_content) {
                                        Ok(()) => FileStatus::Completed,
                                        Err(e) => FileStatus::Error(format!("Write failed: {}", e)),
                                    };
                                let write_ok =
                                    matches!(state.files[idx].status, FileStatus::Completed);
                                state.current_file_index += 1;
                                // Easter egg: drop a random ASCII art into the terminal
                                if write_ok {
                                    state.push_log(String::new());
                                    for line in pick_ascii_art() {
                                        state.push_log(format!("{}{}", ART_LINE, line));
                                    }
                                    state.push_log(String::new());
                                }
                            }
                        } else {
                            state.files[idx].status = FileStatus::Animating {
                                fix_index,
                                phase: AnimPhase::Typing(typed + 1),
                            };
                        }
                    }
                }
            }

            FileStatus::Error(msg) => {
                let name = state.files[idx]
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                if state.files[idx].retry_count < MAX_FILE_RETRIES {
                    state.files[idx].retry_count += 1;
                    let attempt = state.files[idx].retry_count;
                    state.push_log(format!(
                        "  ~ retrying {} ({}/{}) — {}",
                        name, attempt, MAX_FILE_RETRIES,
                        msg.chars().take(80).collect::<String>()
                    ));
                    // Clear stale fixes and reset to Pending for a fresh attempt
                    state.files[idx].fixes.clear();
                    state.files[idx].status = FileStatus::Pending;
                } else {
                    state.push_log(format!(
                        "  x skipping {} after {} failed attempt(s): {}",
                        name, MAX_FILE_RETRIES,
                        msg.chars().take(120).collect::<String>()
                    ));
                    state.current_file_index += 1;
                }
            }

            FileStatus::Analyzing => {
                // Watchdog: if the tokio task crashed silently the file would be
                // stuck here forever. Force it to Error so the retry logic unblocks.
                let timed_out = state.files[idx]
                    .analyzing_since
                    .map_or(false, |t| {
                        t.elapsed() > Duration::from_secs(TIMEOUT_ANALYZING_SECS)
                    });
                if timed_out {
                    state.files[idx].status =
                        FileStatus::Error("analysis task timed out".into());
                    state.files[idx].analyzing_since = None;
                } else {
                    ctx.request_repaint_after(Duration::from_secs(30));
                }
            }

            _ => {}
        }
    }
}

impl eframe::App for RefactorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_message();
        self.run_animation_loop(ctx);

        // Top/bottom panels must be added before side panels so they span full width.
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::NONE
                    .fill(C_SURFACE)
                    .stroke(egui::Stroke::new(1.0, C_BORDER))
                    .inner_margin(egui::Margin {
                        left: 4,
                        right: 8,
                        top: 7,
                        bottom: 7,
                    }),
            )
            .show(ctx, |ui| {
                self.render_header(ui);
            });

        egui::TopBottomPanel::bottom("terminal")
            .min_height(180.0)
            .max_height(380.0)
            .resizable(true)
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(10, 14, 20))
                    .stroke(egui::Stroke::new(1.0, C_BORDER))
                    .inner_margin(0.0),
            )
            .show(ctx, |ui| {
                self.render_terminal(ui);
            });

        // Side + central panels fill the remaining space between header and terminal.
        egui::SidePanel::left("sidebar")
            .exact_width(240.0)
            .resizable(false)
            .frame(
                egui::Frame::NONE
                    .fill(C_SURFACE)
                    .stroke(egui::Stroke::new(1.0, C_BORDER))
                    .inner_margin(0.0),
            )
            .show(ctx, |ui| {
                self.render_sidebar(ui);
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(C_BASE).inner_margin(8.0))
            .show(ctx, |ui| {
                self.render_editor(ui);
            });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let loaded = self
            .available_models
            .iter()
            .find(|m| m.is_loaded)
            .map(|m| (m.key.clone(), m.instance_id.clone()));

        // Clean up backup directory.
        if let Some(dir) = self.backup_dir.take() {
            std::fs::remove_dir_all(dir).ok();
        }

        if let Some((key, instance_id)) = loaded {
            let body = if let Some(id) = instance_id {
                serde_json::json!({ "instance_id": id })
            } else {
                serde_json::json!({ "instance_id": key })
            };
            // The main tokio runtime is still alive (rt.enter() guard in main()),
            // so we reuse its handle rather than creating a new runtime.
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let _ = handle.block_on(async {
                    if let Ok(client) = reqwest::Client::builder()
                        .timeout(Duration::from_secs(TIMEOUT_MODEL_UNLOAD_SECS))
                        .build()
                    {
                        let _ = client
                            .post(format!("{}/api/v1/models/unload", LM_STUDIO_BASE))
                            .json(&body)
                            .send()
                            .await;
                    }
                });
            }
        }
    }
}

fn main() -> eframe::Result<()> {
    let _ = env_logger::builder().format_timestamp(None).try_init();

    // eframe drives the event loop on the main thread, so we can't use
    // #[tokio::main]. Instead, build the runtime manually and call rt.enter()
    // so that tokio::spawn works from anywhere and on_exit can use block_on.
    let rt = tokio::runtime::Runtime::new().expect("Unable to create Tokio runtime");
    let _enter = rt.enter();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_min_inner_size([700.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Optimizer",
        native_options,
        Box::new(|cc| Ok(Box::new(RefactorApp::new(cc)))),
    )
}
