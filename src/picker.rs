//! Choosing a model at startup: lists the models under `~/models` and lets
//! the user pick one with the arrow keys.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Flex, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

const ACCENT: Color = Color::Rgb(137, 180, 250);
const MUTED: Color = Color::Rgb(108, 112, 134);

/// A model that can be loaded.
pub struct ModelChoice {
    pub name: String,
    pub engine: &'static str,
    /// A Hugging Face model directory (TensorRT-LLM) or a .gguf file (llama.cpp).
    pub path: PathBuf,
    pub bytes: u64,
}

/// The models found under `~/models`: TensorRT-LLM engines built by
/// engine/build-engine.sh in `trt/`, and GGUF files in `gguf/`.
pub fn discover() -> Vec<ModelChoice> {
    let models = PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("models");
    let mut choices = tensorrt_models(&models.join("trt"));
    if cfg!(llamacpp) {
        choices.extend(gguf_models(&models.join("gguf")));
    }
    choices.sort_by(|a, b| a.name.cmp(&b.name));
    choices
}

/// `<name>-engine/` directories next to their Hugging Face model directory `<name>/`.
fn tensorrt_models(dir: &Path) -> Vec<ModelChoice> {
    let mut choices = Vec::new();
    for engine_dir in entries(dir) {
        let Some(name) = engine_dir.file_name().and_then(|name| name.to_str()?.strip_suffix("-engine")) else {
            continue;
        };
        let model_dir = dir.join(name);
        let engines: Vec<PathBuf> = entries(&engine_dir)
            .into_iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "engine"))
            .collect();
        if engines.is_empty() || !model_dir.join("tokenizer.json").exists() {
            continue;
        }
        let bytes = engines.iter().filter_map(|path| fs::metadata(path).ok()).map(|meta| meta.len()).sum();
        choices.push(ModelChoice { name: name.to_owned(), engine: "TensorRT-LLM", path: model_dir, bytes });
    }
    choices
}

/// .gguf files in `dir` and its subdirectories (vision projectors excluded).
fn gguf_models(dir: &Path) -> Vec<ModelChoice> {
    let mut choices = Vec::new();
    for path in entries(dir).into_iter().flat_map(|path| if path.is_dir() { entries(&path) } else { vec![path] }) {
        let Some(name) = path.file_stem().and_then(|name| name.to_str()) else { continue };
        if path.extension().is_none_or(|ext| ext != "gguf") || name.starts_with("mmproj") {
            continue;
        }
        let bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        choices.push(ModelChoice { name: name.to_owned(), engine: "llama.cpp", path, bytes });
    }
    choices
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    match fs::read_dir(dir) {
        Ok(entries) => entries.filter_map(|entry| Some(entry.ok()?.path())).collect(),
        Err(_) => Vec::new(),
    }
}

/// Shows the models and returns the chosen one, or `None` if the user quit.
pub fn choose(choices: &[ModelChoice]) -> Result<Option<&ModelChoice>> {
    let mut selected = 0;
    ratatui::run(|terminal| {
        loop {
            terminal.draw(|frame| {
                let name_width = choices.iter().map(|choice| choice.name.len()).max().unwrap_or(0);
                let mut lines =
                    vec![Line::styled("Choose a model", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))];
                lines.push(Line::default());
                for (index, choice) in choices.iter().enumerate() {
                    let is_selected = index == selected;
                    let style = if is_selected { Style::new().add_modifier(Modifier::BOLD) } else { Style::new() };
                    lines.push(Line::from(vec![
                        Span::styled(if is_selected { "❯ " } else { "  " }, ACCENT),
                        Span::styled(format!("{:name_width$}   ", choice.name), style),
                        Span::styled(format!("{:12}", choice.engine), MUTED),
                        Span::styled(format!("{:>5.1} GB", choice.bytes as f64 / 1e9), MUTED),
                    ]));
                }
                lines.push(Line::default());
                lines.push(Line::styled("↑↓ select · enter load · esc quit", MUTED));

                let width = lines.iter().map(Line::width).max().unwrap_or(0) as u16;
                let [column] = Layout::horizontal([Constraint::Length(width)]).flex(Flex::Center).areas(frame.area());
                let [area] =
                    Layout::vertical([Constraint::Length(lines.len() as u16)]).flex(Flex::Center).areas(column);
                frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
            })?;

            let Event::Key(key) = event::read()? else { continue };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match (key.code, key.modifiers) {
                (KeyCode::Up, _) => selected = selected.saturating_sub(1),
                (KeyCode::Down, _) => selected = (selected + 1).min(choices.len() - 1),
                (KeyCode::Enter, _) => return Ok(Some(&choices[selected])),
                (KeyCode::Esc, _) | (KeyCode::Char('c' | 'd'), KeyModifiers::CONTROL) => return Ok(None),
                _ => {}
            }
        }
    })
}
