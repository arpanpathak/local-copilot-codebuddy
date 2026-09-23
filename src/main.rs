//! local-copilot-codebuddy: a terminal coding assistant powered by an LLM that
//! runs locally on an NVIDIA Jetson. The TensorRT-LLM engine runs inside this
//! process; there is no server in between.

mod app;
mod chat;
mod cli;
mod clipboard;
mod engine;
mod event;
mod highlight;
mod markdown;
mod memory;
mod ui;

use std::io;
use std::sync::mpsc;

use anyhow::Result;
use clap::Parser;
use ratatui::crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use ratatui::crossterm::execute;
use ratatui::crossterm::style::Print;

use crate::app::App;
use crate::chat::{ChatModel, Settings};
use crate::cli::Cli;
use crate::highlight::Highlighter;

/// Reports mouse clicks and the wheel (in SGR form), but not mouse movement,
/// which would wake the UI for every pixel the pointer crosses.
const MOUSE_CLICKS_ON: &str = "\x1b[?1000h\x1b[?1006h";
const MOUSE_CLICKS_OFF: &str = "\x1b[?1006l\x1b[?1000l";

fn main() -> Result<()> {
    let cli = Cli::parse();
    let system_prompt = cli.system_prompt()?;
    let model_dir = cli.model_dir();
    let engine_dir = cli.engine_dir(&model_dir);
    let settings =
        Settings { temperature: cli.temperature, max_tokens: cli.max_tokens, kv_cache_tokens: cli.kv_cache_tokens };

    eprintln!("loading {} ...", engine_dir.display());
    let model = ChatModel::load(&model_dir, &engine_dir, settings)?;

    let (event_sender, event_receiver) = mpsc::channel();
    event::spawn_terminal_reader(event_sender.clone());
    let app = App::new(model, Highlighter::new(), Some(system_prompt), event_sender);

    // ratatui::run sets up the terminal and restores it afterwards, even on panic.
    ratatui::run(|terminal| {
        execute!(io::stdout(), EnableBracketedPaste, Print(MOUSE_CLICKS_ON))?; // pasted text arrives as one event
        let result = app.run(terminal, &event_receiver);
        execute!(io::stdout(), Print(MOUSE_CLICKS_OFF), DisableBracketedPaste)?;
        result
    })
}
