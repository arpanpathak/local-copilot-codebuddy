//! local-copilot-codebuddy: a terminal coding assistant powered by an LLM that
//! runs locally on an NVIDIA Jetson. The TensorRT-LLM engine runs inside this
//! process; there is no server in between.

mod app;
mod chat;
mod cli;
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

use crate::app::App;
use crate::chat::{ChatModel, Settings};
use crate::cli::Cli;
use crate::highlight::Highlighter;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model_dir = cli.model_dir();
    let engine_dir = cli.engine_dir(&model_dir);
    let settings =
        Settings { temperature: cli.temperature, max_tokens: cli.max_tokens, kv_cache_tokens: cli.kv_cache_tokens };

    eprintln!("loading {} ...", engine_dir.display());
    let model = ChatModel::load(&model_dir, &engine_dir, settings)?;

    let (event_sender, event_receiver) = mpsc::channel();
    event::spawn_terminal_reader(event_sender.clone());
    let app = App::new(model, Highlighter::new(), Some(cli.system), event_sender);

    // ratatui::run sets up the terminal and restores it afterwards, even on panic.
    ratatui::run(|terminal| {
        execute!(io::stdout(), EnableBracketedPaste)?; // pasted text arrives as one event
        let result = app.run(terminal, &event_receiver);
        execute!(io::stdout(), DisableBracketedPaste)?;
        result
    })
}
