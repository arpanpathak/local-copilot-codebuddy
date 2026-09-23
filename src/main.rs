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
#[cfg(llamacpp)]
mod llama;
mod markdown;
mod memory;
mod picker;
mod ui;

use std::io;
use std::path::{Path, PathBuf};
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

/// Lets the user pick one of the models under ~/models; `None` if they quit.
fn choose_model() -> Result<Option<PathBuf>> {
    let choices = picker::discover();
    match choices.as_slice() {
        [] => anyhow::bail!(
            "no models found in ~/models: build one with engine/build-engine.sh, put a .gguf file in \
             ~/models/gguf, or pass a model path"
        ),
        [only] => Ok(Some(only.path.clone())),
        _ => Ok(picker::choose(&choices)?.map(|choice| choice.path.clone())),
    }
}

#[cfg(llamacpp)]
fn load_gguf(path: &Path, settings: Settings) -> Result<ChatModel> {
    ChatModel::load_gguf(path, settings)
}

#[cfg(not(llamacpp))]
fn load_gguf(_path: &Path, _settings: Settings) -> Result<ChatModel> {
    anyhow::bail!("this build cannot run GGUF models: run engine/install-llama.sh, then reinstall")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let system_prompt = cli.system_prompt()?;
    let settings =
        Settings { temperature: cli.temperature, max_tokens: cli.max_tokens, kv_cache_tokens: cli.kv_cache_tokens };
    let model_path = match &cli.model {
        Some(path) => path.clone(),
        None => match choose_model()? {
            Some(path) => path,
            None => return Ok(()),
        },
    };

    eprintln!("loading {} ...", model_path.display());
    let model = if model_path.extension().is_some_and(|extension| extension == "gguf") {
        load_gguf(&model_path, settings)?
    } else {
        ChatModel::load(&model_path, &cli.engine_dir(&model_path), settings)?
    };

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
