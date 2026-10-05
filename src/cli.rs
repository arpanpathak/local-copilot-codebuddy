//! Command-line interface.

use std::path::PathBuf;

use clap::Parser;

/// Terminal chat with an LLM running locally on TensorRT-LLM.
#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Cli {
    /// The model: a Hugging Face model directory with a TensorRT-LLM engine
    /// next to it, or a .gguf file (llama.cpp). Default: choose from the
    /// models in ~/models.
    #[arg(env = "CODEBUDDY_MODEL")]
    pub model: Option<PathBuf>,

    /// TensorRT engine directory. Default: `<MODEL_DIR>-engine`, where
    /// engine/build-engine.sh puts it.
    #[arg(short, long, env = "CODEBUDDY_ENGINE")]
    pub engine: Option<PathBuf>,

    /// System prompt that starts every conversation. None by default: the
    /// model gets only what you give here and in the rules file.
    #[arg(short, long, env = "CODEBUDDY_SYSTEM")]
    pub system: Option<String>,

    /// Your coding rules (a Markdown file), added to the system prompt of
    /// every conversation so the model keeps following them.
    /// Default: ~/.config/local-copilot-codebuddy/rules.md, if it exists.
    #[arg(short, long, env = "CODEBUDDY_RULES")]
    pub rules: Option<PathBuf>,

    /// Sampling temperature (0 = greedy). Default: the model's recommendation
    /// from its generation_config.json (0.7 for Qwen2.5-Coder).
    #[arg(short, long)]
    pub temperature: Option<f32>,

    /// Maximum number of tokens generated per reply.
    #[arg(long, default_value_t = 8192)]
    pub max_tokens: u32,

    /// KV cache size in tokens: how much conversation fits in memory.
    /// Default: a GGUF model's full trained context, and 32768 for a
    /// TensorRT-LLM engine. Lowered automatically when free memory is short,
    /// so the system never swaps.
    #[arg(long)]
    pub kv_cache_tokens: Option<u32>,
}

impl Cli {
    /// The system prompt: `--system` and the rules file, each used exactly as
    /// written, or `None` when there is neither.
    pub fn system_prompt(&self) -> anyhow::Result<Option<String>> {
        let (path, required) = match &self.rules {
            Some(path) => (path.clone(), true),
            None => {
                let home = std::env::var_os("HOME").unwrap_or_default();
                (PathBuf::from(home).join(".config/local-copilot-codebuddy/rules.md"), false)
            }
        };
        let rules = match std::fs::read_to_string(&path) {
            Ok(rules) => Some(rules.trim().to_string()),
            Err(_) if !required => None,
            Err(error) => anyhow::bail!("cannot read the rules file {}: {error}", path.display()),
        };
        let parts: Vec<String> =
            [self.system.clone(), rules].into_iter().flatten().filter(|part| !part.trim().is_empty()).collect();
        Ok((!parts.is_empty()).then(|| parts.join("\n\n")))
    }

    /// The engine directory, falling back to `<model_dir>-engine`.
    pub fn engine_dir(&self, model_dir: &std::path::Path) -> PathBuf {
        match &self.engine {
            Some(engine) => engine.clone(),
            None => {
                let mut engine_dir = model_dir.as_os_str().to_owned();
                engine_dir.push("-engine");
                PathBuf::from(engine_dir)
            }
        }
    }
}
