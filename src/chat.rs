//! Chatting with the local model: turns a conversation into a prompt, runs it
//! on the TensorRT-LLM engine, and streams the reply back as text.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokenizers::Tokenizer;

use crate::engine::{Engine, RequestId, Sampling};
use crate::event::{Event, ReplyPiece};
use crate::memory;

/// Qwen's chat format (ChatML) marks the end of every message with this token.
const END_OF_MESSAGE: &str = "<|im_end|>";

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One message of the conversation.
#[derive(Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Settings chosen on the command line.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Overrides the model's recommended temperature when set.
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    pub kv_cache_tokens: u32,
}

/// The model's recommended sampling settings, from its `generation_config.json`.
#[derive(Debug, Clone, Copy, Deserialize)]
struct GenerationConfig {
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    top_k: u32,
    #[serde(default = "no_repetition_penalty")]
    repetition_penalty: f32,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.8
}

fn no_repetition_penalty() -> f32 {
    1.0
}

/// The limits the engine was built with (`build_config` in its `config.json`).
#[derive(Debug, Clone, Copy, Deserialize)]
struct EngineLimits {
    /// Longest prompt, conversation history included.
    max_input_len: usize,
    /// Longest prompt plus reply.
    max_seq_len: usize,
}

impl EngineLimits {
    /// The limits once the KV cache holds only `kv_cache_tokens`: the whole
    /// sequence must fit in it, still leaving the engine's room for a reply.
    fn within_kv_cache(self, kv_cache_tokens: usize) -> Self {
        let room_for_reply = self.max_seq_len - self.max_input_len;
        let max_seq_len = self.max_seq_len.min(kv_cache_tokens);
        Self { max_input_len: self.max_input_len.min(max_seq_len.saturating_sub(room_for_reply)), max_seq_len }
    }
}

/// The parts of the engine's `config.json` that local-copilot-codebuddy needs.
#[derive(Deserialize)]
struct EngineConfig {
    pretrained_config: ModelShape,
    build_config: EngineBuildConfig,
}

/// The model dimensions that decide how much memory each token's KV cache takes.
#[derive(Deserialize)]
struct ModelShape {
    num_hidden_layers: u64,
    num_key_value_heads: u64,
    head_size: u64,
    #[serde(default)]
    quantization: KvCacheQuantization,
}

#[derive(Default, Deserialize)]
struct KvCacheQuantization {
    /// `INT8` or `FP8` when the KV cache is quantized; fp16 otherwise.
    kv_cache_quant_algo: Option<String>,
}

impl ModelShape {
    /// Bytes of KV cache per token: a key and a value in every layer.
    fn kv_bytes_per_token(&self) -> u64 {
        let bytes_per_value = if self.quantization.kv_cache_quant_algo.is_some() { 1 } else { 2 };
        2 * self.num_hidden_layers * self.num_key_value_heads * self.head_size * bytes_per_value
    }
}

#[derive(Deserialize)]
struct EngineBuildConfig {
    #[serde(flatten)]
    limits: EngineLimits,
    plugin_config: EnginePlugins,
}

#[derive(Deserialize)]
struct EnginePlugins {
    /// Built with `--use_paged_context_fmha`: chunked prefill and KV-cache reuse work.
    use_paged_context_fmha: bool,
}

/// The engine and tokenizer of one model.
pub struct ChatModel {
    name: String,
    engine: Arc<Engine>,
    tokenizer: Arc<Tokenizer>,
    end_of_message_token: u32,
    generation: GenerationConfig,
    limits: EngineLimits,
    max_tokens: u32,
}

impl ChatModel {
    /// Loads the tokenizer and sampling settings from `model_dir` (the Hugging
    /// Face download) and the TensorRT engine from `engine_dir`.
    pub fn load(model_dir: &Path, engine_dir: &Path, settings: Settings) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow!("cannot load {}: {error}", tokenizer_path.display()))?;
        let end_of_message_token = tokenizer
            .token_to_id(END_OF_MESSAGE)
            .with_context(|| format!("{END_OF_MESSAGE} is not in the tokenizer; is this a Qwen chat model?"))?;

        let mut generation: GenerationConfig = read_json(&model_dir.join("generation_config.json"))?;
        if let Some(temperature) = settings.temperature {
            generation.temperature = temperature;
        }
        let engine_config: EngineConfig = read_json(&engine_dir.join("config.json"))?;

        let build = engine_config.build_config;
        let kv_cache_tokens =
            kv_cache_tokens_that_fit(settings.kv_cache_tokens, engine_dir, &engine_config.pretrained_config)?;
        let engine = Engine::load(engine_dir, kv_cache_tokens, build.plugin_config.use_paged_context_fmha)?;
        let name = match model_dir.file_name() {
            Some(name) => name.to_string_lossy().into_owned(),
            None => "model".to_owned(),
        };

        Ok(Self {
            name,
            engine: Arc::new(engine),
            tokenizer: Arc::new(tokenizer),
            end_of_message_token,
            generation,
            limits: build.limits.within_kv_cache(kv_cache_tokens as usize),
            max_tokens: settings.max_tokens,
        })
    }

    /// The model's name, as shown in the status bar.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The most tokens a conversation can have (the engine's prompt limit).
    pub fn context_limit(&self) -> usize {
        self.limits.max_input_len
    }

    /// Starts generating the assistant's reply to `messages`. The reply is
    /// streamed to `events` as [`Event::Reply`] pieces tagged with its request id.
    pub fn start_reply(&self, messages: &[Message], events: Sender<Event>) -> Result<ReplyStream> {
        let prompt = chatml_prompt(messages);
        let encoding = self.tokenizer.encode(prompt, false).map_err(|error| anyhow!("cannot tokenize: {error}"))?;
        let prompt_tokens = encoding.get_ids();

        if prompt_tokens.len() > self.limits.max_input_len {
            bail!(
                "the conversation is {} tokens, over this engine's {} limit; press Ctrl+L to start a new one",
                prompt_tokens.len(),
                self.limits.max_input_len
            );
        }
        // Prompt and reply together must fit in the engine's sequence length.
        let room_for_reply = self.limits.max_seq_len - prompt_tokens.len();

        let sampling = Sampling {
            max_new_tokens: self.max_tokens.min(room_for_reply as u32),
            temperature: self.generation.temperature,
            top_p: self.generation.top_p,
            top_k: self.generation.top_k,
            repetition_penalty: self.generation.repetition_penalty,
            random_seed: random_seed(),
            end_token: self.end_of_message_token,
        };
        let request_id = self.engine.start(prompt_tokens, sampling)?;

        let engine = Arc::clone(&self.engine);
        let tokenizer = Arc::clone(&self.tokenizer);
        thread::spawn(move || stream_reply(&engine, &tokenizer, request_id, &events));

        Ok(ReplyStream { request_id, prompt_tokens: prompt_tokens.len(), engine: Arc::clone(&self.engine) })
    }
}

/// How many tokens of KV cache to reserve: `requested`, or fewer if that would
/// leave the rest of the system short of memory.
fn kv_cache_tokens_that_fit(requested: u32, engine_dir: &Path, shape: &ModelShape) -> Result<u32> {
    let Some(available) = memory::available_bytes() else {
        return Ok(requested); // not Linux: nothing to measure
    };
    let engine_bytes = engine_file_bytes(engine_dir)?;
    let tokens = memory::kv_cache_tokens_that_fit(requested, available, engine_bytes, shape.kv_bytes_per_token());
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;

    if tokens < memory::MIN_KV_CACHE_TOKENS.min(requested) {
        bail!(
            "not enough free memory: {:.1} GB available, but the model needs about {:.1} GB plus room for \
             the rest of the system. Close some applications (a web browser is often the largest) and try again.",
            gib(available),
            gib(engine_bytes + u64::from(memory::MIN_KV_CACHE_TOKENS) * shape.kv_bytes_per_token()),
        );
    }
    if tokens < requested {
        eprintln!(
            "only {:.1} GB of memory is free: context limited to {tokens} tokens instead of {requested} \
             to keep the system responsive",
            gib(available),
        );
    }
    Ok(tokens)
}

/// Total size of the engine files in `engine_dir`: about what the weights take in GPU memory.
fn engine_file_bytes(engine_dir: &Path) -> Result<u64> {
    let mut total = 0;
    for entry in fs::read_dir(engine_dir).with_context(|| format!("cannot read {}", engine_dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "engine") {
            total += fs::metadata(&path)?.len();
        }
    }
    Ok(total)
}

/// Reads and parses a JSON file.
fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

/// A different seed for every request, so asking again gives a fresh answer.
fn random_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(since_epoch) => since_epoch.as_nanos() as u64,
        Err(_) => 0,
    }
}

/// Handle to a reply that is being generated.
pub struct ReplyStream {
    /// Identifies the events that belong to this reply.
    pub request_id: RequestId,
    /// Length of the prompt (the whole conversation so far), in tokens.
    pub prompt_tokens: usize,
    engine: Arc<Engine>,
}

impl ReplyStream {
    /// Stops generating on the GPU.
    pub fn cancel(&self) {
        self.engine.cancel(self.request_id);
    }
}

/// Runs on a background thread: collects tokens from the engine, turns them
/// into text, and sends it to the UI until the reply is complete.
fn stream_reply(engine: &Engine, tokenizer: &Tokenizer, request_id: RequestId, events: &Sender<Event>) {
    let mut decoder = TextDecoder::default();
    let mut new_tokens = Vec::new();

    loop {
        new_tokens.clear();
        let finished = match engine.wait_for_tokens(request_id, &mut new_tokens) {
            Ok(finished) => finished,
            Err(error) => {
                let _ = events.send(Event::Reply { request_id, piece: ReplyPiece::Failed(format!("{error:#}")) });
                return;
            }
        };

        let mut text = String::new();
        for &token in &new_tokens {
            if let Some(piece) = decoder.push(tokenizer, token) {
                text.push_str(&piece);
            }
        }
        let piece = ReplyPiece::Text { text, token_count: new_tokens.len() as u32 };
        if events.send(Event::Reply { request_id, piece }).is_err() {
            engine.cancel(request_id); // the UI has exited
            return;
        }

        if finished {
            let _ = events.send(Event::Reply { request_id, piece: ReplyPiece::Finished });
            return;
        }
    }
}

/// Formats the conversation in ChatML, Qwen's chat format, ending with an
/// open assistant turn for the model to complete:
///
/// ```text
/// <|im_start|>user
/// Hello<|im_end|>
/// <|im_start|>assistant
/// ```
fn chatml_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Turns a stream of token ids into text, one piece at a time.
///
/// A single token can hold part of a multi-byte character, and how a token
/// decodes can depend on the one before it. So each new token is decoded
/// together with a few previous ones, and only complete new text is returned.
#[derive(Default)]
struct TextDecoder {
    tokens: Vec<u32>,
    /// Start of the tokens re-decoded for context.
    context_start: usize,
    /// End of the tokens whose text was already returned.
    returned_end: usize,
}

impl TextDecoder {
    /// Adds a token; returns the new text if it completes any.
    fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> Option<String> {
        self.tokens.push(token);

        let already_returned = tokenizer.decode(&self.tokens[self.context_start..self.returned_end], true).ok()?;
        let with_new_token = tokenizer.decode(&self.tokens[self.context_start..], true).ok()?;

        let is_complete = with_new_token.len() > already_returned.len() && !with_new_token.ends_with('\u{FFFD}');
        if !is_complete {
            return None; // wait for the rest of the character
        }

        self.context_start = self.returned_end;
        self.returned_end = self.tokens.len();
        Some(with_new_token[already_returned.len()..].to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_prompt_ends_with_an_open_assistant_turn() {
        let messages = [
            Message { role: Role::System, content: "Be brief.".to_owned() },
            Message { role: Role::User, content: "Hi".to_owned() },
        ];
        let expected =
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(chatml_prompt(&messages), expected);
    }

    #[test]
    fn a_smaller_kv_cache_keeps_room_for_a_reply() {
        let built = EngineLimits { max_input_len: 30720, max_seq_len: 32768 };
        let limits = built.within_kv_cache(16384);
        assert_eq!((limits.max_input_len, limits.max_seq_len), (14336, 16384));
        let limits = built.within_kv_cache(32768);
        assert_eq!((limits.max_input_len, limits.max_seq_len), (30720, 32768));
    }

    #[test]
    fn streamed_text_matches_one_shot_decoding() {
        let home = std::env::var("HOME").unwrap_or_default();
        let tokenizer_path = format!("{home}/models/trt/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4/tokenizer.json");
        let Ok(tokenizer) = Tokenizer::from_file(&tokenizer_path) else {
            eprintln!("skipped: no tokenizer at {tokenizer_path}");
            return;
        };

        let text = "fn main() { println!(\"héllo, 世界 🦀\"); }";
        let tokens = tokenizer.encode(text, false).unwrap().get_ids().to_vec();

        let mut decoder = TextDecoder::default();
        let mut streamed = String::new();
        for token in tokens {
            if let Some(piece) = decoder.push(&tokenizer, token) {
                streamed.push_str(&piece);
            }
        }
        assert_eq!(streamed, text);
    }
}
