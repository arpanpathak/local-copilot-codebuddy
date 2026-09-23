//! Chatting with the local model: turns a conversation into a prompt, runs it
//! on the model's engine (TensorRT-LLM, or llama.cpp for GGUF models), and
//! streams the reply back as text.

use std::fs;
use std::path::Path;
use std::sync::Arc;
#[cfg(llamacpp)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokenizers::Tokenizer;

use crate::engine::{Engine, RequestId, Sampling};
use crate::event::{Event, ReplyPiece};
#[cfg(llamacpp)]
use crate::llama::{self, LlamaEngine, Progress};
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

/// The engine a model runs on.
enum Backend {
    /// A TensorRT-LLM engine, with the model's Hugging Face tokenizer.
    TensorRt { engine: Arc<Engine>, tokenizer: Arc<Tokenizer>, end_of_message_token: u32 },
    /// A GGUF model on llama.cpp, which tokenizes by itself.
    #[cfg(llamacpp)]
    Llama {
        engine: Arc<LlamaEngine>,
        /// The model thinks before answering unless its reply starts with an
        /// empty `<think></think>` block (Qwen3 and later).
        skip_thinking: bool,
    },
}

/// One loaded model, ready to chat.
pub struct ChatModel {
    name: String,
    backend: Backend,
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
            backend: Backend::TensorRt {
                engine: Arc::new(engine),
                tokenizer: Arc::new(tokenizer),
                end_of_message_token,
            },
            generation,
            limits: build.limits.within_kv_cache(kv_cache_tokens as usize),
            max_tokens: settings.max_tokens,
        })
    }

    /// Loads a GGUF model (`path` is the .gguf file) on llama.cpp.
    #[cfg(llamacpp)]
    pub fn load_gguf(path: &Path, settings: Settings) -> Result<Self> {
        let file_bytes = fs::metadata(path).with_context(|| format!("cannot read {}", path.display()))?.len();
        check_room_for_weights(file_bytes)?;
        let engine = LlamaEngine::load(path)?;

        let name = match path.file_stem() {
            Some(name) => name.to_string_lossy().into_owned(),
            None => "model".to_owned(),
        };
        let template = engine.chat_template().unwrap_or_default();
        if !template.contains("<|im_start|>") {
            bail!("{name} does not use the ChatML chat format; only Qwen-style GGUF models are supported for now");
        }
        let skip_thinking = template.contains("<think>");

        let requested = settings.kv_cache_tokens.min(engine.trained_context().max(memory::MIN_KV_CACHE_TOKENS));
        let tokens = context_that_fits(&engine, requested)?;
        let room_for_reply = (tokens / 4).min(2048) as usize;
        let limits = EngineLimits { max_input_len: tokens as usize - room_for_reply, max_seq_len: tokens as usize };
        // GGUF files carry no generation_config.json: use Qwen's recommended settings.
        let generation = GenerationConfig {
            temperature: settings.temperature.unwrap_or(0.7),
            top_p: 0.8,
            top_k: 20,
            repetition_penalty: 1.1,
        };

        Ok(Self {
            name,
            backend: Backend::Llama { engine: Arc::new(engine), skip_thinking },
            generation,
            limits,
            max_tokens: settings.max_tokens,
        })
    }

    /// The engine's name, as shown in the status bar.
    pub fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::TensorRt { .. } => "TensorRT-LLM",
            #[cfg(llamacpp)]
            Backend::Llama { .. } => "llama.cpp",
        }
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
        match &self.backend {
            Backend::TensorRt { engine, tokenizer, end_of_message_token } => {
                let encoding = tokenizer
                    .encode(chatml_prompt(messages), false)
                    .map_err(|error| anyhow!("cannot tokenize: {error}"))?;
                let prompt_tokens = encoding.get_ids();
                let sampling = Sampling {
                    max_new_tokens: self.room_for_reply(prompt_tokens.len())?,
                    temperature: self.generation.temperature,
                    top_p: self.generation.top_p,
                    top_k: self.generation.top_k,
                    repetition_penalty: self.generation.repetition_penalty,
                    random_seed: random_seed(),
                    end_token: *end_of_message_token,
                };
                let request_id = engine.start(prompt_tokens, sampling)?;

                let engine = Arc::clone(engine);
                let tokenizer = Arc::clone(tokenizer);
                let cancel = Cancel::TensorRt(Arc::clone(&engine));
                thread::spawn(move || stream_reply(&engine, &tokenizer, request_id, &events));
                Ok(ReplyStream { request_id, prompt_tokens: prompt_tokens.len(), cancel })
            }
            #[cfg(llamacpp)]
            Backend::Llama { engine, skip_thinking } => {
                let mut prompt = chatml_prompt(messages);
                if *skip_thinking {
                    prompt.push_str("<think>\n\n</think>\n\n");
                }
                let prompt_tokens = engine.tokenize(&prompt)?;
                let sampling = llama::Sampling {
                    max_new_tokens: self.room_for_reply(prompt_tokens.len())?,
                    temperature: self.generation.temperature,
                    top_p: self.generation.top_p,
                    top_k: self.generation.top_k,
                    repetition_penalty: self.generation.repetition_penalty,
                    random_seed: random_seed() as u32,
                };
                static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
                let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
                let stop = Arc::new(AtomicBool::new(false));

                let engine = Arc::clone(engine);
                let prompt_len = prompt_tokens.len();
                let cancel = Cancel::Llama(Arc::clone(&stop));
                thread::spawn(move || {
                    stream_llama_reply(&engine, &prompt_tokens, sampling, request_id, &stop, &events)
                });
                Ok(ReplyStream { request_id, prompt_tokens: prompt_len, cancel })
            }
        }
    }

    /// How many tokens the reply to a prompt of `prompt_len` tokens may have.
    fn room_for_reply(&self, prompt_len: usize) -> Result<u32> {
        if prompt_len > self.limits.max_input_len {
            bail!(
                "the conversation is {prompt_len} tokens, over this engine's {} limit; press Ctrl+L to start a new one",
                self.limits.max_input_len
            );
        }
        // Prompt and reply together must fit in the engine's sequence length.
        Ok(self.max_tokens.min((self.limits.max_seq_len - prompt_len) as u32))
    }
}

/// Refuses to load a model whose weights would leave the system short of memory.
#[cfg(llamacpp)]
fn check_room_for_weights(weight_bytes: u64) -> Result<()> {
    let Some(available) = memory::available_bytes() else {
        return Ok(()); // not Linux: nothing to measure
    };
    if weight_bytes + memory::SYSTEM_HEADROOM > available {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        bail!(
            "not enough free memory: {:.1} GB available, but the model needs {:.1} GB plus room for the rest of \
             the system. Close some applications (a web browser is often the largest) and try again.",
            gib(available),
            gib(weight_bytes),
        );
    }
    Ok(())
}

/// Creates the largest context, up to `requested` tokens, that leaves the
/// system its headroom: tries it, measures, and halves it until it fits.
#[cfg(llamacpp)]
fn context_that_fits(engine: &LlamaEngine, requested: u32) -> Result<u32> {
    let mut tokens = requested;
    loop {
        engine.set_context(tokens)?;
        let available = memory::available_bytes().unwrap_or(u64::MAX);
        if available >= memory::SYSTEM_HEADROOM {
            if tokens < requested {
                eprintln!("memory is short: context limited to {tokens} tokens instead of {requested}");
            }
            return Ok(tokens);
        }
        if tokens / 2 < memory::MIN_KV_CACHE_TOKENS {
            bail!("not enough free memory for even a {tokens}-token conversation; close some applications");
        }
        tokens /= 2;
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
    cancel: Cancel,
}

/// How to stop a reply on its engine.
enum Cancel {
    TensorRt(Arc<Engine>),
    #[cfg(llamacpp)]
    Llama(Arc<AtomicBool>),
}

impl ReplyStream {
    /// Stops generating on the GPU.
    pub fn cancel(&self) {
        match &self.cancel {
            Cancel::TensorRt(engine) => engine.cancel(self.request_id),
            #[cfg(llamacpp)]
            Cancel::Llama(stop) => stop.store(true, Ordering::Relaxed),
        }
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

/// Runs on a background thread: generates a reply on llama.cpp and sends its
/// text to the UI, until the reply is complete or `stop` is set.
#[cfg(llamacpp)]
fn stream_llama_reply(
    engine: &LlamaEngine,
    prompt: &[u32],
    sampling: llama::Sampling,
    request_id: RequestId,
    stop: &AtomicBool,
    events: &Sender<Event>,
) {
    let mut pending = Vec::new();
    let result = engine.generate(prompt, sampling, |progress| {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        let Progress::Token(bytes) = progress else {
            return true; // still reading the prompt
        };
        pending.extend_from_slice(bytes);
        let piece = ReplyPiece::Text { text: take_complete_utf8(&mut pending), token_count: 1 };
        events.send(Event::Reply { request_id, piece }).is_ok() // stops once the UI has exited
    });
    let piece = match result {
        Ok(()) => ReplyPiece::Finished,
        Err(error) => ReplyPiece::Failed(format!("{error:#}")),
    };
    let _ = events.send(Event::Reply { request_id, piece });
}

/// Removes and returns the complete UTF-8 text at the start of `pending`,
/// leaving the first bytes of a character that is still being generated.
#[cfg(any(llamacpp, test))]
fn take_complete_utf8(pending: &mut Vec<u8>) -> String {
    let complete = match std::str::from_utf8(pending) {
        Ok(_) => pending.len(),
        Err(error) if error.error_len().is_none() => error.valid_up_to(), // the rest may arrive with the next token
        Err(_) => pending.len(), // invalid bytes: show them as replacement characters
    };
    let text = String::from_utf8_lossy(&pending[..complete]).into_owned();
    pending.drain(..complete);
    text
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
    fn characters_split_across_tokens_are_held_back_until_complete() {
        let crab = "🦀".as_bytes();
        let mut pending = b"hi ".to_vec();
        pending.extend_from_slice(&crab[..2]);
        assert_eq!(take_complete_utf8(&mut pending), "hi ");
        pending.extend_from_slice(&crab[2..]);
        assert_eq!(take_complete_utf8(&mut pending), "🦀");
        assert!(pending.is_empty());
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
