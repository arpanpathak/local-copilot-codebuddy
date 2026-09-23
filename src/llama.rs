//! Safe Rust wrapper around llama.cpp (see `cpp/llama_shim.cpp`), the engine
//! for GGUF models.

use std::ffi::{CStr, CString, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use anyhow::{Result, bail};

/// Generation settings for one reply.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub max_new_tokens: u32,
    /// 0 means greedy decoding.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 means no top-k limit.
    pub top_k: u32,
    /// 1.0 means no penalty.
    pub repetition_penalty: f32,
    pub random_seed: u32,
}

/// What the engine reports while it works on a reply.
pub enum Progress<'a> {
    /// Still reading the prompt.
    Reading,
    /// Generated a token; `bytes` is its text (possibly part of a UTF-8 character).
    Token(&'a [u8]),
}

/// A GGUF model loaded on the GPU with llama.cpp.
pub struct LlamaEngine {
    raw: Mutex<NonNull<ffi::EdgeLlama>>,
}

// Every call goes through the mutex, so one thread uses the engine at a time.
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Loads the model in `path` (a .gguf file) with all layers on the GPU.
    /// Call [`LlamaEngine::set_context`] before generating.
    pub fn load(path: &Path) -> Result<Self> {
        let path = CString::new(path.as_os_str().as_bytes())?;
        let mut error = ErrorBuffer::new();
        let raw = unsafe { ffi::edge_llama_open(path.as_ptr(), error.as_mut_ptr(), error.len()) };
        match NonNull::new(raw) {
            Some(raw) => Ok(Self { raw: Mutex::new(raw) }),
            None => bail!("cannot load the model: {}", error.message()),
        }
    }

    /// Reserves memory for a conversation of `tokens` tokens, replacing any earlier context.
    pub fn set_context(&self, tokens: u32) -> Result<()> {
        let raw = self.lock();
        let mut error = ErrorBuffer::new();
        if !unsafe { ffi::edge_llama_set_context(raw.as_ptr(), tokens, error.as_mut_ptr(), error.len()) } {
            bail!("{}", error.message());
        }
        Ok(())
    }

    /// The context length the model was trained for.
    pub fn trained_context(&self) -> u32 {
        unsafe { ffi::edge_llama_n_ctx_train(self.lock().as_ptr()) }.max(0) as u32
    }

    /// The model's chat template, if it has one.
    pub fn chat_template(&self) -> Option<String> {
        let template = unsafe { ffi::edge_llama_chat_template(self.lock().as_ptr()) };
        if template.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(template) }.to_string_lossy().into_owned())
    }

    /// Turns text into token ids; special tokens such as `<|im_start|>` are recognised.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let raw = self.lock();
        let mut tokens = vec![0i32; text.len() + 16];
        let count = unsafe {
            ffi::edge_llama_tokenize(
                raw.as_ptr(),
                text.as_ptr().cast(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
            )
        };
        if count < 0 {
            bail!("cannot tokenize the conversation");
        }
        tokens.truncate(count as usize);
        Ok(tokens.into_iter().map(|token| token as u32).collect())
    }

    /// Generates a reply to `prompt`, calling `on_progress` while reading the
    /// prompt and for each generated token. Returning `false` from it stops.
    pub fn generate<F: FnMut(Progress) -> bool>(
        &self,
        prompt: &[u32],
        sampling: Sampling,
        on_progress: F,
    ) -> Result<()> {
        /// What the C++ side hands back to `trampoline`.
        struct Callback<F> {
            raw: NonNull<ffi::EdgeLlama>,
            on_progress: F,
            piece: Vec<u8>,
        }

        extern "C" fn trampoline<F: FnMut(Progress) -> bool>(context: *mut c_void, token: i32) -> bool {
            let callback = unsafe { &mut *context.cast::<Callback<F>>() };
            if token == -1 {
                return (callback.on_progress)(Progress::Reading);
            }
            // The engine is locked by `generate` on this thread, so the handle can be used directly.
            token_piece(callback.raw, token, &mut callback.piece);
            (callback.on_progress)(Progress::Token(&callback.piece))
        }

        let raw = self.lock();
        let mut callback = Callback { raw: *raw, on_progress, piece: Vec::with_capacity(64) };
        let sampling = ffi::EdgeLlamaSampling {
            max_new_tokens: sampling.max_new_tokens as i32,
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k as i32,
            repetition_penalty: sampling.repetition_penalty,
            random_seed: sampling.random_seed,
        };
        let mut error = ErrorBuffer::new();
        // Token ids are below 2^31, so u32 and i32 share the same bits: pass the slice as-is.
        let succeeded = unsafe {
            ffi::edge_llama_generate(
                raw.as_ptr(),
                prompt.as_ptr().cast(),
                prompt.len(),
                sampling,
                trampoline::<F>,
                (&mut callback as *mut Callback<F>).cast(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if !succeeded {
            bail!("generation failed: {}", error.message());
        }
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, NonNull<ffi::EdgeLlama>> {
        self.raw.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        unsafe { ffi::edge_llama_close(self.lock().as_ptr()) }
    }
}

/// Writes the bytes `token` stands for into `piece`.
fn token_piece(raw: NonNull<ffi::EdgeLlama>, token: i32, piece: &mut Vec<u8>) {
    piece.resize(piece.capacity().max(64), 0);
    loop {
        let length =
            unsafe { ffi::edge_llama_token_piece(raw.as_ptr(), token, piece.as_mut_ptr().cast(), piece.len() as i32) };
        if length >= 0 {
            piece.truncate(length as usize);
            return;
        }
        piece.resize(length.unsigned_abs() as usize, 0);
    }
}

/// Receives error messages from the C++ side.
struct ErrorBuffer([u8; 512]);

impl ErrorBuffer {
    fn new() -> Self {
        Self([0; 512])
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr().cast()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn message(&self) -> String {
        match CStr::from_bytes_until_nul(&self.0) {
            Ok(message) => message.to_string_lossy().into_owned(),
            Err(_) => "unknown error".to_owned(),
        }
    }
}

/// Declarations matching `cpp/llama_shim.cpp`.
mod ffi {
    use std::ffi::{c_char, c_void};

    /// Opaque handle to the model and its context.
    #[repr(C)]
    pub struct EdgeLlama {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct EdgeLlamaSampling {
        pub max_new_tokens: i32,
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: i32,
        pub repetition_penalty: f32,
        pub random_seed: u32,
    }

    pub type EdgeLlamaTokenCallback = extern "C" fn(context: *mut c_void, token: i32) -> bool;

    unsafe extern "C" {
        pub fn edge_llama_open(path: *const c_char, error: *mut c_char, error_len: usize) -> *mut EdgeLlama;
        pub fn edge_llama_set_context(engine: *mut EdgeLlama, n_ctx: u32, error: *mut c_char, error_len: usize)
        -> bool;
        pub fn edge_llama_close(engine: *mut EdgeLlama);
        pub fn edge_llama_n_ctx_train(engine: *const EdgeLlama) -> i32;
        pub fn edge_llama_chat_template(engine: *const EdgeLlama) -> *const c_char;
        pub fn edge_llama_tokenize(
            engine: *const EdgeLlama,
            text: *const c_char,
            text_len: i32,
            tokens: *mut i32,
            max: i32,
        ) -> i32;
        pub fn edge_llama_token_piece(engine: *const EdgeLlama, token: i32, buf: *mut c_char, len: i32) -> i32;
        pub fn edge_llama_generate(
            engine: *mut EdgeLlama,
            prompt: *const i32,
            prompt_len: usize,
            sampling: EdgeLlamaSampling,
            on_token: EdgeLlamaTokenCallback,
            context: *mut c_void,
            error: *mut c_char,
            error_len: usize,
        ) -> bool;
    }
}
