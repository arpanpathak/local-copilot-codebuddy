//! Safe Rust wrapper around TensorRT-LLM's C++ executor (see `cpp/trtllm_shim.cpp`).

use std::ffi::{CStr, CString, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

use anyhow::{Result, bail};

/// Identifies one generation running on the engine.
pub type RequestId = u64;

/// Generation settings for one request.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub max_new_tokens: u32,
    /// 0 means greedy decoding.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 means no top-k limit.
    pub top_k: u32,
    /// 1.0 means no penalty; higher values discourage repeating earlier text.
    pub repetition_penalty: f32,
    /// Seeds the sampler, so asking again gives a different answer.
    pub random_seed: u64,
    /// Generation stops after this token (e.g. `<|im_end|>`).
    pub end_token: u32,
}

/// A TensorRT-LLM engine loaded on the GPU.
pub struct Engine {
    raw: NonNull<ffi::EdgeEngine>,
}

// The C++ executor is thread-safe: requests can be started, awaited and
// cancelled from any thread.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Loads the engine built by `trtllm-build` in `engine_dir`, reserving a
    /// KV cache of `kv_cache_tokens` tokens. `paged_context` must match the
    /// engine's build (`--use_paged_context_fmha`); it turns on chunked
    /// prefill and KV-cache reuse between turns.
    pub fn load(engine_dir: &Path, kv_cache_tokens: u32, paged_context: bool) -> Result<Self> {
        let engine_dir = CString::new(engine_dir.as_os_str().as_bytes())?;
        let mut error = ErrorBuffer::new();
        let raw = unsafe {
            ffi::edge_engine_open(
                engine_dir.as_ptr(),
                kv_cache_tokens as i32,
                paged_context,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        match NonNull::new(raw) {
            Some(raw) => Ok(Self { raw }),
            None => bail!("cannot load the engine: {}", error.message()),
        }
    }

    /// Starts generating a reply to `prompt` (token ids). Tokens are then
    /// collected with [`Engine::wait_for_tokens`].
    pub fn start(&self, prompt: &[u32], sampling: Sampling) -> Result<RequestId> {
        let sampling = ffi::EdgeSampling {
            max_new_tokens: sampling.max_new_tokens as i32,
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k as i32,
            repetition_penalty: sampling.repetition_penalty,
            random_seed: sampling.random_seed,
            end_token: sampling.end_token as i32,
        };
        let mut error = ErrorBuffer::new();
        // Token ids are below 2^31, so u32 and i32 share the same bits: pass the slice as-is.
        let request_id = unsafe {
            ffi::edge_engine_start(
                self.raw.as_ptr(),
                prompt.as_ptr().cast(),
                prompt.len(),
                sampling,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        match request_id {
            0 => bail!("cannot start generating: {}", error.message()),
            request_id => Ok(request_id),
        }
    }

    /// Blocks until `request` produces tokens and appends them to `tokens`.
    /// Returns `true` once the reply is complete (or was cancelled).
    pub fn wait_for_tokens(&self, request: RequestId, tokens: &mut Vec<u32>) -> Result<bool> {
        extern "C" fn push_token(context: *mut c_void, token: i32) {
            let tokens = unsafe { &mut *context.cast::<Vec<u32>>() };
            tokens.push(token as u32);
        }

        let mut finished = false;
        let mut error = ErrorBuffer::new();
        let succeeded = unsafe {
            ffi::edge_engine_wait(
                self.raw.as_ptr(),
                request,
                push_token,
                (tokens as *mut Vec<u32>).cast(),
                &mut finished,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if !succeeded {
            bail!("generation failed: {}", error.message());
        }
        Ok(finished)
    }

    /// Stops `request`. A thread waiting on it then sees it as finished.
    pub fn cancel(&self, request: RequestId) {
        unsafe { ffi::edge_engine_cancel(self.raw.as_ptr(), request) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::edge_engine_close(self.raw.as_ptr()) }
    }
}

/// Receives error messages from the C++ side.
struct ErrorBuffer([u8; 1024]);

impl ErrorBuffer {
    fn new() -> Self {
        Self([0; 1024])
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

/// Declarations matching `cpp/trtllm_shim.cpp`.
mod ffi {
    use std::ffi::{c_char, c_void};

    /// Opaque handle to the C++ executor.
    #[repr(C)]
    pub struct EdgeEngine {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct EdgeSampling {
        pub max_new_tokens: i32,
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: i32,
        pub repetition_penalty: f32,
        pub random_seed: u64,
        pub end_token: i32,
    }

    pub type EdgeTokenCallback = extern "C" fn(context: *mut c_void, token: i32);

    unsafe extern "C" {
        pub fn edge_engine_open(
            engine_dir: *const c_char,
            kv_cache_tokens: i32,
            paged_context: bool,
            error: *mut c_char,
            error_len: usize,
        ) -> *mut EdgeEngine;

        pub fn edge_engine_close(engine: *mut EdgeEngine);

        pub fn edge_engine_start(
            engine: *mut EdgeEngine,
            prompt: *const i32,
            prompt_len: usize,
            sampling: EdgeSampling,
            error: *mut c_char,
            error_len: usize,
        ) -> u64;

        pub fn edge_engine_wait(
            engine: *mut EdgeEngine,
            request_id: u64,
            on_token: EdgeTokenCallback,
            context: *mut c_void,
            finished: *mut bool,
            error: *mut c_char,
            error_len: usize,
        ) -> bool;

        pub fn edge_engine_cancel(engine: *mut EdgeEngine, request_id: u64);
    }
}
