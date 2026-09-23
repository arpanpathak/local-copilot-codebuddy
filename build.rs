//! Compiles the C++ shims over the inference engines and links them:
//! TensorRT-LLM (required) and llama.cpp (optional, for GGUF models).
//!
//! Expects the runtime installed by `engine/install-runtime.sh`, in
//! `$TRTLLM_ROOT` (default `~/.local/lib/local-copilot-codebuddy-trtllm`).

use std::env;
use std::path::PathBuf;

fn main() {
    let trtllm_root = match env::var_os("TRTLLM_ROOT") {
        Some(root) => PathBuf::from(root),
        None => PathBuf::from(env::var_os("HOME").expect("HOME is not set"))
            .join(".local/lib/local-copilot-codebuddy-trtllm"),
    };
    let include_dir = trtllm_root.join("include");
    let lib_dir = trtllm_root.join("lib");
    assert!(
        lib_dir.join("libtensorrt_llm.so").exists(),
        "TensorRT-LLM runtime not found in {}; run engine/install-runtime.sh first",
        trtllm_root.display()
    );

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/trtllm_shim.cpp")
        .include(&include_dir)
        .include("/usr/local/cuda/include")
        .warnings(false) // TensorRT-LLM's headers are noisy
        .compile("trtllm_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");
    // Find the runtime at startup without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    println!("cargo:rerun-if-changed=cpp/trtllm_shim.cpp");
    println!("cargo:rerun-if-env-changed=TRTLLM_ROOT");

    link_llama_cpp();
}

/// Adds the llama.cpp engine (GGUF models) when engine/install-llama.sh has
/// installed it in `$LLAMA_ROOT` (default `~/.local/lib/local-copilot-codebuddy-llama`).
/// Without it, the app runs TensorRT-LLM engines only.
fn link_llama_cpp() {
    println!("cargo::rustc-check-cfg=cfg(llamacpp)");
    println!("cargo:rerun-if-env-changed=LLAMA_ROOT");
    println!("cargo:rerun-if-changed=cpp/llama_shim.cpp");

    let llama_root = match env::var_os("LLAMA_ROOT") {
        Some(root) => PathBuf::from(root),
        None => PathBuf::from(env::var_os("HOME").expect("HOME is not set"))
            .join(".local/lib/local-copilot-codebuddy-llama"),
    };
    let lib_dir = llama_root.join("lib");
    println!("cargo:rerun-if-changed={}", lib_dir.join("libllama.so").display());
    if !lib_dir.join("libllama.so").exists() {
        println!("cargo:warning=llama.cpp not found in {}: GGUF models are disabled", llama_root.display());
        return;
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/llama_shim.cpp")
        .include(llama_root.join("include"))
        .compile("llama_shim");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rustc-cfg=llamacpp");
}
