//! Compiles the C++ shim over TensorRT-LLM's executor and links the runtime.
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
}
