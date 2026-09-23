<div align="center">

# 🤖 local-copilot-codebuddy

### A private, offline coding copilot that lives in your terminal and runs entirely on an NVIDIA Jetson.

**No cloud. No API keys. No server. No Python at runtime.**<br>
Just a single native binary that loads a TensorRT-LLM engine straight onto the Jetson GPU.

<br>

[![Rust](https://img.shields.io/badge/Rust-2024_edition-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](cpp/trtllm_shim.cpp)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson_Orin-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-orin)
[![TensorRT-LLM](https://img.shields.io/badge/TensorRT--LLM-0.12-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/NVIDIA/TensorRT-LLM)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder_7B-6E40C9?style=for-the-badge&logo=alibabacloud&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4)
[![Quantization](https://img.shields.io/badge/GPTQ-INT4-FF6F00?style=for-the-badge)](https://arxiv.org/abs/2210.17323)
[![Context](https://img.shields.io/badge/Context-32K_tokens-0A84FF?style=for-the-badge)](#limits)
[![Speed](https://img.shields.io/badge/Speed-~14_tok%2Fs-FF2D55?style=for-the-badge)](#performance-per-watt)

[![100% Offline](https://img.shields.io/badge/100%25-Offline_%26_Private-2EA44F?style=for-the-badge&logo=shield&logoColor=white)](#why)
[![License](https://img.shields.io/badge/License-Apache_2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-FF69B4?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arpanpathak/local-copilot-codebuddy/pulls)
[![Stars](https://img.shields.io/github/stars/arpanpathak/local-copilot-codebuddy?style=for-the-badge&logo=github&color=FFD700)](https://github.com/arpanpathak/local-copilot-codebuddy/stargazers)

<br>

[Why](#why) · [Features](#features) · [How it works](#how-it-works) · [Setup](#setup) · [Usage](#usage) · [Memory](#memory-friendly-by-design) · [Limits](#limits)

</div>

---

```
 ❯ you
 Write a Rust function that checks if a string is a palindrome, ignoring case.

 ◆ assistant
 ▏ rust
 ▏ fn is_palindrome(s: &str) -> bool {
 ▏     let cleaned: String = s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
 ▏     cleaned.chars().eq(cleaned.chars().rev())
 ▏ }
╭ message ─────────────────────────────────────────────────────────────────────╮
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
 ▲ NVIDIA  TensorRT-LLM  Qwen2.5-Coder-7B-Instruct-GPTQ-Int4 ctx 212/30720 · 51 tok · 14.5 tok/s
```

> That reply is real output from a Jetson Orin NX 16GB in MAXN_SUPER mode.

## Why

Cloud copilots send your code to someone else's server. This one never leaves
the board on your desk. local-copilot-codebuddy turns a Jetson into a
self-contained coding assistant:

- 🔒 **Private by construction.** Prompts, code and answers stay on the device.
  It works on a plane, in a lab with no internet, or behind the strictest firewall.
- ⚡ **Native all the way down.** The TUI, tokenizer and inference engine live
  in one Rust process that calls TensorRT-LLM's C++ executor directly. There is
  no HTTP hop, no Python interpreter and no container at runtime.
- 🧠 **Gives thorough answers.** Qwen2.5-Coder 7B writes long, structured
  explanations with headings, examples and code, and replies can run to 8K tokens.
- 🖥️ **Kind to the rest of your system.** Loading a 5.5 GB model on a 16 GB
  board used to freeze the desktop. It no longer does (see
  [Memory friendly by design](#memory-friendly-by-design)).

## Features

| | Feature | What it means |
|---|---|---|
| 🌊 | **Token streaming** | Text appears as the GPU produces it, with live tokens/sec in the status bar. |
| 🎨 | **Markdown and syntax highlighting** | Headings, lists, quotes and tables render in the terminal. Code is highlighted with bat's grammar set, even while a code block is still streaming in. |
| 📚 | **32K-token context** | Paste a 15K-token codebase in one go. It is answered correctly, and questions can refer back to its first line. |
| 🚀 | **Fast follow-ups** | The KV cache of the conversation so far is reused, so a new turn only processes the new message. After that 15K-token paste, the next answer starts in 0.5 s instead of about 30 s. |
| 📋 | **One-click copy** | Every code block has a `⧉ copy` button. Click it, or press `Ctrl+Y` for the latest block, and the code lands on your clipboard (over SSH too, via OSC 52). |
| 🛑 | **Instant stop** | `Ctrl+C` or `Esc` cancels generation on the GPU itself and keeps the partial reply. |
| 🪶 | **Idle means idle** | With no reply streaming, the process uses 0% CPU. |

## How it works

```
┌────────────────── local-copilot-codebuddy (one process) ───────────────────┐
│  TUI (ratatui)  ──▶  chat.rs: ChatML prompt, tokenizer (Rust `tokenizers`)  │
│                          │                                                 │
│                          ▼                                                 │
│                  engine.rs  ──FFI──▶  cpp/trtllm_shim.cpp                  │
│                                          │                                 │
│                                          ▼                                 │
│                            TensorRT-LLM C++ Executor  ──▶  GPU             │
└────────────────────────────────────────────────────────────────────────────┘
```

```
src/
├── main.rs        startup: load the model, run the TUI
├── cli.rs         command-line flags
├── chat.rs        conversation → prompt → tokens → streamed text
├── engine.rs      safe Rust wrapper around the C++ shim
├── memory.rs      sizes the KV cache to the memory that is actually free
├── event.rs       the one channel the UI thread waits on
├── app.rs         state: input, streaming status, scrolling
├── ui.rs          drawing: transcript, input box, status bar
├── markdown.rs    Markdown → styled lines
└── highlight.rs   code highlighting (syntect + two-face)
cpp/
└── trtllm_shim.cpp   C interface over TensorRT-LLM's Executor (~200 lines)
engine/               one-time setup: build the engine, install the runtime
```

- **Threads and channels.** Terminal input and generated text arrive on one
  channel. The UI thread applies everything waiting, then redraws once, so a
  burst of tokens costs one frame and an idle chat uses no CPU.
- **No needless copies.**
  - Token ids go to the engine as a borrowed slice.
  - Rendered Markdown and highlighted code borrow from the message text.
  - Your typed message is moved, not copied, into the conversation.

## Memory friendly by design

On a Jetson the CPU and GPU share the same RAM, and GPU memory can never be
swapped out. Every byte the model holds is a byte the desktop loses, so
local-copilot-codebuddy is careful with it:

1. **The engine file is memory-mapped, not read.** A plain load reads the whole
   5.5 GB engine into the heap and then copies the weights to the GPU, so for a
   moment the model sits in RAM twice (about 11 GB). On a 16 GB board that
   pushes the desktop into swap and the system freezes. Mapped pages are file
   cache that the kernel drops as soon as each weight is copied, and the cache
   is released once loading finishes.
2. **The KV cache is sized to what is free.** At startup the app reads
   `MemAvailable` and keeps the full 32K-token context when it fits. If it
   would leave the system less than about 1.5 GB, the context shrinks (the `ctx`
   counter shows the limit in use). If even 4K tokens do not fit, it refuses to
   start and asks you to close something, rather than letting the system freeze.

Measured on a Jetson Orin NX 16GB with a desktop, a browser and a Kubernetes
cluster running:

| | Before | After |
|---|---|---|
| Process memory during load | 5.5 GB | 0.37 GB |
| Load time | 20 to 40 s | about 7 s |
| Memory left for the OS while chatting | system froze | about 1.7 GB, steady |
| Swapped out during a full session | gigabytes | about 7 MB |

## Setup

These steps target JetPack 6.2 (L4T r36.4, CUDA 12.6), the setup it was
tested on. TensorRT-LLM for Jetson comes from the
[`dustynv/tensorrt_llm:0.12-r36.4.0`](https://hub.docker.com/r/dustynv/tensorrt_llm)
image. Docker is used only for the setup steps, never to chat.

**1. Build the engine** (once per model, about 4 minutes):

```sh
engine/build-engine.sh        # default: Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4
```

This downloads the 4-bit GPTQ weights to `~/models/trt/`, converts them, and
compiles a TensorRT engine for this GPU with `trtllm-build`. TensorRT-LLM's
engine compiler exists only as a Python tool, so this one step uses Python,
inside the container.

**2. Install the runtime** (once):

```sh
engine/install-runtime.sh
```

This copies TensorRT-LLM's C++ libraries (about 900 MB) out of the image into
`~/.local/lib/local-copilot-codebuddy-trtllm`, along with TensorRT 10.4. An
engine only loads with the TensorRT version that built it, and JetPack 6.2
ships 10.7.

**3. Install and run:**

```sh
cargo install --path .
local-copilot-codebuddy
```

Loading takes about 10 seconds. The model then uses about 4.5 GB for weights
and runtime, plus up to 1.75 GB for the 32K-token KV cache.

The status bar shows how full the context is (`ctx 1393/30720`), the
generation speed, and how long the model took to read the prompt before the
first token (`first token 0.5s`).

## Usage

```
local-copilot-codebuddy [OPTIONS] [MODEL_DIR]

  [MODEL_DIR]                  Hugging Face model directory (tokenizer)
                               [env: CODEBUDDY_MODEL]
                               [default: ~/models/trt/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4]
  -e, --engine <DIR>           Engine directory [env: CODEBUDDY_ENGINE]
                               [default: <MODEL_DIR>-engine]
  -s, --system <PROMPT>        System prompt [env: CODEBUDDY_SYSTEM]
  -t, --temperature <T>        Sampling temperature, 0 = greedy
                               [default: the model's generation_config.json, 0.7 for Qwen]
      --max-tokens <N>         Maximum tokens per reply [default: 8192]
      --kv-cache-tokens <N>    KV cache size in tokens, lowered automatically
                               when memory is short [default: 32768]
```

| Key                     | Action                                     |
| ----------------------- | ------------------------------------------ |
| `Enter`                 | Send                                       |
| `Alt+Enter` / `Ctrl+J`  | New line                                   |
| `Ctrl+C` / `Esc`        | Stop generating (keeps the partial reply); when idle, `Ctrl+C` clears the input |
| `↑` `↓` / `PgUp` `PgDn` | Scroll                                     |
| `End`                   | Back to the bottom, follow new output      |
| `Ctrl+Y`                | Copy the latest code block                 |
| Click `⧉ copy`          | Copy that code block                       |
| Mouse wheel             | Scroll                                     |
| `Ctrl+L`                | New conversation                           |
| `Ctrl+D`                | Quit                                       |

The app listens for mouse clicks (for the copy buttons) and the wheel. To
select any other text with the mouse, hold `Shift` while dragging.

## Other models

Any Qwen2 / Qwen2.5 chat model published in GPTQ-Int4 form works the same way:

```sh
engine/build-engine.sh Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4
local-copilot-codebuddy ~/models/trt/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4
```

The prompt format is ChatML (Qwen's). Other model families need their own chat
format in `chat.rs` and their own conversion step in `engine/`.

## Performance per watt

Token generation is limited by memory bandwidth, so the power mode matters most:

```sh
nvpmodel -q               # current mode (tested: MAXN_SUPER → 13.3 tok/s)
sudo nvpmodel -m 3        # 25 W
tegrastats                # power draw (VDD_IN) while a reply streams
```

To compare modes, divide the tok/s in the status bar by the watts that
`tegrastats` reports. Memory the desktop or browser holds is memory the model
cannot use.

## Limits

- A conversation can hold 30,720 tokens, and conversation plus reply 32,768.
  These limits are baked into the engine (`MAX_INPUT_LEN` / `MAX_SEQ_LEN` in
  `build-engine.sh`); 32K is also the most Qwen2.5 handles without extra
  scaling. Press `Ctrl+L` to start fresh when the `ctx` counter gets close.
- The first long paste takes time to read (about 500 tokens/s, so ~30 s for
  15K tokens); follow-ups reuse it.
- A 7B model at 4 bits writes long, structured answers but still makes
  mistakes, and can be out of date (e.g. old Kubernetes API versions).
- TensorRT-LLM 0.12 is the newest version available for JetPack 6. Newer
  NVIDIA runtimes (TensorRT Edge-LLM) need JetPack 7.

## Development

```sh
cargo test                   # markdown, prompt format, streaming decode, memory sizing
cargo clippy --all-targets
cargo fmt
```

## Roadmap

- [ ] Other model families (Llama, Mistral, Phi, Gemma): read each model's own
      chat template and stop tokens, and a generic engine build script.

## Contributing

Issues and pull requests are welcome. Good first contributions: chat formats for
other model families (Llama, Mistral, Phi), support for other Jetson modules, or
benchmarks from your own board.

## License

[Apache License 2.0](LICENSE)

<div align="center">
<br>

**Built for the edge, with ❤️ and Rust.** If it helps you, consider giving it a ⭐

</div>
