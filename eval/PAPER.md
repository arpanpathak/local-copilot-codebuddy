# Confident but Non-Compliant: How Small Local Code Models Follow (and Ignore) Explicit Style Rules on a 16 GB Edge Device

**Arpan Pathak** · September 2026

## Abstract

Developers who run coding assistants locally, for privacy or cost, often report that small models "forget" the style rules they are given and behave lazily. We measure this on an NVIDIA Jetson Orin NX 16GB, the kind of low-cost edge device such users own. Two quantized models that fit its shared memory, Qwen2.5-Coder-7B-Instruct (GPTQ-Int4, TensorRT-LLM) and the newer Qwen3.5-9B (Q4_K_XL, llama.cpp), were given five explicit, machine-checkable Rust coding rules and a two-turn task, at sampling temperatures 0, 0.2 and 0.7. Across 46 answers, **no answer followed all five rules, only 1 compiled, and none had passing tests.** Rules that agree with common Rust habits were followed almost perfectly (iterators instead of index loops: 100%; no `unwrap` in main code: 44 of 46 answers), while rules that contradict habit were ignored almost always (no `unwrap` in tests, despite the rule saying "not even in tests": 236 violations; no comments inside function bodies: 0 of 46 compliant). The newer model wrote three times more explanation but was no more compliant, and it claimed full compliance in **every** answer (22 of 22), in 6 of 22 answers under a dedicated heading such as "Summary of Compliance with Rules", placed next to code that broke them. Lowering the temperature did not help. We release the harness, raw answers and scores.

## 1. Introduction

A local coding assistant is attractive when code must stay private or when paid APIs are out of reach. On affordable hardware the model must be small: the Jetson Orin NX used here has 16 GB of memory shared between CPU and GPU, which caps practical models at roughly 7 to 9 billion parameters at 4-bit precision. Users of such setups frequently describe the model as "dumb" or "lazy": it forgets the coding style it was asked to follow and produces code that looks finished but is not.

This paper turns those impressions into measurements. We ask:

1. **RQ1:** How often do small local models follow explicit, checkable coding rules?
2. **RQ2:** Which rules are followed, and which are ignored?
3. **RQ3:** Does a newer model, or a lower sampling temperature, help?
4. **RQ4:** What "lazy" behaviours appear, and how often do models misreport their own compliance?

## 2. Setup

### 2.1 Hardware and software

| | |
|---|---|
| Device | NVIDIA Jetson Orin NX 16GB (Super developer kit), power mode MAXN_SUPER |
| System | JetPack 6.2 (L4T R36.4.3), CUDA 12.6 |
| Memory | 16 GB shared by CPU and GPU; a GNOME desktop was running during all measurements |

### 2.2 Models

| | Model A | Model B |
|---|---|---|
| Model | Qwen2.5-Coder-7B-Instruct | Qwen3.5-9B |
| Released | September 2024 | February 2026 |
| Specialization | code | general (hybrid linear/full attention) |
| Quantization | GPTQ 4-bit (official) | Unsloth UD-Q4_K_XL GGUF |
| Runtime | TensorRT-LLM 0.12, in-process C++ executor | llama.cpp (commit e6ab7c1, CUDA) |
| Front end | local-copilot-codebuddy terminal app | llama-completion |
| Reply limit | 8192 tokens | 8192 tokens |

Model B is the newest model family we found that fits this device: at the time of writing, the newer Qwen releases (3.6 and 3.8) are 27B dense models or mixture-of-experts models far larger than that (Qwen3.8-Flash-Next has roughly 120B parameters by our estimate from its configuration), whose 4-bit weights alone exceed the board's memory. Model B cannot run on TensorRT-LLM 0.12 (the last version for JetPack 6), which is why a second runtime was needed. Model B was used in its default non-thinking mode.

Both models used Qwen's recommended sampling settings (top-p 0.8, top-k 20, repetition penalty 1.1), with the temperature varied.

### 2.3 Prompt

Both models received the same system prompt: local-copilot-codebuddy's default prompt, which asks for thorough, book-style answers when detail is requested, followed by the user's rules as the app appends them:

> The user's coding rules. Follow them in every answer, for the whole conversation:
>
> 1. Never use `unwrap()` or `expect()`, not even in tests. Use `?` and return `Result`.
> 2. Errors are a custom `enum` that implements `std::fmt::Display` and `std::error::Error`. No external crates.
> 3. Every `pub` item has a `///` doc comment.
> 4. No comments inside function bodies.
> 5. No index loops like `for i in 0..n`; use iterators.

The rules constrain only the code. Answer length was deliberately left free, because the app is meant to give generous, detailed explanations.

### 2.4 Task

Each conversation has two turns:

1. *"Write a Rust function that parses a config file of `key = value` lines (skip blank lines and lines starting with #) into a HashMap<String, String>, with a unit test. Explain your design in detail."*
2. *"Now add support for `[section]` headers: keys under a section become `section.key`. Show the full updated code."*

The second turn tests whether rules survive a follow-up and whether the model keeps the whole program when asked for "the full updated code".

### 2.5 Design

| Temperature | Model A conversations | Model B conversations |
|---|---|---|
| 0 (greedy) | 2 | 1 |
| 0.2 | 5 | 5 |
| 0.7 | 5 | 5 |

Greedy decoding was verified to be deterministic (both Model A runs at T=0 were byte-identical), so Model B was run once at T=0. This gives 24 answers for Model A and 22 for Model B, 46 in total.

## 3. Method

### 3.1 Collection

Model A was driven through the real terminal app in a virtual terminal (pyte), exactly as a user would use it: each conversation starts the app, types the two turns, and reads the answer from the screen. Model B was run through llama.cpp's `llama-completion` with the same system prompt in Qwen's ChatML format. Both harnesses record the answer text, the number of generated tokens and the generation speed.

### 3.2 Extracting the code

Answers are long and split code across several blocks (error type, function, tests, usage examples), often repeating an updated version of an item. For each answer we assemble the code the way a reader would: every top-level item (`use`, `struct`, `enum`, `fn`, `impl`, `mod`) is taken from all blocks, a later version of an item replaces an earlier one, and loose example statements are dropped. For Model A, the on-screen code was checked against the app's own exact clipboard copy of the last block: they matched in 23 of 24 answers, and in the 24th the clipboard held unrelated code, most likely copied by another program at that moment, so the on-screen code was used.

### 3.3 Scoring

| Metric | Definition |
|---|---|
| R1 | no `.unwrap(` or `.expect(` anywhere (string literals excluded; `unwrap_or` and friends allowed) |
| R2 | an `enum` with `impl Display` and `impl Error` (aliases such as `StdError` accepted), and no external crate, whether imported with `use` or called by path (e.g. `tempfile::tempdir()`) |
| R3 | every `pub` item is directly preceded by a `///` comment (attributes allowed in between) |
| R4 | no `//` or `/* */` comment inside any function body |
| R5 | no `for x in 0..` loops |
| Compiles | `cargo test --no-run` succeeds on the assembled code (edition 2021, no dependencies) |
| Tests pass | `cargo test` passes with at least one test |
| False claim | the prose claims compliance ("follows all the provided guidelines", "adheres to", "no `unwrap`", "no external crates", ...) while at least one rule is broken |
| Placeholder | the code elides parts ("`// ...`", "rest of the code unchanged") |
| Dropped | turn 2 lost the unit tests or the error enum that turn 1 had |

Every automatic check was validated by hand against the raw answers. Three scorer defects found this way were fixed before the results below: external crates used by full path were missed, the turbofish in `.collect::<Vec<_>>()` was mistaken for a crate, and aliased trait names (`impl StdError for ...`) were not recognised.

## 4. Results

### 4.1 Overall (RQ1)

| Model | n | All 5 rules | R1 | R2 | R3 | R4 | R5 | Compiles | Tests pass | False claim |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | 24 | **0%** | 21% | 38% | 92%\* | 0% | 100% | 4% (1) | 0% | 50% |
| Qwen3.5-9B | 22 | **0%** | 14% | 68% | 73% | 0% | 100% | 0% | 0% | **100%** |

\* Only 10 of the 24 Model A answers contain any `pub` item; the other 14 pass R3 trivially (see Section 5.3).

No answer from either model followed all five rules. On average an answer broke 2.5 rules (Model A: 2.50, Model B: 2.45). Only one of 46 answers compiled, and none had passing tests.

### 4.2 By temperature (RQ3)

| Model | T | n | All 5 rules | R1 | R2 | R3 | R4 | R5 | Compiles | False claim | Dropped (turn 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | 0 | 4 | 0% | 0% | 0% | 100% | 0% | 100% | 0% | 50% | 0% |
| Qwen2.5-Coder-7B | 0.2 | 10 | 0% | 20% | 40% | 80% | 0% | 100% | 0% | 50% | 0% |
| Qwen2.5-Coder-7B | 0.7 | 10 | 0% | 30% | 50% | 100% | 0% | 100% | 10% | 50% | 20% |
| Qwen3.5-9B | 0 | 2 | 0% | 0% | 100% | 100% | 0% | 100% | 0% | 100% | 0% |
| Qwen3.5-9B | 0.2 | 10 | 0% | 10% | 70% | 80% | 0% | 100% | 0% | 100% | 0% |
| Qwen3.5-9B | 0.7 | 10 | 0% | 20% | 60% | 60% | 0% | 100% | 0% | 100% | 0% |

Lowering the temperature did not improve compliance. Full compliance was 0% at every temperature for both models, and the per-rule rates move in no consistent direction. Greedy decoding (T=0) was, if anything, the worst setting for R1.

### 4.3 Which rules are followed (RQ2)

The per-rule rates split cleanly into rules that agree with common Rust practice and rules that go against it.

| Rule | Agrees with common practice? | Compliance (both models) |
|---|---|---|
| R5 iterators, no index loops | yes | 46 / 46 |
| R1 in main code: no `unwrap` | yes | 44 / 46 answers |
| R1 in tests: "not even in tests" | **no** (tests commonly use `unwrap`) | 8 / 45 answers with tests; **236** `unwrap`/`expect` calls in tests |
| R4 no comments in function bodies | **no** (explanatory comments are common) | **0 / 46**; 506 body comments in total |

| | `unwrap`/`expect` in main code | in tests | Body comments in main code | in tests |
|---|---|---|---|---|
| Qwen2.5-Coder-7B | 0 | 109 | 67 | 78 |
| Qwen3.5-9B | 2 | 127 | 163 | 198 |

Both models obey "never use `unwrap()`" exactly where the training data already agrees (library code) and ignore the explicit exception "not even in tests", which contradicts it. The same pattern appears for comments: models trained to explain code keep commenting it, even when told not to.

R2 fails mostly through one habit: reaching for the `tempfile` crate to create a test file (Model A: 15 of 24 answers; Model B: 6 of 22, plus `rand` in 2).

### 4.4 Compilation

The first compiler error of each answer:

| First compiler error | Qwen2.5-Coder-7B | Qwen3.5-9B |
|---|---|---|
| `?` cannot convert `io::Error` into the custom error (no `From` impl) | 3 | 7 |
| `?` applied to a value of the wrong type (e.g. a tuple where a `Vec` was expected) | 6 | 0 |
| external crate (`tempfile`) not available | 7 | 4 |
| `fs::write` used with only `std::fs::File` imported | 5 | 0 |
| `assert_eq!` on a `Result` whose error type lacks `PartialEq` | 2 | 0 |
| non-existent macro or method (`assert_matches!`, `.sorted()`) | 0 | 3 |
| pattern and slice errors (`splitn` matched as a slice, non-exhaustive match, unbound variable) | 0 | 6 |
| missing `Debug`, missing `mut` | 0 | 2 |
| compiles | 1 | 0 |

The most common failure of Qwen3.5-9B is instructive, and it also appears in Qwen2.5-Coder-7B: the rules demand a custom error enum and `?`, and the model writes both, but does not add the `From<std::io::Error>` conversion that makes `?` work. The code looks compliant and idiomatic yet cannot compile. For Qwen2.5-Coder-7B, the leading causes are the forbidden `tempfile` crate (7) and misuse of `?` on values of the wrong type (6).

### 4.5 Answer length and speed

| | Qwen2.5-Coder-7B | Qwen3.5-9B |
|---|---|---|
| Tokens per answer (mean) | 1,135 | 2,821 |
| Prose words per answer (mean) | 255 | 761 |
| Code lines per answer (mean) | 113 | 199 |
| Generation speed | 16.3 tok/s | 11.9 tok/s |
| Generation time per answer (mean) | 70 s | 237 s |

Model B writes three times more explanation, in the requested book style, and takes more than three times longer per answer on this device, with no gain in compliance.

## 5. Laziness and misreporting (RQ4)

### 5.1 False claims of compliance

Half of Model A's answers and **all** of Model B's answers claim to follow the rules while breaking at least one. The claims are specific, not vague. Examples, verbatim:

> "This implementation follows all the provided guidelines and provides a robust solution for parsing a configuration file into a HashMap."
> *(Qwen2.5-Coder-7B, T=0: the code has 6 `unwrap`/`expect` calls, uses the `tempfile` crate, and does not compile.)*

> "The design strictly adheres to the following constraints: 1. **Error Handling**: No use of `.unwrap()` or `.expect()`. All fallible operations r..."
> *(Qwen3.5-9B, T=0: the same answer contains 10 `unwrap`/`expect` calls; in turn 2, 12.)*

In 6 of its 22 answers, Model B adds a section with a heading such as "Summary of Compliance with Rules" that lists each rule as satisfied, and 8 of its answers state that "The design strictly adheres to the following constraints". In one answer it notices a violation mid-answer ("Let's refine the integration test to be fully compliant") and presents a "refined, fully compliant" version, which still breaks rules. We count that answer as a false claim; it is the only borderline case.

This is the most practically harmful behaviour measured: a user who trusts the summary will not look for the violations.

### 5.2 Dropping work in the follow-up

Asked for "the full updated code", Model A dropped the unit tests it had written in turn 1 in 1 of 12 follow-ups (at T=0.7). Model B kept its tests in every follow-up. Neither model used placeholders such as "`// ... rest unchanged`"; both rewrote the full program each time.

### 5.3 Avoiding a rule instead of following it

Rule R3 applies only to `pub` items. In 14 of 24 answers, Model A made nothing `pub` at all, so the documentation rule passed without a single doc comment being required. Model B made its API public in every answer, and so was held to R3 every time (73% compliant). A naive compliance score therefore flatters the older model on R3.

## 6. Threats to validity

- **One task, one language.** All results come from one Rust task with two turns. Other tasks or languages may behave differently.
- **Small samples.** 12 and 11 conversations per model. The results are clear-cut (0% full compliance in 46 of 46 answers), but per-rule differences between temperatures are within noise.
- **Automatic checks.** Rule checks use pattern matching plus the Rust compiler, not a full parser. Each check was validated against the raw answers, and three defects were fixed (Section 3.3), but edge cases may remain.
- **Quantization.** Both models run at 4-bit precision. Full-precision models may comply more often, but they do not fit this device alongside a desktop.
- **Different runtimes.** The models run on different inference engines, which affects speed. It should not affect content beyond sampling randomness.
- **Non-thinking mode.** Model B was tested in its default non-thinking mode. In two informal trials with thinking enabled, it closed its reasoning block immediately without reasoning; a proper study of thinking mode is future work.
- **Harness defects found and fixed.** A first run of Model B capped replies at 3,000 tokens, which cut 9 of 24 answers short. It was discarded and fully re-run with the 8,192-token limit the app uses; no answer in the reported run exceeded 3,449 tokens.

## 7. Implications for users of local models

- Rules that match common practice are followed for free; do not spend prompt space on them.
- Rules that contradict common practice (no `unwrap` in tests, no comments) are almost never followed by 7 to 9B models, regardless of temperature. Enforce them with tools (`clippy::unwrap_used`, a formatter, a pre-commit hook) instead of prompts.
- Do not trust a small model's own statement that it followed the rules. In this study such statements were false in 34 of 46 answers.
- Always compile. 45 of 46 answers that looked complete did not build.
- On this device, the newer, larger model gave longer explanations but not better compliance, at three times the time per answer.

## 8. Conclusion

On a 16 GB edge device, today's small local code models reliably follow coding rules that agree with their training habits and reliably ignore rules that do not, while confidently reporting full compliance. Neither a newer model generation that fits the device nor a lower temperature changed this. For users who cannot afford larger hardware, the practical path is to pair a local model with deterministic tools that enforce the rules, and to treat its self-assessment as unverified.

## Reproducing

Everything is in [`eval/`](.): the rules (`rules.md`), the harness (`run.py`), the scorer (`score.py`), the raw answers (`results/*.jsonl`) and the per-answer scores (`results/scores.csv`).

```sh
# Model A: needs the app installed, an X display, xclip and the pyte package
python3 eval/run.py trtllm   --temps 0 0.2 0.7 --runs 2 5 5
# Model B: needs llama.cpp built in ~/.local/src/llama.cpp and the GGUF in ~/models/gguf
python3 eval/run.py llamacpp --temps 0 0.2 0.7 --runs 1 5 5
# Scores every answer (needs cargo) and prints the tables
python3 eval/score.py
```

A full run takes about 30 minutes for Model A and 1 hour 40 minutes for Model B on the Jetson Orin NX.
