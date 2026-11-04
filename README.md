# Langtrain Studio

Fast local LLM serving and fine-tuning that works on a plain CPU. One binary for macOS, Linux and Windows, with:

- an **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/fine_tuning/jobs`)
- an **Ollama-compatible API** (`/api/chat`, `/api/generate`, `/api/tags`), so existing Ollama clients work unchanged
- the **Studio web UI** at `http://127.0.0.1:11435/` (Chat, Models, Fine-tune, System), compiled into the binary
- a **native Rust inference engine**, with llama.cpp as a fallback for models it doesn't cover yet
- **LangTune**, LoRA fine-tuning designed for CPUs. Trained adapters are served immediately as `model@adapter`, with no merge step.

## Quick start

```bash
cargo build --release -p langtrain
./target/release/langtrain pull bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
./target/release/langtrain run SmolLM2-135M-Instruct-Q8_0 "What is the capital of France?"
./target/release/langtrain serve            # API + Studio UI on :11435 (use --port 11434 to stand in for Ollama)
```

Fine-tune (needs Python with `torch`, `transformers` and `gguf`):

```bash
pip install -e langtune
langtrain tune SmolLM2-135M-Instruct-Q8_0 train.jsonl --name support-bot   # talks to a running `langtrain serve`
langtrain run SmolLM2-135M-Instruct-Q8_0@support-bot
```

`train.jsonl` holds one example per line, as `{"messages": [...]}`, `{"prompt", "completion"}` or `{"text"}`.

## Layout

| Path | What it is |
|---|---|
| `crates/engine` | Scheduler: continuous batching, chunked prefill, slot prefix cache with cross-slot KV copy, per-step multi-LoRA groups, sampling, stop strings. Backend-agnostic, with a mock backend for tests. |
| `crates/native` | Langtrain's own engine: zero-copy GGUF reader, int8 kernels (NEON, AVX2, portable) for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and the K-quants Q4_K/Q5_K/Q6_K, paged f16 KV cache with copy-on-write, byte-level BPE tokenizer, Jinja chat templates, Llama/Mistral/Qwen2 forward pass, unmerged LoRA. |
| `crates/engine/src/llama.rs` | llama.cpp backend (all GGUF quant types and architectures; Metal/CUDA/Vulkan builds; reloads on the CPU if the GPU cannot be used). |
| `crates/server` | axum daemon: OpenAI + Ollama APIs, model store, fine-tuning job runner. |
| `crates/web` | `langtrain-web`: the Studio UI, embedded with `include_str!`; no Node build step. |
| `crates/cli` | `langtrain serve / run / list / pull / doctor / bench / tune`. |
| `langtune/` | Python trainer: LoRA on the top K layers, frozen-prefix activation cache, GGUF adapter export. |
| `studio/` | Tauri 2 desktop shell that runs `langtrain serve` as a sidecar. |
| `.github/workflows` | CI on Linux x64/arm64, macOS and Windows; tagged releases for 6 targets plus desktop bundles. |

`--backend auto` (the default) uses the native engine when it supports the model file and llama.cpp otherwise. Force one with `--backend native` or `--backend llama`.

Concurrency and context are tuned to the machine's RAM; override them with `--parallel <slots>` and `--ctx <tokens per slot>` (or `LANGTRAIN_PARALLEL` / `LANGTRAIN_CTX`).

## How CPU fine-tuning stays fast

1. **Frozen, quantized base.** Only the LoRA weights train; the base model never needs gradients or optimizer state.
2. **Top-K layers only.** LoRA is applied to the last K of L layers (on CPU the default is about a third of the layers).
3. **Frozen-prefix activation cache.** The hidden states entering layer L-K never change during training, so they are computed once (fp16) and reused. Every step runs forward and backward through K layers instead of L.
4. **Sparse LM head.** Logits are computed only at labeled (assistant) positions, which matters with 100k+ vocabularies.
5. **Built to run in the background.** The trainer runs at low OS priority, checkpoints and resumes, and reports progress and an ETA as JSON lines.

## Measured so far

Apple M2 (4 performance + 4 efficiency cores, 16 GB), CPU only, SmolLM2-135M-Instruct, 4 threads (the auto-tuned default):

| | Native engine | llama.cpp |
|---|---|---|
| Greedy output vs llama.cpp, Q8_0 | Identical over 120 tokens (code, quotes, emoji) | reference |
| Greedy output vs llama.cpp, Q4_K_M | Coherent; diverges after a few tokens (activation block size differs at near-tied logits) | reference |
| Dequantization vs gguf-py reference, Q4_K_M | All 211 tensors match (Q4_K, Q6_K, Q5_0, Q8_0) | |
| Q4_K_M, 1 stream | ~132 tok/s per stream | ~320 tok/s |
| Q4_K_M, 4 concurrent streams | ~188 tok/s total | ~409 tok/s total |
| Prefix cache on a shared system prompt | 93–96% of prompt tokens reused | same scheduler |

LangTune: 30 examples, 8 epochs, top 10/30 layers took 43 s end to end on the CPU (loss 2.29 → 0.004), and the adapter answered correctly on both engines.

Threads matter: on hybrid CPUs, using the efficiency cores too made both engines slower, so the profile uses performance cores only. The native engine is currently about 2x slower than llama.cpp; the gap is in the matmul and attention loops, not in batching or caching.

## Native engine roadmap

- Close the speed gap with llama.cpp: tiled multi-row matmul (i8mm `smmla`), a lighter attention loop, int8 KV cache
- AVX-512/AMX and SVE kernels
- SentencePiece tokenizers (Llama 2, Mistral v0.1) and more architectures (Gemma, Phi, Qwen3)
- Speculative decoding, constrained JSON output, KV cache saved to disk
- A Metal compute path
