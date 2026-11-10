# Langtrain Studio Desktop

<div align="center">
  <img src="public/langtrain-app-logo.svg" alt="Langtrain Logo" width="128" height="128">
  
  **The unified desktop studio for [Langtrain](https://langtrain.xyz) — Autonomous Agents, LLM Fine-Tuning, and Serving**
  
  [![Tauri](https://img.shields.io/badge/Tauri-2.0-blue?logo=tauri)](https://tauri.app)
  [![React](https://img.shields.io/badge/React-19-blue?logo=react)](https://react.dev)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue?logo=typescript)](https://typescriptlang.org)
  [![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite)](https://vitejs.dev)
  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
</div>

---

## Overview

**Langtrain Studio** is a cross-platform desktop application designed for AI engineers building, aligning, fine-tuning, and serving LLMs and autonomous agents. Built with Tauri 2.0 and React 19, it gives you native desktop speed (~50MB RAM footprint) and complete control over local and cloud workflows.

- 🤖 **Autonomous Agents** — Create, configure, monitor, and interact with agent workflows
- 📁 **Datasets & Curation** — Ingest raw data, auto-detect schemas, and convert to training-ready JSONL
- 🎯 **Fine-Tuning Jobs** — Configure and monitor QLoRA, LoRA, DPO, and full training runs locally or on cloud GPUs
- 🚀 **vLLM Serving** — Spin up optimized inference engines with real-time token streaming
- 📊 **Analytics & Telemetry** — Monitor token throughput, cost estimation, and job completion metrics
- 💾 **Local & Offline Models** — Connect to local Ollama / LM Studio or local weight checkouts
- ⚙️ **Configurable Endpoints** — Switch seamlessly between Cloud (`api.langtrain.xyz`) and local dev servers (`localhost:8000`)

---

## Views & Capabilities

| View | Description |
|---|---|
| **Overview** | System summary, active agents, running fine-tuning jobs, and GPU utilization |
| **Agents** | Multi-agent runtime configuration, tool permissions, and interactive execution |
| **Projects** | Workspace organization for datasets, models, and run configurations |
| **Datasets** | Dataset upload, validation, row inspection, and Hugging Face import |
| **Data Curation** | Pattern detection, column mapping, and automated JSONL conversion |
| **Model Registry** | Curated catalog of open weights (Llama 3.1/3.2, Mistral, Qwen, DeepSeek) |
| **vLLM Serve** | Local and remote model serving with OpenAI-compatible API endpoints |
| **Tuning Jobs** | Real-time training loss graphs, evaluation metrics, and checkpoint exports |
| **Analytics** | Token consumption, historical spend, and job latency telemetry |
| **Local Models** | On-device model discovery via Ollama/LM Studio and local runners |
| **Settings** | Live authentication tokens, customizable backend endpoints, and theme management |

---

## Quick Start

### Prerequisites
- [Node.js](https://nodejs.org/) 20+
- [pnpm](https://pnpm.io/) 10+
- [Rust](https://rustup.rs/) 1.75+ (for Tauri builds)
- Platform-specific build tools:
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
  - **Linux**: `libwebkit2gtk-4.1-dev`, `libssl-dev`, `libgtk-3-dev`, `build-essential`
  - **Windows**: Visual Studio 2022 C++ Build Tools

### Running the App

```bash
# Clone the repository
git clone https://github.com/langtrain-ai/langtrain_studio_desktop.git
cd langtrain_studio_desktop

# Install dependencies
pnpm install

# Run the frontend Vite dev server (browser preview)
pnpm dev

# Run inside the native Tauri desktop shell
pnpm tauri dev

# Build production desktop binaries
pnpm tauri build
```

---

## Configuration & Environments

Langtrain Studio supports both cloud infrastructure and self-hosted environments:

1. **Cloud Production** (default):
   - **Backend API**: `https://api.langtrain.xyz` (auto-routes through `/api/v1`)
   - **Web / Auth**: `https://app.langtrain.xyz` and `https://auth.langtrain.xyz`
2. **Local Self-Hosted**:
   - Navigate to **Settings → API Settings** in the app.
   - Click **Local Server (:8000)** or enter `http://localhost:8000`.
   - The app immediately updates its target to `http://localhost:8000/api/v1`.

---

## Architecture

```
langtrain_studio_desktop/
├── src/
│   ├── components/
│   │   ├── layout/            # Sidebar, Topbar, Window Frame
│   │   ├── views/             # AgentsView, DatasetsView, TrainingView, SettingsView, etc.
│   │   └── common/            # Reusable buttons, cards, modals, tables
│   ├── services/
│   │   ├── api.ts             # Unified API client matching langtrain-server /api/v1
│   │   ├── auth.ts            # TOTP & session authentication manager
│   │   └── local.ts           # Ollama / LM Studio discovery and local inference
│   ├── lib/
│   │   ├── settings.ts        # SettingsManager with reactive endpoint switching
│   │   ├── storage.ts         # Secure local storage wrapper
│   │   └── tokens/            # Design tokens (colors, typography, navigation)
│   └── main.tsx               # App bootstrapper
└── src-tauri/                 # Rust Tauri shell configuration & native window handlers
```

---

## License

MIT © [Langtrain AI](https://langtrain.xyz)
