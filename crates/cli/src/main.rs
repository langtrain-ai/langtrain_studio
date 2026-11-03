//! `langtrain`: serve, chat with, benchmark and fine-tune local models.

use std::io::{BufRead, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use langtrain_engine::hardware;
use langtrain_engine::sampling::SamplingParams;
use langtrain_engine::{ChatMessage, Event, Prompt};
use langtrain_server::{AppState, BackendChoice, ServerConfig};
use serde_json::{json, Value};

const DEFAULT_PORT: u16 = 11435;

#[derive(Parser)]
#[command(name = "langtrain", version, about = "Fast local LLM serving and CPU fine-tuning")]
struct Cli {
    /// Data directory for models, adapters and jobs.
    #[arg(long, env = "LANGTRAIN_HOME", global = true)]
    home: Option<PathBuf>,
    /// Inference engine: `auto` (native when supported), `native`, or `llama`.
    #[arg(long, env = "LANGTRAIN_BACKEND", global = true, default_value = "auto")]
    backend: BackendChoice,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the API server and Studio web UI.
    Serve {
        #[arg(long, env = "LANGTRAIN_HOST", default_value = "127.0.0.1")]
        host: String,
        /// Use 11434 to stand in for Ollama.
        #[arg(long, env = "LANGTRAIN_PORT", default_value_t = DEFAULT_PORT)]
        port: u16,
        /// Expose the built-in `mock` model (for UI development and tests).
        #[arg(long)]
        mock: bool,
    },
    /// Chat with a model in the terminal (no server needed).
    Run {
        model: String,
        /// One-shot prompt; omit for an interactive chat.
        prompt: Option<String>,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// List installed models and adapters.
    List,
    /// Download a GGUF from Hugging Face: `pull <repo> <file>` or `pull <url>`.
    Pull {
        repo_or_url: String,
        file: Option<String>,
        /// Local model name (defaults to the file name).
        #[arg(long)]
        name: Option<String>,
    },
    /// Show hardware and the auto-tuned runtime profile.
    Doctor,
    /// Measure single-stream and concurrent throughput and prefix-cache gains.
    Bench {
        model: String,
        #[arg(long, value_delimiter = ',', default_value = "1,4,8")]
        concurrency: Vec<usize>,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
    },
    /// Start a fine-tuning job on a running server and follow its progress.
    Tune {
        model: String,
        /// JSONL training data.
        training_file: PathBuf,
        /// Adapter name; the result is served as `model@name`.
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value_t = 3)]
        epochs: u32,
        /// Train only the top N layers (enables the frozen-prefix activation cache).
        #[arg(long)]
        train_layers: Option<u32>,
        #[arg(long)]
        device: Option<String>,
        #[arg(long, default_value_t = format!("http://127.0.0.1:{DEFAULT_PORT}"))]
        server: String,
    },
}

fn home_dir(arg: Option<PathBuf>) -> PathBuf {
    arg.unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".langtrain"))
}

/// The trainer ships next to the binary (`langtune/`) in release archives.
fn langtune_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LANGTUNE_DIR") {
        return Some(PathBuf::from(d));
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    [dir.join("langtune"), dir.join("../langtune"), dir.join("../../langtune"), dir.join("../Resources/langtune")]
        .into_iter()
        .find(|p| p.join("langtune").join("__init__.py").exists())
}

fn config(home: PathBuf, mock: bool) -> ServerConfig {
    let backend = BACKEND.get().copied().unwrap_or(BackendChoice::Auto);
    let hw = hardware::detect();
    let profile = hw.profile();
    ServerConfig {
        home,
        enable_mock: mock,
        langtune_dir: langtune_dir(),
        python: std::env::var("LANGTRAIN_PYTHON").unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into()),
        hardware: hw,
        profile,
        backend,
    }
}

static BACKEND: std::sync::OnceLock<BackendChoice> = std::sync::OnceLock::new();

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("LANGTRAIN_LOG").unwrap_or_else(|_| "langtrain=info,langtrain_server=info,warn".into()))
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let home = home_dir(cli.home);
    let _ = BACKEND.set(cli.backend);
    match cli.cmd {
        Cmd::Serve { host, port, mock } => {
            let addr: SocketAddr = format!("{host}:{port}").parse().context("invalid host/port")?;
            langtrain_server::serve(config(home, mock), addr).await
        }
        Cmd::Run { model, prompt, temperature, max_tokens } => run(home, model, prompt, temperature, max_tokens).await,
        Cmd::List => list(home),
        Cmd::Pull { repo_or_url, file, name } => pull(home, repo_or_url, file, name).await,
        Cmd::Doctor => doctor(),
        Cmd::Bench { model, concurrency, tokens } => bench(home, model, concurrency, tokens).await,
        Cmd::Tune { model, training_file, name, epochs, train_layers, device, server } => {
            tune(server, model, training_file, name, epochs, train_layers, device).await
        }
    }
}

async fn stream_to_stdout(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>) -> Result<String> {
    let mut out = String::new();
    let mut stdout = std::io::stdout();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Text(t) => {
                print!("{t}");
                stdout.flush()?;
                out.push_str(&t);
            }
            Event::Done { usage, .. } => {
                println!();
                eprintln!(
                    "\x1b[2m{} tokens, {:.1} tok/s, prefill {:.0} ms, {} prompt tokens cached\x1b[0m",
                    usage.completion_tokens,
                    usage.decode_tokens_per_sec(),
                    usage.prefill_ms,
                    usage.cached_tokens
                );
                break;
            }
            Event::Error(e) => bail!(e),
        }
    }
    Ok(out)
}

async fn run(home: PathBuf, model: String, prompt: Option<String>, temperature: f32, max_tokens: Option<usize>) -> Result<()> {
    let state = AppState::new(config(home, model == "mock"));
    let params = SamplingParams { temperature, max_tokens, ..Default::default() };
    if let Some(p) = prompt {
        let rx = state.generate(&model, Prompt::Chat(vec![ChatMessage::new("user", p)]), params).await?;
        stream_to_stdout(rx).await?;
        return Ok(());
    }
    eprintln!("Chatting with {model}. Empty line or Ctrl-D to quit.");
    let mut history: Vec<ChatMessage> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        eprint!("\x1b[1m> \x1b[0m");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        history.push(ChatMessage::new("user", line.trim()));
        let rx = state.generate(&model, Prompt::Chat(history.clone()), params.clone()).await?;
        let reply = stream_to_stdout(rx).await?;
        history.push(ChatMessage::new("assistant", reply));
    }
}

fn list(home: PathBuf) -> Result<()> {
    let state = AppState::new(config(home, false));
    let models = state.list_models();
    if models.is_empty() {
        println!("No models in {}. Try:\n  langtrain pull bartowski/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", state.config.models_dir().display());
    }
    for m in models {
        println!("{:<48} {:>8.2} GB", m.name, m.size_bytes as f64 / 1e9);
        for a in m.adapters {
            println!("  @{a}");
        }
    }
    Ok(())
}

async fn pull(home: PathBuf, repo_or_url: String, file: Option<String>, name: Option<String>) -> Result<()> {
    let url = if repo_or_url.starts_with("http://") || repo_or_url.starts_with("https://") {
        repo_or_url
    } else {
        let file = file.context("usage: langtrain pull <repo> <file.gguf>")?;
        format!("https://huggingface.co/{repo_or_url}/resolve/main/{file}")
    };
    let file_name = url.rsplit('/').next().unwrap_or("model.gguf").split('?').next().unwrap_or("model.gguf");
    let name = name.unwrap_or_else(|| file_name.trim_end_matches(".gguf").to_string());
    let dir = home.join("models");
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{name}.gguf"));
    let part = dir.join(format!("{name}.gguf.part"));

    let mut req = reqwest::Client::new().get(&url);
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        req = req.bearer_auth(tok);
    }
    let res = req.send().await?.error_for_status()?;
    let total = res.content_length().unwrap_or(0);
    let bar = indicatif::ProgressBar::new(total).with_style(
        indicatif::ProgressStyle::with_template("{msg} {bar:40} {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}")?,
    );
    bar.set_message(name.clone());
    let mut f = std::fs::File::create(&part)?;
    let mut stream = res.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        f.write_all(&chunk)?;
        bar.inc(chunk.len() as u64);
    }
    f.sync_all()?;
    drop(f);
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut std::fs::File::open(&part)?, &mut magic)?;
    if &magic != b"GGUF" {
        std::fs::remove_file(&part)?;
        bail!("downloaded file is not a GGUF model");
    }
    std::fs::rename(&part, &dest)?;
    bar.finish();
    println!("Installed {name}. Try: langtrain run {name}");
    Ok(())
}

fn doctor() -> Result<()> {
    let hw = hardware::detect();
    let profile = hw.profile();
    println!("{}", serde_json::to_string_pretty(&json!({ "hardware": hw, "profile": profile }))?);
    Ok(())
}

async fn bench(home: PathBuf, model: String, concurrency: Vec<usize>, tokens: usize) -> Result<()> {
    let state = AppState::new(config(home, model == "mock"));
    let params = SamplingParams { max_tokens: Some(tokens), temperature: 0.0, ..Default::default() };
    let system = "You are a concise assistant. Answer in plain prose. ".repeat(20);
    let topics = ["rivers", "volcanoes", "bridges", "compilers", "bees", "tides", "glass", "maps"];

    // Warm-up loads the model and the shared system prompt into the cache.
    let warm = state.generate(&model, Prompt::Chat(vec![ChatMessage::new("system", &system), ChatMessage::new("user", "Hi")]), params.clone()).await?;
    let (_, warm_usage) = drain(warm).await?;
    println!("model {model}: warm-up prefill {:.0} ms", warm_usage.prefill_ms);
    println!("{:>11} {:>14} {:>16} {:>12} {:>14}", "concurrency", "total tok/s", "per-stream tok/s", "TTFT p50 ms", "cached prompt");

    for &n in &concurrency {
        let t0 = Instant::now();
        let mut tasks = Vec::new();
        for i in 0..n {
            let state = state.clone();
            let msgs = vec![
                ChatMessage::new("system", &system),
                ChatMessage::new("user", format!("Write about {} in 200 words.", topics[i % topics.len()])),
            ];
            let params = params.clone();
            let model = model.clone();
            tasks.push(tokio::spawn(async move {
                let start = Instant::now();
                let mut rx = state.generate(&model, Prompt::Chat(msgs), params).await?;
                let mut ttft = None;
                loop {
                    match rx.recv().await {
                        Some(Event::Text(_)) => {
                            ttft.get_or_insert(start.elapsed());
                        }
                        Some(Event::Done { usage, .. }) => return Ok::<_, anyhow::Error>((ttft.unwrap_or_default(), usage)),
                        Some(Event::Error(e)) => bail!(e),
                        None => bail!("engine stopped"),
                    }
                }
            }));
        }
        let mut ttfts = Vec::new();
        let (mut gen, mut cached, mut prompt, mut per_stream) = (0usize, 0usize, 0usize, 0.0f64);
        for t in tasks {
            let (ttft, u) = t.await??;
            ttfts.push(ttft);
            gen += u.completion_tokens;
            cached += u.cached_tokens;
            prompt += u.prompt_tokens;
            per_stream += u.decode_tokens_per_sec();
        }
        ttfts.sort();
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "{:>11} {:>14.1} {:>16.1} {:>12.0} {:>13.0}%",
            n,
            gen as f64 / wall,
            per_stream / n as f64,
            ttfts[ttfts.len() / 2].as_secs_f64() * 1000.0,
            100.0 * cached as f64 / prompt.max(1) as f64
        );
    }
    Ok(())
}

async fn drain(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>) -> Result<(String, langtrain_engine::Usage)> {
    let mut s = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Text(t) => s.push_str(&t),
            Event::Done { usage, .. } => return Ok((s, usage)),
            Event::Error(e) => bail!(e),
        }
    }
    bail!("engine stopped")
}

async fn tune(
    server: String,
    model: String,
    training_file: PathBuf,
    name: Option<String>,
    epochs: u32,
    train_layers: Option<u32>,
    device: Option<String>,
) -> Result<()> {
    let client = reqwest::Client::new();
    let file = std::fs::canonicalize(&training_file).with_context(|| format!("{} not found", training_file.display()))?;
    let res = client
        .post(format!("{server}/v1/fine_tuning/jobs"))
        .json(&json!({
            "model": model,
            "training_file": file,
            "suffix": name,
            "hyperparameters": { "n_epochs": epochs, "train_layers": train_layers, "device": device },
        }))
        .send()
        .await
        .with_context(|| format!("is `langtrain serve` running at {server}?"))?;
    let job: Value = res.json().await?;
    if let Some(err) = job.get("error") {
        bail!("{}", err["message"]);
    }
    let id = job["id"].as_str().context("no job id")?.to_string();
    eprintln!("Started {id}");
    let job = Arc::new(id);
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let j: Value = client.get(format!("{server}/v1/fine_tuning/jobs/{job}")).send().await?.json().await?;
        let p = &j["progress"];
        if !p.is_null() {
            eprint!(
                "\r\x1b[2K{} step {}/{}  loss {:.3}  {:.0} tok/s  ETA {:.0}s",
                p["phase"].as_str().unwrap_or(""),
                p["step"],
                p["total_steps"],
                p["loss"].as_f64().unwrap_or(f64::NAN),
                p["tokens_per_sec"].as_f64().unwrap_or(0.0),
                p["eta_seconds"].as_f64().unwrap_or(0.0)
            );
        }
        match j["status"].as_str() {
            Some("succeeded") => {
                eprintln!("\nDone. Chat with it: langtrain run {}", j["fine_tuned_model"].as_str().unwrap_or_default());
                return Ok(());
            }
            Some("failed") | Some("cancelled") => bail!("\njob {}: {}", j["status"], j["error"].as_str().unwrap_or_default()),
            _ => {}
        }
    }
}
