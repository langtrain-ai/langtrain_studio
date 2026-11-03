// Langtrain Studio UI. Talks only to the local daemon's public API.
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const gb = (bytes) => (bytes / 1024 ** 3).toFixed(2) + " GB";
const fmtSecs = (s) => {
  if (!isFinite(s) || s < 0) return "-";
  if (s < 90) return Math.round(s) + "s";
  if (s < 5400) return Math.round(s / 60) + " min";
  return (s / 3600).toFixed(1) + " h";
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...opts,
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error?.message || res.statusText);
  return data;
}

// ---------- navigation ----------
const views = { chat: null, models: loadModels, tune: loadTune, system: loadSystem };
document.querySelectorAll(".nav").forEach((b) =>
  b.addEventListener("click", () => show(b.dataset.view))
);
function show(name) {
  document.querySelectorAll(".nav").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (views[name]) views[name]();
  try { localStorage.setItem("lt.view", name); } catch {}
}

// ---------- shared model list ----------
let modelIds = [];
async function refreshModelIds() {
  const { data } = await api("/v1/models");
  modelIds = data.map((m) => m.id);
  const sel = $("#chat-model");
  const prev = sel.value || safeGet("lt.model");
  sel.replaceChildren(...modelIds.map((id) => new Option(id, id)));
  if (modelIds.includes(prev)) sel.value = prev;
  if (!modelIds.length) sel.append(new Option("No models: see the Models tab", ""));
}
function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }

// ---------- chat ----------
let history = [];
let controller = null;

$("#chat-model").addEventListener("change", (e) => { try { localStorage.setItem("lt.model", e.target.value); } catch {} });
$("#chat-clear").addEventListener("click", () => {
  history = [];
  $("#messages").replaceChildren(el("div", "empty", "New chat."));
  $("#chat-stats").textContent = "";
});
$("#prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
});
$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (controller) { controller.abort(); return; }
  const text = $("#prompt").value.trim();
  const model = $("#chat-model").value;
  if (!text || !model) return;
  $("#prompt").value = "";
  $("#messages").querySelector(".empty")?.remove();
  addMsg("user", text);
  history.push({ role: "user", content: text });
  const out = addMsg("assistant", "");
  await streamChat(model, out);
});

function addMsg(role, text) {
  const m = el("div", "msg " + role, text);
  $("#messages").append(m);
  m.scrollIntoView({ block: "end" });
  return m;
}

async function streamChat(model, out) {
  const system = $("#chat-system").value.trim();
  const messages = system ? [{ role: "system", content: system }, ...history] : history;
  controller = new AbortController();
  $("#send").textContent = "Stop";
  const t0 = performance.now();
  let first = 0, reply = "", usage = null;
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, messages, stream: true, temperature: +$("#chat-temp").value, stream_options: { include_usage: true } }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error?.message || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).replace(/^data: ?/, "");
        buf = buf.slice(i + 2);
        if (line === "[DONE]") continue;
        const ev = JSON.parse(line);
        if (ev.error) throw new Error(ev.error.message);
        if (ev.usage) usage = ev.usage;
        const delta = ev.choices?.[0]?.delta?.content;
        if (delta) {
          if (!first) first = performance.now();
          reply += delta;
          out.textContent = reply;
          out.scrollIntoView({ block: "end" });
        }
      }
    }
    history.push({ role: "assistant", content: reply });
    const secs = (performance.now() - (first || t0)) / 1000;
    if (usage) {
      $("#chat-stats").textContent =
        `${usage.completion_tokens} tokens · ${(usage.completion_tokens / Math.max(secs, 1e-3)).toFixed(1)} tok/s · ` +
        `first token ${Math.round((first || performance.now()) - t0)} ms · ` +
        `${usage.prompt_tokens_details?.cached_tokens ?? 0}/${usage.prompt_tokens} prompt tokens from cache`;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      history.push({ role: "assistant", content: reply });
    } else {
      out.classList.add("error");
      out.textContent = reply ? reply + "\n\n" + err.message : err.message;
      history.pop();
    }
  } finally {
    controller = null;
    $("#send").textContent = "Send";
    loadLoaded();
  }
}

// ---------- models ----------
async function loadModels() {
  const sys = await api("/v1/system");
  $("#models-dir").textContent = sys.models_dir;
  const body = $("#models-body");
  body.replaceChildren();
  for (const m of sys.models) {
    const tr = el("tr");
    tr.append(el("td", "", m.name), el("td", "", m.size_bytes ? gb(m.size_bytes) : "-"));
    const ad = el("td");
    if (!m.adapters.length) ad.textContent = "-";
    m.adapters.forEach((a) => ad.append(el("span", "pill", a)));
    const act = el("td");
    const b = el("button", "small ghost", "Chat");
    b.onclick = async () => { await refreshModelIds(); $("#chat-model").value = m.name; show("chat"); };
    act.append(b);
    tr.append(ad, act);
    body.append(tr);
  }
  if (!sys.models.length) body.append(Object.assign(el("tr"), { innerHTML: '<td colspan="4" class="muted">No models yet.</td>' }));
}

// ---------- fine-tuning ----------
let jobTimer = null;
async function loadTune() {
  const sys = await api("/v1/system");
  const sel = $("#tune-model");
  const prev = sel.value;
  const bases = sys.models.filter((m) => m.path).map((m) => m.name);
  sel.replaceChildren(...bases.map((n) => new Option(n, n)));
  if (bases.includes(prev)) sel.value = prev;
  const ft = sys.profile.fine_tune;
  $("#tune-advice").textContent = `Suggested on this machine: ${ft.device.toUpperCase()}, models up to ~${ft.recommended_max_params_b}B parameters. ${ft.note}`;
  await loadJobs();
  clearInterval(jobTimer);
  jobTimer = setInterval(() => {
    if ($("#view-tune").classList.contains("active")) loadJobs();
  }, 2000);
}

$("#tune-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#tune-error").textContent = "";
  const n = (id) => ($(id).value ? +$(id).value : undefined);
  try {
    await api("/v1/fine_tuning/jobs", {
      method: "POST",
      body: {
        model: $("#tune-model").value,
        training_file: $("#tune-file").value.trim(),
        suffix: $("#tune-suffix").value.trim() || undefined,
        hyperparameters: {
          n_epochs: n("#tune-epochs"),
          lora_rank: n("#tune-rank"),
          train_layers: n("#tune-layers"),
          device: $("#tune-device").value || undefined,
        },
      },
    });
    loadJobs();
  } catch (err) {
    $("#tune-error").textContent = err.message;
  }
});

async function loadJobs() {
  const { data } = await api("/v1/fine_tuning/jobs");
  const box = $("#jobs");
  if (!data.length) { box.replaceChildren(el("p", "muted", "No jobs yet.")); return; }
  box.replaceChildren(...data.map(renderJob));
}

function renderJob(j) {
  const card = el("div", "job");
  const head = el("div", "job-head");
  const title = el("div");
  title.append(el("strong", "", j.suffix), el("span", "muted", `  on ${j.model}`));
  head.append(title, el("span", "status " + j.status, j.status));
  card.append(head);

  const p = j.progress || {};
  if (j.status === "running" || j.status === "queued") {
    const frac = p.total_steps ? p.step / p.total_steps : 0;
    const bar = el("div", "bar");
    const fill = el("div");
    fill.style.width = (frac * 100).toFixed(1) + "%";
    bar.append(fill);
    card.append(bar);
    const parts = [];
    if (p.phase) parts.push(p.phase);
    if (p.total_steps) parts.push(`step ${p.step}/${p.total_steps}`);
    if (p.loss !== undefined) parts.push(`loss ${(+p.loss).toFixed(3)}`);
    if (p.tokens_per_sec) parts.push(`${p.tokens_per_sec.toFixed(0)} tok/s`);
    if (p.eta_seconds !== undefined) parts.push(`ETA ${fmtSecs(p.eta_seconds)}`);
    card.append(el("div", "job-meta", parts.join(" · ") || "starting…"));
    const c = el("button", "small ghost", "Cancel");
    c.onclick = () => api(`/v1/fine_tuning/jobs/${j.id}/cancel`, { method: "POST" }).then(loadJobs);
    head.append(c);
  }
  if (j.status === "succeeded" && j.fine_tuned_model) {
    const meta = el("div", "job-meta");
    meta.append(`Ready as `, el("code", "", j.fine_tuned_model), "  ");
    const b = el("button", "small", "Chat with it");
    b.onclick = async () => { await refreshModelIds(); $("#chat-model").value = j.fine_tuned_model; show("chat"); };
    meta.append(b);
    card.append(meta);
  }
  if (j.error) card.append(el("pre", "", j.error));
  return card;
}

// ---------- system ----------
function fill(dl, pairs) {
  dl.replaceChildren();
  for (const [k, v] of pairs) dl.append(el("dt", "", k), el("dd", "", v));
}
async function loadSystem() {
  const s = await api("/v1/system");
  const hw = s.hardware, pr = s.profile;
  fill($("#hw"), [
    ["CPU", hw.cpu_brand || hw.arch],
    ["Cores", `${hw.physical_cores} physical / ${hw.logical_cores} logical`],
    ["Memory", `${hw.total_memory_gb.toFixed(1)} GB total, ${hw.available_memory_gb.toFixed(1)} GB free`],
    ["CPU features", hw.cpu_features.join(", ") || "-"],
    ["GPU backends", hw.gpu_backends.join(", ") || "none (CPU build)"],
    ["OS", `${hw.os} ${hw.arch}`],
  ]);
  fill($("#profile"), [
    ["Threads", `${pr.threads} (batch ${pr.threads_batch})`],
    ["Parallel slots", pr.parallel_slots],
    ["Context per slot", pr.context_per_slot],
    ["KV cache", pr.kv_cache_type],
    ["GPU layers", pr.gpu_layers],
    ["Max model size", `${pr.max_model_gb.toFixed(1)} GB`],
    ["Fine-tuning", `${pr.fine_tune.device}, up to ~${pr.fine_tune.recommended_max_params_b}B params`],
  ]);
  const e = s.loaded;
  fill($("#engine"), e
    ? [
        ["Model", e.model],
        ["Backend", e.info.backend],
        ["Context (all slots)", e.info.n_ctx],
        ["Slots", e.info.n_seq_max],
        ["Requests", e.metrics.requests],
        ["Prompt tokens", `${e.metrics.prompt_tokens} (${e.metrics.cached_tokens} from prefix cache)`],
        ["Generated tokens", e.metrics.generated_tokens],
        ["Active / queued", `${e.metrics.active} / ${e.metrics.queued}`],
      ]
    : [["Model", "none loaded yet"]]);
}

async function loadLoaded() {
  try {
    const s = await api("/v1/system");
    $("#loaded-model").textContent = s.loaded ? `Loaded: ${s.loaded.model}\n${s.loaded.info.backend}` : "No model loaded";
  } catch {}
}

// ---------- boot ----------
refreshModelIds().catch((e) => addMsg("assistant error", "Cannot reach the Langtrain daemon: " + e.message));
loadLoaded();
const startView = safeGet("lt.view");
if (startView && startView in views) show(startView);
