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

// Langtrain Cloud: the API key lives in this browser only.
const KEY_STORE = "lt.apiKey";
function apiKey() { return safeGet(KEY_STORE); }
function authHeaders(extra = {}) {
  const k = apiKey();
  return k ? { ...extra, authorization: "Bearer " + k } : extra;
}
let cloud = false;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: authHeaders({ "content-type": "application/json" }),
    ...opts,
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error?.message || res.statusText);
  return data;
}

// ---------- navigation ----------
const views = { chat: null, models: loadModels, tune: loadTune, system: loadSystem, billing: loadBilling };
document.querySelectorAll(".nav").forEach((b) =>
  b.addEventListener("click", () => show(b.dataset.view))
);
function show(name) {
  document.querySelectorAll(".nav").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  // Each view has its own address, so back/forward and shared links work.
  if (location.hash !== "#" + name) history.pushState(null, "", "#" + name);
  if (views[name]) views[name]();
  try { localStorage.setItem("lt.view", name); } catch {}
}

window.addEventListener("hashchange", () => {
  const name = location.hash.replace("#", "");
  const tab = document.querySelector(`[data-view="${name}"]`);
  if (name in views && tab && !tab.hidden) show(name);
});

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
// Messages of the current chat (not to be confused with window.history).
let conversation = [];
let controller = null;

$("#chat-model").addEventListener("change", (e) => { try { localStorage.setItem("lt.model", e.target.value); } catch {} });
$("#chat-clear").addEventListener("click", () => {
  conversation = [];
  $("#messages").replaceChildren(emptyChat());
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
  conversation.push({ role: "user", content: text });
  const out = addMsg("assistant", "");
  await streamChat(model, out);
});

function emptyChat() {
  const e = el("div", "empty");
  e.append(el("div", "eyebrow", "New chat"), el("p", "empty-title", "Ask anything."), el("p", "empty-body", "Your conversation stays on this server."));
  return e;
}

function emptyRow(cols, text) {
  const tr = el("tr", "table-empty");
  const td = el("td", "", text);
  td.colSpan = cols;
  tr.append(td);
  return tr;
}

function addMsg(role, text) {
  const m = el("div", "msg " + role, text);
  $("#messages").append(m);
  m.scrollIntoView({ block: "end" });
  return m;
}

async function streamChat(model, out) {
  const system = $("#chat-system").value.trim();
  const messages = system ? [{ role: "system", content: system }, ...conversation] : conversation;
  controller = new AbortController();
  $("#send").textContent = "Stop";
  const t0 = performance.now();
  let first = 0, reply = "", usage = null;
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ model, messages, stream: true, temperature: +$("#chat-temp").value, stream_options: { include_usage: true } }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const msg = (await res.json().catch(() => ({}))).error?.message || res.statusText;
      if (cloud && (res.status === 401 || res.status === 402)) {
        // Send the person to the place that fixes it.
        setTimeout(() => show("billing"), 1200);
      }
      throw new Error(msg);
    }
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
    conversation.push({ role: "assistant", content: reply });
    const secs = (performance.now() - (first || t0)) / 1000;
    if (usage) {
      $("#chat-stats").textContent =
        `${usage.completion_tokens} tokens · ${(usage.completion_tokens / Math.max(secs, 1e-3)).toFixed(1)} tok/s · ` +
        `first token ${Math.round((first || performance.now()) - t0)} ms · ` +
        `${usage.prompt_tokens_details?.cached_tokens ?? 0}/${usage.prompt_tokens} prompt tokens from cache`;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      conversation.push({ role: "assistant", content: reply });
    } else {
      out.classList.add("error");
      out.textContent = reply ? reply + "\n\n" + err.message : err.message;
      conversation.pop();
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
  $("#models-dir").textContent = sys.models_dir || "the server's model folder";
  const body = $("#models-body");
  body.replaceChildren();
  for (const m of sys.models) {
    const tr = el("tr");
    tr.append(el("td", "", m.name), el("td", "", m.size_bytes ? gb(m.size_bytes) : "-"));
    const ad = el("td");
    if (!m.adapters.length) ad.textContent = "-";
    m.adapters.forEach((a) => ad.append(el("span", "chip mint", a)));
    const act = el("td");
    const b = el("button", "btn ghost small", "Chat");
    b.onclick = async () => { await refreshModelIds(); $("#chat-model").value = m.name; show("chat"); };
    act.append(b);
    tr.append(ad, act);
    body.append(tr);
  }
  if (!sys.models.length) body.append(emptyRow(4, "No models yet. Pull one with the command above."));
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
  const tone = { succeeded: "mint", failed: "rose", cancelled: "rose", running: "sky", queued: "amber" }[j.status] || "";
  head.append(title, el("span", "chip " + tone, j.status));
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
    const c = el("button", "btn ghost small", "Cancel");
    c.onclick = () => api(`/v1/fine_tuning/jobs/${j.id}/cancel`, { method: "POST" }).then(loadJobs);
    head.append(c);
  }
  if (j.status === "succeeded" && j.fine_tuned_model) {
    const meta = el("div", "job-meta");
    meta.append(`Ready as `, el("code", "", j.fine_tuned_model), "  ");
    const b = el("button", "btn primary small", "Chat with it");
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
    ["Cores", `${hw.physical_cores} physical / ${hw.logical_cores} logical` + (hw.performance_cores ? `, ${hw.performance_cores} performance` : "")],
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
  if (cloud) {
    $("#loaded-model").textContent = apiKey() ? "Langtrain Cloud" : "Langtrain Cloud · not signed in";
    return;
  }
  try {
    const s = await api("/v1/system");
    const chip = $("#loaded-model");
    chip.textContent = s.loaded ? s.loaded.model : "No model loaded";
    chip.title = s.loaded ? `Running on ${s.loaded.info.backend}` : "";
  } catch {}
}

// ---------- segmented toggles ----------
document.querySelectorAll(".segmented[data-toggle]").forEach((group) => {
  const input = document.getElementById(group.dataset.toggle);
  group.querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      group.querySelectorAll("button").forEach((o) => o.setAttribute("aria-checked", String(o === b)));
      if (input) input.value = b.dataset.value;
    });
  });
});

// ---------- boot ----------
// ---------- billing (Langtrain Cloud) ----------
let billingConfig = null;
// Usage charges are often a fraction of a paisa, so small amounts get more decimals.
const money = (micro, cur) => {
  const v = Math.abs(micro) / 1e6;
  const digits = v > 0 && v < 1 ? 4 : 2;
  const text = v.toLocaleString(cur === "USD" ? "en-US" : "en-IN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return (micro < 0 ? "−" : "") + (cur === "USD" ? "$" : "₹") + text;
};
const when = (unix) => (unix ? new Date(unix * 1000).toLocaleString() : "never");
const KIND_LABEL = { topup: "Credit purchase", bonus: "Bonus credit", grant: "Credit from Langtrain", charge: "Usage", refund: "Refund", expiry: "Expired credit" };

function toast(text) {
  const t = el("div", "toast", text);
  t.setAttribute("role", "status");
  document.body.append(t);
  setTimeout(() => t.remove(), 5000);
}

function showNewKey(secret) {
  $("#bill-newkey").hidden = false;
  $("#bill-newkey-value").textContent = secret;
}

$("#bill-copy").addEventListener("click", async () => {
  const v = $("#bill-newkey-value").textContent;
  try {
    await navigator.clipboard.writeText(v);
    toast("API key copied");
  } catch {
    const r = document.createRange();
    r.selectNodeContents($("#bill-newkey-value"));
    getSelection().removeAllRanges();
    getSelection().addRange(r);
  }
});

$("#bill-signup").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#bill-signin-error").textContent = "";
  try {
    const r = await api("/v1/signup", { method: "POST", body: { email: $("#bill-email").value.trim(), currency: $("#bill-currency").value } });
    try { localStorage.setItem(KEY_STORE, r.api_key); } catch {}
    showNewKey(r.api_key);
    loadBilling();
    loadLoaded();
  } catch (err) {
    $("#bill-signin-error").textContent = err.message;
  }
});

$("#bill-usekey").addEventListener("submit", async (e) => {
  e.preventDefault();
  const k = $("#bill-key").value.trim();
  if (!k) return;
  try { localStorage.setItem(KEY_STORE, k); } catch {}
  $("#bill-key").value = "";
  loadBilling();
});

$("#bill-signout").addEventListener("click", () => {
  try { localStorage.removeItem(KEY_STORE); } catch {}
  $("#bill-newkey").hidden = true;
  loadBilling();
});

$("#bill-keyform").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const r = await api("/v1/keys", { method: "POST", body: { name: $("#bill-keyname").value.trim() } });
    $("#bill-keyname").value = "";
    showNewKey(r.api_key);
    loadBilling();
  } catch (err) {
    toast(err.message);
  }
});

function loadRazorpay() {
  return new Promise((resolve, reject) => {
    if (window.Razorpay) return resolve();
    const s = document.createElement("script");
    s.src = "https://checkout.razorpay.com/v1/checkout.js";
    s.onload = resolve;
    s.onerror = () => reject(new Error("Could not load Razorpay. Check your connection and try again."));
    document.head.append(s);
  });
}

async function buy(pack) {
  $("#bill-pay-error").textContent = "";
  try {
    const r = await api("/v1/billing/checkout", { method: "POST", body: { pack_id: pack.id } });
    if (r.provider === "stripe") {
      location.href = r.url;
      return;
    }
    await loadRazorpay();
    const rzp = new window.Razorpay({
      key: r.key_id,
      order_id: r.order_id,
      amount: r.amount,
      currency: r.currency,
      name: r.name,
      description: r.description,
      prefill: { email: r.email },
      theme: { color: getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#3a5bd9" },
      handler: () => paymentReceived(),
    });
    rzp.on("payment.failed", (resp) => {
      $("#bill-pay-error").textContent = "Payment failed: " + (resp.error?.description || "please try another method.");
    });
    rzp.open();
  } catch (err) {
    $("#bill-pay-error").textContent = err.message;
  }
}

// Credits are added by the provider's webhook, which can lag the browser by a few seconds.
async function paymentReceived() {
  toast("Payment received. Your credits will appear in a few seconds.");
  const before = (await api("/v1/account").catch(() => null))?.balance;
  for (let i = 0; i < 15; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const now = (await api("/v1/account").catch(() => null))?.balance;
    if (now !== undefined && now !== before) break;
  }
  loadBilling();
}

async function loadBilling() {
  if (!cloud) return;
  billingConfig = billingConfig || (await api("/v1/billing/config").catch(() => null));
  const signedIn = !!apiKey();
  let acct = null;
  if (signedIn) {
    acct = await api("/v1/account").catch((err) => {
      $("#bill-signin-error").textContent = err.message + ". Paste a valid key or create an account.";
      try { localStorage.removeItem(KEY_STORE); } catch {}
      return null;
    });
  }
  $("#bill-signin").hidden = !!acct;
  $("#bill-account").hidden = !acct;
  $("#bill-signup").hidden = !billingConfig?.open_signup;
  if (!acct) return;

  const cur = acct.account.currency;
  $("#bill-balance").textContent = money(acct.balance, cur);
  $("#bill-held").textContent = acct.held > 0 ? `${money(acct.held, cur)} reserved by requests that are running` : "";
  $("#bill-who").textContent = acct.account.email;
  const p = acct.prices;
  const sym = cur === "USD" ? "$" : "₹";
  $("#bill-prices").replaceChildren(
    ...[["Input tokens", p.input_per_m], ["Cached input tokens", p.cached_input_per_m], ["Output tokens", p.output_per_m]].map(([label, v]) => {
      const tr = el("tr");
      tr.append(el("td", "", label), el("td", "dim num", sym + v));
      return tr;
    })
  );

  const packs = (billingConfig?.packs || []).filter((x) => x.currency === cur);
  const box = $("#bill-packs");
  box.replaceChildren();
  const providers = billingConfig?.providers || {};
  const canPay = providers.razorpay || providers.stripe;
  const tiers = ["Small", "Medium", "Large", "Extra large"];
  const fmt = (v) => sym + v.toLocaleString(cur === "USD" ? "en-US" : "en-IN");
  packs.forEach((pk, i) => {
    const card = el("div", "pack-card");
    const credit = pk.amount + (pk.amount * pk.bonus_pct) / 100;
    const price = el("div", "price");
    price.append(el("strong", "", fmt(pk.amount)));
    const b = el("button", "btn ghost", "Buy credits");
    b.type = "button";
    b.disabled = !canPay;
    b.onclick = () => buy(pk);
    card.append(
      el("div", "eyebrow-tier", tiers[i] || "Pack"),
      el("div", "credit", `${fmt(credit)} credit`),
      price,
      el("div", "note", pk.bonus_pct ? `Includes a ${pk.bonus_pct}% bonus` : "Pay as you go"),
      b
    );
    box.append(card);
  });
  $("#bill-pay-note").textContent = canPay
    ? cur === "INR" ? "Pay with UPI, cards or netbanking. GST invoice by email." : "Pay by card. Receipt by email."
    : "Online payment is not set up on this server yet. Contact the operator to add credit.";

  const [keys, usage, ledger] = await Promise.all([
    api("/v1/keys").catch(() => ({ data: [] })),
    api("/v1/usage").catch(() => ({ data: [] })),
    api("/v1/billing/ledger").catch(() => ({ data: [] })),
  ]);
  const kb = $("#bill-keys");
  kb.replaceChildren(...keys.data.filter((k) => !k.revoked).map((k) => {
    const tr = el("tr");
    const keyCell = el("td");
    keyCell.append(el("code", "", k.prefix + "…"));
    tr.append(el("td", "", k.name), keyCell, el("td", "dim", when(k.created_at)), el("td", "dim", when(k.last_used_at)));
    const td = el("td");
    if (k.id === keys.current) {
      td.append(el("span", "chip mint", "this browser"));
    } else {
      const r = el("button", "btn ghost small danger", "Revoke");
      r.onclick = async () => {
        if (r.dataset.confirm !== "1") { r.dataset.confirm = "1"; r.textContent = "Click again to revoke"; return; }
        await api(`/v1/keys/${k.id}`, { method: "DELETE" }).catch((err) => toast(err.message));
        loadBilling();
      };
      td.append(r);
    }
    tr.append(td);
    return tr;
  }));
  const ub = $("#bill-usage");
  ub.replaceChildren(...usage.data.slice(0, 50).map((u) => {
    const tr = el("tr");
    const prompt = u.cached_tokens ? `${u.prompt_tokens} (${u.cached_tokens} cached)` : String(u.prompt_tokens);
    tr.append(el("td", "dim", when(u.created_at)), el("td", "", u.model), el("td", "num", prompt), el("td", "num", String(u.completion_tokens)), el("td", "amount-neg", money(u.cost, cur)));
    return tr;
  }));
  if (!usage.data.length) ub.append(emptyRow(5, "No requests yet. Your API calls will show up here."));
  const lb = $("#bill-ledger");
  lb.replaceChildren(...ledger.data.map((e) => {
    const tr = el("tr");
    const label = KIND_LABEL[e.kind] || e.kind;
    tr.append(el("td", "dim", when(e.created_at)), el("td", "", e.note ? `${label} · ${e.note}` : label), el("td", e.amount > 0 ? "amount-pos" : "amount-neg", (e.amount > 0 ? "+" : "") + money(e.amount, cur)));
    return tr;
  }));
  if (!ledger.data.length) lb.append(emptyRow(3, "No credit yet. Buy a pack above to get started."));
}

// ---------- boot ----------
async function boot() {
  const sys = await api("/v1/system").catch(() => null);
  cloud = !!sys?.cloud;
  if (cloud) {
    // Customers see Chat, Models and Billing; hardware and training are operator tools.
    document.querySelector('[data-view="billing"]').hidden = false;
    document.querySelector('[data-view="tune"]').hidden = true;
    document.querySelector('[data-view="system"]').hidden = true;
  }
  refreshModelIds().catch((e) => addMsg("assistant error", "Cannot reach the Langtrain server: " + e.message));
  loadLoaded();
  const params = new URLSearchParams(location.search);
  if (cloud && params.get("checkout")) {
    history.replaceState(null, "", location.pathname + location.hash);
    if (params.get("checkout") === "success") paymentReceived();
    show("billing");
    return;
  }
  const hash = location.hash.replace("#", "");
  const startView = hash in views ? hash : safeGet("lt.view");
  const allowed = (v) => v && v in views && !document.querySelector(`[data-view="${v}"]`)?.hidden;
  if (allowed(startView)) show(startView);
  else if (cloud && !apiKey()) show("billing");
}
boot();
