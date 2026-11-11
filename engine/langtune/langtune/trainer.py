"""The LangTune training loop.

Pipeline:
1. Load the frozen base model straight from the GGUF file the server uses.
2. Wrap the top K layers' projections with LoRA.
3. Frozen-prefix cache: run every example through the bottom L-K layers
   once and keep those hidden states (fp16). Every training step then runs
   forward and backward through the top K layers only.
4. Compute the LM head only at labeled positions (a large saving with
   100k+ token vocabularies and assistant-only loss).
5. Export the adapter as GGUF and report progress as JSON lines.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import IGNORE, Example, load_examples
from .export import write_gguf_lora
from .lora import inject


def emit(kind: str, **fields) -> None:
    print(json.dumps({"type": kind, **fields}), flush=True)


@dataclass
class Config:
    base_gguf: str
    train_file: str
    out_dir: str
    adapter_name: str = "adapter"
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    train_layers: int | None = None
    max_seq_len: int = 1024
    tokens_per_step: int = 2048
    device: str = "auto"
    threads: int | None = None
    seed: int = 0
    checkpoint_every: int = 50
    resume: bool = True
    validation_file: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        raw = json.loads(Path(path).read_text())
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and v is not None}
        return cls(**known)


def lower_priority() -> None:
    """Training is a background job: never make the laptop feel slow."""
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL = 0x4000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
        else:
            os.nice(10)
    except Exception:
        pass


def pick_device(name: str) -> torch.device:
    if name in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_base(gguf_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = Path(gguf_path)
    tok = AutoTokenizer.from_pretrained(str(p.parent), gguf_file=p.name)
    model = AutoModelForCausalLM.from_pretrained(str(p.parent), gguf_file=p.name, dtype=torch.float32)
    if tok.chat_template is None:
        import gguf

        r = gguf.GGUFReader(str(p))
        f = r.fields.get("tokenizer.chat_template")
        if f is not None:
            tok.chat_template = bytes(f.parts[f.data[0]]).decode()
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, tok


class Stack:
    """Runs decoder layers manually so training can start mid-network."""

    def __init__(self, model):
        self.inner = model.model
        self.layers = self.inner.layers
        self.lm_head = model.lm_head

    def run(self, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        emb = self.inner.rotary_emb(hidden, position_ids=pos)
        for layer in self.layers[start:end]:
            out = layer(hidden, attention_mask=None, position_ids=pos, position_embeddings=emb)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def embed(self, ids: list[int], device) -> torch.Tensor:
        return self.inner.embed_tokens(torch.tensor([ids], device=device))

    def loss(self, hidden: torch.Tensor, labels: list[int]) -> torch.Tensor:
        """Mean next-token loss at labeled positions only."""
        tgt = torch.tensor(labels[1:], device=hidden.device)
        keep = (tgt != IGNORE).nonzero(as_tuple=True)[0]
        h = self.inner.norm(hidden[0, :-1][keep])
        logits = self.lm_head(h).float()
        return F.cross_entropy(logits, tgt[keep])


def train(cfg: Config) -> Path:
    lower_priority()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    emit("status", message=f"loading {Path(cfg.base_gguf).name} on {device.type}")
    t0 = time.time()
    model, tok = load_base(cfg.base_gguf, device)
    mcfg = model.config
    n_layers = mcfg.num_hidden_layers
    k = cfg.train_layers or (n_layers if device.type != "cpu" else max(4, min(n_layers, n_layers // 3)))
    k = max(1, min(k, n_layers))
    split = n_layers - k
    emit("status", message=f"loaded in {time.time() - t0:.1f}s; training top {k}/{n_layers} layers", train_layers=k, n_layers=n_layers)

    examples = load_examples(cfg.train_file, tok, cfg.max_seq_len)
    n_tok = sum(len(e.input_ids) for e in examples)
    n_trained = sum(e.n_trained for e in examples)
    emit("status", message=f"{len(examples)} examples, {n_tok} tokens ({n_trained} trained)")

    adapters = inject(model.model.layers, list(range(split, n_layers)), cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    params = [p for m in adapters.values() for p in (m.lora_a, m.lora_b)]
    stack = Stack(model)

    # Frozen-prefix activation cache.
    cache: list[torch.Tensor] = []
    if split > 0:
        t = time.time()
        with torch.inference_mode():
            for i, ex in enumerate(examples):
                h = stack.run(stack.embed(ex.input_ids, device), 0, split)
                cache.append(h.to(torch.float16).cpu())
                if i % 16 == 0 or i == len(examples) - 1:
                    emit("progress", phase="cache", step=i + 1, total_steps=len(examples), tokens_per_sec=sum(len(e.input_ids) for e in examples[: i + 1]) / max(time.time() - t, 1e-6))
        cache_mb = sum(c.numel() * 2 for c in cache) / 2**20
        emit("status", message=f"cached layer-{split} activations ({cache_mb:.0f} MB) in {time.time() - t:.1f}s")

    def hidden_for(i: int) -> torch.Tensor:
        if split > 0:
            return cache[i].to(device=device, dtype=torch.float32)
        return stack.embed(examples[i].input_ids, device)

    # Steps are sized by token budget, so short and long examples weigh fairly.
    order = list(range(len(examples)))
    steps_per_epoch = max(1, math.ceil(n_tok / cfg.tokens_per_step))
    total_steps = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))
    )

    step = 0
    ckpt = out / "checkpoint.pt"
    if cfg.resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        for key, m in adapters.items():
            m.lora_a.data.copy_(state["lora"][f"{key[0]}.{key[1]}.a"])
            m.lora_b.data.copy_(state["lora"][f"{key[0]}.{key[1]}.b"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        emit("status", message=f"resumed from step {step}")

    def save_checkpoint() -> None:
        lora = {}
        for key, m in adapters.items():
            lora[f"{key[0]}.{key[1]}.a"] = m.lora_a.detach().cpu()
            lora[f"{key[0]}.{key[1]}.b"] = m.lora_b.detach().cpu()
        torch.save({"lora": lora, "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ckpt)

    model.train()
    t_train = time.time()
    tokens_done = 0
    first_step = step
    for epoch in range(step // steps_per_epoch, cfg.epochs):
        random.shuffle(order)
        batch_tokens, batch_loss, batch_n = 0, 0.0, 0
        for idx in order:
            ex: Example = examples[idx]
            h = stack.run(hidden_for(idx), split, n_layers)
            loss = stack.loss(h, ex.labels)
            # Weight each example by its share of the step's token budget.
            (loss * len(ex.input_ids) / cfg.tokens_per_step).backward()
            batch_tokens += len(ex.input_ids)
            batch_loss += loss.item() * len(ex.input_ids)
            batch_n += len(ex.input_ids)
            if batch_tokens < cfg.tokens_per_step and idx != order[-1]:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens_done += batch_tokens
            elapsed = time.time() - t_train
            tps = tokens_done / max(elapsed, 1e-6)
            eta = (total_steps - step) * elapsed / max(step - first_step, 1)
            if step == min(3, total_steps):
                emit("estimate", tokens_per_sec=tps, eta_seconds=eta, total_steps=total_steps)
            emit("progress", phase="train", epoch=epoch + 1, step=step, total_steps=total_steps,
                 loss=batch_loss / max(batch_n, 1), lr=sched.get_last_lr()[0], tokens_per_sec=tps, eta_seconds=eta)
            batch_tokens, batch_loss, batch_n = 0, 0.0, 0
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                save_checkpoint()
            if step >= total_steps:
                break

    path = write_gguf_lora(
        out / f"{cfg.adapter_name}.gguf",
        adapters,
        arch=_gguf_arch(cfg.base_gguf),
        alpha=cfg.lora_alpha,
        n_head=mcfg.num_attention_heads,
        n_head_kv=getattr(mcfg, "num_key_value_heads", mcfg.num_attention_heads),
        name=cfg.adapter_name,
    )
    if ckpt.exists():
        ckpt.unlink()
    emit("done", adapter_path=str(path), steps=step, seconds=time.time() - t0)
    return path


def _gguf_arch(path: str) -> str:
    import gguf

    r = gguf.GGUFReader(path)
    f = r.fields["general.architecture"]
    return bytes(f.parts[f.data[0]]).decode()


def main_train(config_path: str) -> int:
    try:
        train(Config.from_json(config_path))
        return 0
    except Exception as e:  # reported to the job API, then a non-zero exit
        emit("error", message=f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1
