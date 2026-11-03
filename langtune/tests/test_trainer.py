"""Checks that need no downloads: a tiny random Llama stands in for a real model."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from langtune.data import IGNORE
from langtune.export import _hf_to_gguf_permute
from langtune.lora import inject
from langtune.trainer import Stack


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


def test_manual_stack_matches_model_loss():
    m = tiny()
    ids = list(range(5, 30))
    labels = [IGNORE] * 10 + ids[10:]
    with torch.no_grad():
        ref = m(torch.tensor([ids]), labels=torch.tensor([labels])).loss
        s = Stack(m)
        ours = s.loss(s.run(s.embed(ids, "cpu"), 0, 4), labels)
    assert torch.allclose(ref, ours, atol=1e-5), (ref, ours)


def test_frozen_prefix_split_is_exact():
    m = tiny()
    s = Stack(m)
    ids = list(range(3, 23))
    with torch.no_grad():
        full = s.run(s.embed(ids, "cpu"), 0, 4)
        mid = s.run(s.embed(ids, "cpu"), 0, 2).half().float()  # cached as fp16
        split = s.run(mid, 2, 4)
    assert (full - split).abs().max() < 1e-2


def test_lora_starts_as_identity_and_learns():
    m = tiny()
    s = Stack(m)
    ids = list(range(10, 40))
    with torch.no_grad():
        before = s.loss(s.run(s.embed(ids, "cpu"), 0, 4), ids)
    ad = inject(m.model.layers, [2, 3], rank=4, alpha=8)
    params = [p for a in ad.values() for p in (a.lora_a, a.lora_b)]
    with torch.no_grad():
        assert torch.allclose(before, s.loss(s.run(s.embed(ids, "cpu"), 0, 4), ids), atol=1e-6)
    opt = torch.optim.AdamW(params, lr=1e-2)
    h = s.run(s.embed(ids, "cpu"), 0, 2).detach()
    for _ in range(60):
        loss = s.loss(s.run(h, 2, 4), ids)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < before.item() * 0.8


def test_qk_permutation_is_a_bijection():
    w = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3).numpy()
    p = _hf_to_gguf_permute(w, n_head=4)
    assert sorted(p[:, 0].tolist()) == sorted(w[:, 0].tolist())
    assert not (p == w).all()
