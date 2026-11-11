"""Write trained LoRA weights as a GGUF adapter that llama.cpp and the
Langtrain native engine both load."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .lora import PROJ_TO_GGUF, LoRALinear


def _hf_to_gguf_permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp's converter permutes Llama q/k rows; LoRA B must match."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write_gguf_lora(
    path: str | Path,
    adapters: dict[tuple[int, str], LoRALinear],
    arch: str,
    alpha: float,
    n_head: int,
    n_head_kv: int,
    name: str = "langtune-adapter",
) -> Path:
    import gguf

    path = Path(path)
    w = gguf.GGUFWriter(str(path), arch)
    w.add_string("general.type", "adapter")
    w.add_string("general.name", name)
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(alpha))
    permute_qk = arch in ("llama", "mistral", "granite")

    for (layer, proj), mod in sorted(adapters.items()):
        a = mod.lora_a.detach().float().cpu().numpy()  # [rank, in]
        b = mod.lora_b.detach().float().cpu().numpy()  # [out, rank]
        if permute_qk and proj == "q_proj":
            b = _hf_to_gguf_permute(b, n_head)
        elif permute_qk and proj == "k_proj":
            b = _hf_to_gguf_permute(b, n_head_kv)
        base = f"blk.{layer}.{PROJ_TO_GGUF[proj]}.weight"
        w.add_tensor(f"{base}.lora_a", np.ascontiguousarray(a))
        w.add_tensor(f"{base}.lora_b", np.ascontiguousarray(b))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path
