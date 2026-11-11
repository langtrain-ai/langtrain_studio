"""Training data: JSONL in chat, prompt/completion or plain-text form.

Loss is applied to assistant tokens only (for chat and prompt/completion
data), so the model learns to answer rather than to echo prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IGNORE = -100


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]

    @property
    def n_trained(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE)


def _render(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    # Newer transformers return a BatchEncoding-like mapping.
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def chat_example(tokenizer, messages: list[dict], max_len: int) -> Example:
    """Tokenize a conversation; label only the assistant turns."""
    full = _render(tokenizer, messages, add_generation_prompt=False)
    labels = [IGNORE] * len(full)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(_render(tokenizer, messages[:i], add_generation_prompt=True))
        end = len(_render(tokenizer, messages[: i + 1], add_generation_prompt=False))
        for j in range(start, min(end, len(full))):
            labels[j] = full[j]
    return Example(full[:max_len], labels[:max_len])


def load_examples(path: str | Path, tokenizer, max_len: int) -> list[Example]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from None
            if "messages" in row:
                ex = chat_example(tokenizer, row["messages"], max_len)
            elif "prompt" in row and "completion" in row:
                msgs = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]
                ex = chat_example(tokenizer, msgs, max_len)
            elif "text" in row:
                ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"][:max_len]
                ex = Example(ids, list(ids))
            else:
                raise ValueError(f"{path}:{n}: expected 'messages', 'prompt'+'completion' or 'text'")
            if ex.n_trained > 1:
                examples.append(ex)
    if not examples:
        raise ValueError(f"{path}: no usable training examples")
    return examples
