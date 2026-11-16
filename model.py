"""A tiny, deterministic word n-gram language model trained from scratch.

This is an educational toy, not a semantic assistant. It learns local token
transitions from the supplied dialogue examples and has no external weights.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
START = "<s>"
END = "</s>"
MODEL_VERSION = 1


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


class TinyNgramModel:
    """An n-gram next-token model; ties are broken lexicographically."""

    def __init__(self, order: int = 3, counts: dict[str, dict[str, int]] | None = None):
        if order < 2 or order > 5:
            raise ValueError("order must be between 2 and 5")
        self.order = order
        self.counts = counts or {}

    @classmethod
    def train(cls, dialogues: Iterable[dict[str, str]], order: int = 3) -> "TinyNgramModel":
        model = cls(order=order)
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for item in dialogues:
            prompt = item["prompt"].strip()
            response = item["response"].strip()
            if not prompt or not response:
                continue
            tokens = [START] * (order - 1) + tokenize(f"کاربر: {prompt} دستیار: {response}") + [END]
            for index in range(order - 1, len(tokens)):
                context = tokens[index - (order - 1):index]
                counts["\u0001".join(context)][tokens[index]] += 1
        model.counts = {
            context: dict(sorted(next_tokens.items()))
            for context, next_tokens in sorted(counts.items())
        }
        if not model.counts:
            raise ValueError("training data contains no usable dialogue pairs")
        return model

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        if max_tokens < 1 or max_tokens > 128:
            raise ValueError("max_tokens must be between 1 and 128")
        context_tokens = tokenize(f"کاربر: {prompt.strip()} دستیار:")
        generated: list[str] = []
        history = [START] * (self.order - 1) + context_tokens
        for _ in range(max_tokens):
            next_token = None
            for width in range(self.order - 1, 0, -1):
                key = "\u0001".join(history[-width:])
                options = self.counts.get(key)
                if options:
                    # Stable greedy decoding: most frequent token, lexical tie-break.
                    next_token = min(options, key=lambda token: (-options[token], token))
                    break
            if next_token is None or next_token == END:
                break
            generated.append(next_token)
            history.append(next_token)
        return detokenize(generated).strip()

    def to_dict(self) -> dict:
        return {"version": MODEL_VERSION, "order": self.order, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: dict) -> "TinyNgramModel":
        if data.get("version") != MODEL_VERSION or not isinstance(data.get("counts"), dict):
            raise ValueError("unsupported or invalid model file")
        return cls(order=int(data["order"]), counts=data["counts"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "TinyNgramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return re.sub(r"\s+([،؛؟.!,:])", r"\1", text)
