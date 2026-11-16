"""Deterministic local retrieval over a separate, hand-written Persian knowledge base.

This component does not train or modify the language model. It only returns an
entry when enough normalized query terms overlap with its keywords/question.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
STOPWORDS = {
    "چیست", "چیه", "یعنی", "درباره", "برای", "از", "به", "در", "را",
    "و", "یا", "با", "یک", "این", "آن", "چه", "چطور", "چگونه",
    "است", "هست", "می", "شود", "کنند", "کنیم", "لطفا", "لطفاً",
}


def normalize(text: str) -> str:
    return (text.replace("ي", "ی").replace("ك", "ک")
            .replace("ۀ", "ه").replace("ة", "ه").lower())


def terms(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(normalize(text)) if token not in STOPWORDS and len(token) > 1}


class KnowledgeBase:
    def __init__(self, entries: list[dict]):
        if not isinstance(entries, list) or not entries:
            raise ValueError("knowledge base must contain entries")
        self.entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each knowledge entry must be an object")
            question = entry.get("question")
            answer = entry.get("answer")
            keywords = entry.get("keywords", [])
            if not isinstance(question, str) or not question.strip():
                raise ValueError("knowledge entry question is required")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("knowledge entry answer is required")
            if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
                raise ValueError("knowledge entry keywords must be strings")
            searchable = terms(question + " " + " ".join(keywords))
            if not searchable:
                raise ValueError("knowledge entry needs searchable terms")
            self.entries.append({"question": question, "answer": answer, "terms": searchable})

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, threshold: float = 0.34) -> dict | None:
        query_terms = terms(query)
        if len(query_terms) < 2:
            return None
        best = None
        best_score = 0.0
        for entry in self.entries:
            overlap = query_terms & entry["terms"]
            # Require two matching content terms; avoid answering from a lone common word.
            if len(overlap) < 2:
                continue
            score = len(overlap) / len(query_terms)
            if score > best_score:
                best, best_score = entry, score
        if best is None or best_score < threshold:
            return None
        return {"answer": best["answer"], "question": best["question"], "score": best_score}
