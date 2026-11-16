import json

from fastapi.testclient import TestClient

import main
from knowledge_base import KnowledgeBase
from model import TinyNgramModel


def test_training_generation_and_round_trip(tmp_path):
    pairs = [
        {"prompt": "سلام", "response": "سلام دوست من."},
        {"prompt": "سلام", "response": "سلام دوست من."},
        {"prompt": "خداحافظ", "response": "روز خوبی داشته باشی."},
    ]
    first = TinyNgramModel.train(pairs)
    second = TinyNgramModel.train(pairs)
    assert first.to_dict() == second.to_dict()
    assert first.generate("سلام") == "سلام دوست من."
    path = tmp_path / "model.json"
    first.save(path)
    assert TinyNgramModel.load(path).generate("سلام") == first.generate("سلام")


def test_knowledge_base_is_separate_and_retrieves_persian_facts():
    entries = json.loads((main.ROOT / "data/knowledge_base.json").read_text(encoding="utf-8"))
    kb = KnowledgeBase(entries)
    result = kb.search("تفاوت داده آموزشی و پایگاه دانش چیست؟")
    assert result is not None
    assert "مدل" in result["answer"]
    assert kb.search("فرمول ساخت یک اژدهای نامرئی") is None
    assert len(entries) >= 40


def test_api_uses_kb_then_model_fallback(monkeypatch):
    client = TestClient(main.app)
    entries = json.loads((main.ROOT / "data/knowledge_base.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(main, "knowledge_base", KnowledgeBase(entries))
    monkeypatch.setattr(main, "model", TinyNgramModel.train([
        {"prompt": "سلام", "response": "سلام دوست من."}
    ]))
    kb_response = client.post("/api/chat", json={"message": "الگوریتم چیست؟"})
    assert kb_response.status_code == 200
    assert kb_response.json()["source"] == "knowledge_base"
    assert "گام" in kb_response.json()["response"]
    model_response = client.post("/api/chat", json={"message": "سلام"})
    assert model_response.json() == {"response": "سلام دوست من.", "source": "toy_model"}


def test_api_input_validation(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "knowledge_base", None)
    monkeypatch.setattr(main, "model", TinyNgramModel.train([
        {"prompt": "سلام", "response": "سلام دوست من."}
    ]))
    assert client.post("/api/chat", json={"message": "سلام"}).status_code == 200
    assert client.post("/api/chat", json={"message": "  "}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
    assert client.post("/api/chat", json={"other": "سلام"}).status_code == 422
    assert client.post("/api/chat", data="not-json").status_code == 422


def test_missing_model_has_safe_error(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "knowledge_base", None)
    monkeypatch.setattr(main, "model", None)
    response = client.post("/api/chat", json={"message": "سلام"})
    assert response.status_code == 503
    assert "model.json" not in response.text
    assert response.json().get("error")


def test_unexpected_error_is_sanitized(monkeypatch):
    class BrokenModel:
        def generate(self, _message):
            raise RuntimeError("secret local detail")

    client = TestClient(main.app)
    monkeypatch.setattr(main, "knowledge_base", None)
    monkeypatch.setattr(main, "model", BrokenModel())
    response = client.post("/api/chat", json={"message": "سلام"})
    assert response.status_code == 500
    assert "secret local detail" not in response.text
