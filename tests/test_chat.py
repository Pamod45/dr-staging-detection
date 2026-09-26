"""Chatbot: what the model is sent, the Gemini call shape, and the page behaviour, with a fake
client so no key or network is needed."""
from pathlib import Path

import cv2
import pytest
from streamlit.testing.v1 import AppTest

from src import chat, content, state
from src import config as C
from tests.helpers import fake_model_folder, fake_result, fundus

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
FACTS = Path(__file__).resolve().parent / "data" / "dr_facts_sample.md"


class FakeModels:
    def __init__(self, answer="Moderate means more than microaneurysms alone.", fail=False):
        self.answer, self.fail, self.calls = answer, fail, []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.fail:
            raise RuntimeError("quota exceeded")
        return type("R", (), {"text": self.answer})()


class FakeClient:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


def test_result_text_has_the_numbers_and_no_arrays():
    text = chat.result_text(fake_result([0.05, 0.05, 0.7, 0.1, 0.1]))
    assert "Moderate (grade 2 of 4), confidence 70.0%" in text
    assert "90.0%" in text and "threshold 35%" in text and "refer to an eye specialist" in text
    assert "[" not in text and "array" not in text


def test_withheld_grade_is_not_named_in_the_result():
    text = chat.result_text(fake_result([0.3, 0.3, 0.2, 0.1, 0.1], abstention=0.5))
    assert "Grade: not given" in text and "Most likely grade" not in text


def test_system_prompt_holds_rules_facts_and_result(monkeypatch):
    monkeypatch.setattr(content, "FACTS", FACTS)
    prompt = chat.system_prompt(fake_result([0.9, 0.08, 0.01, 0.005, 0.005]))
    assert prompt.startswith(chat.RULES)
    assert "## The five ICDR grades" in prompt and "4-2-1 rule" in prompt
    assert "# RESULT" in prompt and "not referred" in prompt
    assert "# RELIABILITY" in prompt


def test_reply_sends_history_then_question_with_system_instruction(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    client = FakeClient()
    history = [{"role": "user", "text": "hi"}, {"role": "model", "text": "hello"}]
    assert chat.reply(client, "SYSTEM", history, "what is Mild?").startswith("Moderate")
    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][-1].parts[0].text == "what is Mild?"
    assert call["config"].system_instruction == "SYSTEM"


def test_default_model(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert chat.model_name() == "gemini-3.1-flash-lite"


def test_empty_reply_is_an_error():
    with pytest.raises(RuntimeError):
        chat.reply(FakeClient(answer=""), "S", [], "q")


@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(content, "FACTS", FACTS)
    fake_model_folder(tmp_path)
    ok, buf = cv2.imencode(".png", fundus(300, 300, 140, notch=False))
    img = state.decode(buf.tobytes(), "eye.png")

    def open_page(history=None):
        at = AppTest.from_file(MAIN, default_timeout=60)
        at.session_state["shared_image"] = img
        cache = {"screening": {img.sha256: fake_result([0.05, 0.05, 0.7, 0.1, 0.1])}}
        if history is not None:
            cache["chat"] = {img.sha256: history}
        at.session_state["per_image_cache"] = cache
        at.run()
        at.switch_page("views/screening.py").run()
        return at, img
    return open_page


def test_no_key_hides_the_chat(page, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at, _ = page()
    assert not at.exception, at.exception
    assert len(at.chat_input) == 0
    assert any("not available" in i.value for i in at.info)


def test_question_is_answered_and_kept(page, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = FakeClient()
    monkeypatch.setattr(chat, "make_client", lambda key: client)
    at, img = page()
    at.chat_input[0].set_value("What does Moderate mean?").run()
    assert not at.exception, at.exception
    history = at.session_state["per_image_cache"]["chat"][img.sha256]
    assert [m["role"] for m in history] == ["user", "model"]
    assert "0 of 10" not in " ".join(c.value for c in at.caption)
    assert "1 of 10" in " ".join(c.value for c in at.caption)
    assert "# RESULT" in client.models.calls[0]["config"].system_instruction


def test_api_failure_keeps_history_unchanged(page, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(chat, "make_client", lambda key: FakeClient(fail=True))
    at, img = page()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception, at.exception
    assert any("could not answer" in e.value for e in at.error)
    assert not at.session_state["per_image_cache"].get("chat", {}).get(img.sha256)


def test_limit_stops_new_questions(page, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    full = [m for _ in range(chat.MAX_QUESTIONS)
            for m in ({"role": "user", "text": "q"}, {"role": "model", "text": "a"})]
    at, _ = page(history=full)
    assert not at.exception, at.exception
    assert len(at.chat_input) == 0
    assert any("limit" in i.value for i in at.info)


def test_suggested_question_is_sent(page, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = FakeClient()
    monkeypatch.setattr(chat, "make_client", lambda key: client)
    at, img = page()
    btn = next(b for b in at.button if b.label == chat.SUGGESTED[0])
    btn.click().run()
    assert not at.exception, at.exception
    assert client.models.calls[0]["contents"][-1].parts[0].text == chat.SUGGESTED[0]


def test_chat_works_before_any_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(content, "FACTS", FACTS)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = FakeClient(answer="Diabetic retinopathy is damage to the retina's vessels.")
    monkeypatch.setattr(chat, "make_client", lambda key: client)
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    at.switch_page("views/screening.py").run()
    assert not at.exception, at.exception
    assert [s.value for s in at.subheader] == ["Ask about diabetic retinopathy"]
    btn = next(b for b in at.button if b.label == chat.SUGGESTED_GENERAL[0])
    btn.click().run()
    assert not at.exception, at.exception
    assert [m["role"] for m in at.session_state["chat_general"]] == ["user", "model"]
    system = client.models.calls[0]["config"].system_instruction
    assert "No photograph has been graded" in system and "## The five ICDR grades" in system


def test_error_messages_are_specific():
    assert "API key" in chat.error_message(RuntimeError("403 PERMISSION_DENIED. API key not valid"))
    assert "not found" in chat.error_message(RuntimeError("404 NOT_FOUND. models/x is not found"))
    assert "limit" in chat.error_message(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert chat.error_message(RuntimeError("")) == "RuntimeError"


def test_empty_reply_names_the_finish_reason():
    class R:
        text = None
        candidates = [type("C", (), {"finish_reason": "MAX_TOKENS"})()]

    class M:
        def generate_content(self, **kw):
            return R()

    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        chat.reply(type("Cl", (), {"models": M()})(), "S", [], "q")
