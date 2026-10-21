"""Offline tests: no network and no API keys needed."""

import pytest

import chatbot
from chatbot import FALLBACK, DocsChatbot, load_sections


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="module")
def bot():
    return DocsChatbot()


def test_sections_are_split_by_heading():
    sections = load_sections(chatbot.DEFAULT_DOCS)
    titles = {s.title for s in sections}
    assert "When is my bill due?" in titles
    assert len(sections) >= 15
    assert all(s.body for s in sections)


@pytest.mark.parametrize(
    "question, expected_title",
    [
        ("is there a fee if I pay late", "Is there a late fee?"),
        ("how do I turn on autopay", "How do I set up autopay?"),
        ("internet not working and no outage shown", "My internet is down but there is no outage listed. What should I do?"),
        ("how much is the gig plan", "What internet plans do you offer?"),
        ("I'm moving to a new apartment", "How do I move my service to a new address?"),
        ("cancel my account", "How do I cancel my service?"),
        ("my wifi is really slow", "Why is my Wi-Fi slow?"),
        ("can I use my own modem", "Do you rent equipment?"),
    ],
)
def test_questions_find_the_right_section(bot, question, expected_title):
    result = bot.answer(question)
    assert result["mode"] == "extractive"
    assert result["sources"][0].endswith(expected_title)


@pytest.mark.parametrize(
    "question, expected_title",
    [
        ("when do I have to pay my bill", "When is my bill due?"),
        ("do I get money back when the internet was out for 2 days", "Will I get a credit for an outage?"),
    ],
)
def test_paraphrases_reach_the_llm_context(bot, question, expected_title):
    """Loosely worded questions may not rank first offline, but the right
    section must be among the excerpts the LLM sees in grounded mode."""
    titles = [s.title for s, _ in bot.search(question)]
    assert expected_title in titles


def test_unrelated_question_falls_back(bot):
    result = bot.answer("what's the best pizza topping")
    assert result == {"answer": FALLBACK, "sources": [], "mode": "fallback"}


def test_grounded_mode_passes_only_retrieved_sections(bot, monkeypatch):
    seen = {}

    def fake_llm(question, sections):
        seen["titles"] = [s.title for s in sections]
        return "Your bill is due on the 15th."

    monkeypatch.setattr(chatbot, "pick_llm", lambda: fake_llm)
    result = bot.answer("when is my bill due")
    assert result["mode"] == "grounded"
    assert result["answer"] == "Your bill is due on the 15th."
    assert seen["titles"][0] == "When is my bill due?"
    assert len(result["sources"]) == len(seen["titles"]) <= chatbot.TOP_K


def test_empty_docs_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        DocsChatbot(tmp_path)


def test_anthropic_and_openai_requests_are_well_formed(bot, monkeypatch):
    calls = []

    def fake_post(url, headers, payload):
        calls.append((url, headers, payload))
        if "anthropic" in url:
            return {"content": [{"type": "text", "text": " Due on the 15th. "}]}
        return {"choices": [{"message": {"content": " Due on the 15th. "}}]}

    monkeypatch.setattr(chatbot, "_post_json", fake_post)
    sections = [s for s, _ in bot.search("when is my bill due")]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert chatbot._anthropic("when is my bill due", sections) == "Due on the 15th."
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert chatbot._openai("when is my bill due", sections) == "Due on the 15th."

    (a_url, a_headers, a_payload), (o_url, o_headers, o_payload) = calls
    assert a_headers["x-api-key"] == "test-key" and a_payload["system"] == chatbot.SYSTEM_PROMPT
    assert "When is my bill due?" in a_payload["messages"][0]["content"]
    assert o_headers["authorization"] == "Bearer test-key"
    assert o_payload["messages"][0]["role"] == "system"
