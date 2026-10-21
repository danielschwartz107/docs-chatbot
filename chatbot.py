"""Docs chatbot: answers customer questions from a folder of Markdown help docs.

How it works
1. Every Markdown file in the docs folder is split into sections at "## " headings.
2. Questions are matched to sections with TF-IDF retrieval (scikit-learn), offline.
3. The answer is built from the best-matching sections, and the sources are always cited.
   - With no API key, the bot replies with the matching section text (extractive mode).
   - With ANTHROPIC_API_KEY or OPENAI_API_KEY set, an LLM writes a short answer
     using ONLY the retrieved sections (grounded mode).
4. If nothing in the docs is relevant enough, the bot says so and offers a hand-off
   to a human instead of guessing.

Usage:
    python chatbot.py                       # interactive chat
    python chatbot.py --ask "When is my bill due?"
    python chatbot.py --docs path/to/docs --ask "..."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DOCS = Path(__file__).parent / "docs"
MIN_SCORE = 0.12          # below this, the bot says it doesn't know
TOP_K = 3                 # sections passed to the LLM in grounded mode
FALLBACK = (
    "I couldn't find that in our help docs. "
    "Would you like me to connect you with a support agent?"
)


# A short stop-word list on purpose: scikit-learn's built-in English list drops
# words like "bill" and "out", which matter for a support bot.
STOP_WORDS = set(
    "a an the and or but if of to in on at for with from by is are was were be been "
    "am do does did i me my we our you your it its this that what which who how when "
    "where why can could would should will there their them they have has had not no "
    "so as up about any some just".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ment", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS]


@dataclass
class Section:
    source: str   # e.g. "billing.md"
    title: str    # the "## " heading
    body: str

    @property
    def citation(self) -> str:
        return f"{self.source} > {self.title}"


def load_sections(docs_dir: Path) -> list[Section]:
    """Split every .md file into sections at level-2 headings."""
    sections: list[Section] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parts = re.split(r"^## +", text, flags=re.MULTILINE)
        for part in parts[1:]:                       # parts[0] is the file title
            title, _, body = part.partition("\n")
            body = body.strip()
            if body:
                sections.append(Section(path.name, title.strip(), body))
    if not sections:
        raise ValueError(f"No '## ' sections found in Markdown files under {docs_dir}")
    return sections


class DocsChatbot:
    def __init__(self, docs_dir: Path = DEFAULT_DOCS, min_score: float = MIN_SCORE):
        self.sections = load_sections(docs_dir)
        self.min_score = min_score
        # The heading is repeated so question-style headings carry extra weight.
        corpus = [f"{s.title} {s.title} {s.body}" for s in self.sections]
        # Two views of the text: whole words (with a light stemmer) and character
        # n-grams, which catch spelling variants like "pay" / "payment".
        self.word_vec = TfidfVectorizer(
            tokenizer=_tokens, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_matrix = self.word_vec.fit_transform(corpus)
        self.char_matrix = self.char_vec.fit_transform(corpus)

    def _scores(self, question: str):
        word = cosine_similarity(self.word_vec.transform([question]), self.word_matrix)[0]
        char = cosine_similarity(self.char_vec.transform([question]), self.char_matrix)[0]
        return 0.6 * word + 0.4 * char

    def search(self, question: str, k: int = TOP_K) -> list[tuple[Section, float]]:
        if not _tokens(question):
            return []
        scores = self._scores(question)
        ranked = sorted(zip(self.sections, scores), key=lambda p: p[1], reverse=True)
        return [(s, float(score)) for s, score in ranked[:k] if score >= self.min_score]

    def answer(self, question: str) -> dict:
        """Return {"answer": str, "sources": [citations], "mode": str}."""
        hits = self.search(question)
        if not hits:
            return {"answer": FALLBACK, "sources": [], "mode": "fallback"}

        llm = pick_llm()
        if llm:
            text = llm(question, [s for s, _ in hits])
            return {"answer": text, "sources": [s.citation for s, _ in hits], "mode": "grounded"}

        # Extractive mode: give the best section, and if the runner-up scored almost
        # as well, include it too so a near-miss still gets the customer their answer.
        (best, top), rest = hits[0], hits[1:]
        picked = [best] + [s for s, score in rest[:1] if score >= 0.8 * top]
        text = best.body
        if len(picked) > 1:
            text += f"\n\nAlso relevant ({picked[1].title}):\n{picked[1].body}"
        return {"answer": text, "sources": [s.citation for s in picked], "mode": "extractive"}


# ---------- optional LLM answer writing (standard library only) ----------

SYSTEM_PROMPT = (
    "You are a customer-support assistant. Answer the customer's question using ONLY "
    "the help-doc excerpts provided. Be brief and friendly (2-4 sentences). If the "
    "excerpts don't contain the answer, say you're not sure and offer to connect them "
    "with a support agent. Never invent prices, dates, or policies."
)


def _context(sections: list[Section]) -> str:
    return "\n\n".join(f"[{s.citation}]\n{s.body}" for s in sections)


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={**headers, "content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _anthropic(question: str, sections: list[Section]) -> str:
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        {
            "model": os.environ.get("CHATBOT_MODEL", "claude-haiku-4-5"),
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Help docs:\n{_context(sections)}\n\nQuestion: {question}"}],
        },
    )
    return data["content"][0]["text"].strip()


def _openai(question: str, sections: list[Section]) -> str:
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        {
            "model": os.environ.get("CHATBOT_MODEL", "gpt-4o-mini"),
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Help docs:\n{_context(sections)}\n\nQuestion: {question}"},
            ],
        },
    )
    return data["choices"][0]["message"]["content"].strip()


def pick_llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _anthropic
    if os.environ.get("OPENAI_API_KEY"):
        return _openai
    return None


# ---------- command line ----------

def format_reply(result: dict) -> str:
    out = result["answer"]
    if result["sources"]:
        out += "\n\nSources: " + "; ".join(result["sources"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer questions from Markdown help docs.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="folder of .md help docs")
    parser.add_argument("--ask", help="ask one question and exit")
    args = parser.parse_args()

    bot = DocsChatbot(args.docs)
    mode = "grounded (LLM)" if pick_llm() else "extractive (offline)"
    if args.ask:
        print(format_reply(bot.answer(args.ask)))
        return

    print(f"Loaded {len(bot.sections)} help sections. Mode: {mode}. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print("\nBot: " + format_reply(bot.answer(q)) + "\n")


if __name__ == "__main__":
    main()
