"""Stage 10 chat panel stub — answers from the fact sheet by keyword
match, no model call. `ChatBackend` is the interface; the real Muse
Wonder client plugs in later by satisfying the same `answer()` signature
(question in, {"answer", "citations"} out) - nothing in serve.py or the
frontend needs to change when that swap happens.
"""
import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "and", "or", "but", "this", "that", "it", "its", "as", "me", "tell",
    "what", "which", "who", "how", "do", "does", "did", "i", "you",
}

CATEGORY_KEYWORDS = {
    "zone": ["zone", "busiest", "dwell", "entry", "entered"],
    "density": ["density", "crowd", "crowded", "dense", "fruin"],
    "contact": ["contact", "close", "proximity"],
    "outlier": ["outlier", "strange", "unusual", "weird", "anomaly"],
    "od": ["origin", "destination", "route", "path", "direct"],
    "group": ["group", "cluster", "together"],
    "proxemics": ["hall", "personal space", "intimate", "social band", "band"],
}


class ChatBackend:
    def answer(self, question, fact_sheet):
        """question: str. fact_sheet: list of fact dicts (fact_sheet.py
        shape). Returns {"answer": str, "citations": [fact_id, ...]}.
        """
        raise NotImplementedError


class StubChatBackend(ChatBackend):
    """Keyword match against each fact's category + claim text; the
    best-overlap fact wins and is cited. No fact sheet match -> an
    honest "don't know", never a guess.
    """

    def answer(self, question, fact_sheet):
        if not fact_sheet:
            return {"answer": "This clip has no computed facts yet.", "citations": []}

        q_words = set(re.findall(r"[a-z]+", question.lower())) - STOPWORDS
        best, best_score = None, 0
        for fact in fact_sheet:
            score = 0
            for cat_words in CATEGORY_KEYWORDS.get(fact["category"], []):
                if cat_words in question.lower():
                    score += 2
            claim_words = set(re.findall(r"[a-z]+", fact["claim"].lower())) - STOPWORDS
            score += len(q_words & claim_words)
            if score > best_score:
                best, best_score = fact, score

        if best is None or best_score < 2:
            return {
                "answer": "I don't have a fact covering that in this clip — try asking about "
                          "zones, density, contact, outliers, groups, or OD paths.",
                "citations": [],
            }
        return {"answer": best["claim"], "citations": [best["fact_id"]]}


NO_FACT_ANSWER = (
    "I don't have a computed fact covering that in this clip — try asking about "
    "zones, density, contact, outliers, groups, or OD paths.")

CHAT_PROMPT = """You answer questions about one video clip that has already been \
analysed. You may use ONLY the facts listed below. They are the complete set of \
measurements taken from this clip.

Facts:
{facts}

Question: {question}

Rules:
1. Answer only from the facts above. Never estimate, infer a number, or describe \
anything the facts do not state.
2. Cite the fact_id of every fact your answer rests on.
3. If no fact covers the question, set answerable to false, leave citations empty, \
and say in one sentence what the facts do cover that is closest.
4. Two sentences at most. Plain words. No markdown.

Reply with JSON only, in this exact shape:
{{"answerable": true, "answer": "...", "citations": ["<fact_id>"]}}"""


class MuseWonderChatBackend(ChatBackend):
    """The question box, answered by muse-wonder-1.0 over the fact sheet.

    Same `answer()` interface and the same fact-sheet-only context the stub
    had (never raw tracks, per the spec's stage 7 risk note), so `serve.py`
    and the frontend are untouched by the swap.

    Two guards sit between the model and the reader, and they are the reason
    this is safe to demo:

    * **Every citation is checked against the fact sheet** and silently
      dropped if the ID does not exist — the same rule `report_verifier.py`
      applies to the report, applied here.
    * **A confident answer with no surviving citation is refused**, not
      shipped. If the model claims the question is answerable and then cites
      nothing real, the reader gets the honest no-fact line instead. That is
      the one failure mode that would otherwise put an uncited number on
      screen beside cited ones.

    `fallback` is used whenever the model cannot be reached, so the box
    degrades to keyword matching rather than to an error.
    """

    def __init__(self, client=None, fallback=None):
        self.client = client
        self.fallback = fallback or StubChatBackend()

    def answer(self, question, fact_sheet):
        if not fact_sheet:
            return {"answer": "This clip has no computed facts yet.",
                    "citations": []}
        if self.client is None:
            return self.fallback.answer(question, fact_sheet)

        from proxemix.muse import MuseError, render_fact_sheet

        prompt = CHAT_PROMPT.format(facts=render_fact_sheet(fact_sheet),
                                    question=str(question).strip()[:500])
        try:
            reply = self.client.complete_json(prompt, note="chat",
                                              max_output_tokens=1500)
        except MuseError:
            return self.fallback.answer(question, fact_sheet)

        valid = {f["fact_id"] for f in fact_sheet}
        cited = [c for c in (reply.get("citations") or []) if c in valid]
        text = str(reply.get("answer") or "").strip()

        if not text:
            return self.fallback.answer(question, fact_sheet)
        if reply.get("answerable") and not cited:
            return {"answer": NO_FACT_ANSWER, "citations": []}
        return {"answer": text, "citations": cited}
