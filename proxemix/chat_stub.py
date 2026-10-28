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


class MuseWonderChatBackend(ChatBackend):
    """Where the real Muse Wonder client plugs in - same answer()
    interface, same fact-sheet-only context (never raw tracks, per the
    spec's stage 7 risk note), currently unimplemented.
    """

    def __init__(self, client=None):
        self.client = client

    def answer(self, question, fact_sheet):
        raise NotImplementedError(
            "Muse Wonder chat integration not wired up yet — no hackathon API key exists. "
            "Use StubChatBackend until it does."
        )
