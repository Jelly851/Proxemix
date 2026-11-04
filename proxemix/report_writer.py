"""Stage 8b — the report drafter. `ReportWriter` is the interface Muse
Wonder's real implementation will satisfy once the hackathon key exists;
`StubReportWriter` satisfies the same interface deterministically (no
LLM call at all) so the verifier and the rest of the pipeline can be
built and proven today.

A draft is `{"statements": [{"text": str, "citations": [fact_id, ...]}]}`.
Nothing downstream trusts a citation until `report_verifier.py` has
checked it against the fact sheet — that is the whole point of the
interface boundary living here.
"""


class ReportWriter:
    def draft(self, fact_sheet):
        """fact_sheet: the list of dicts from fact_sheet.build_fact_sheet().
        Returns {"statements": [{"text": str, "citations": [fact_id, ...]}]}.
        """
        raise NotImplementedError


class StubReportWriter(ReportWriter):
    """Deterministic placeholder narrative: one statement per fact, citing
    that fact's own ID, plus a closing summary statement that cites every
    fact — enough to exercise the verifier's single- and multi-citation
    paths without needing any model call.
    """

    def draft(self, fact_sheet):
        statements = [
            {"text": fact["claim"], "citations": [fact["fact_id"]]}
            for fact in fact_sheet
        ]
        if fact_sheet:
            statements.append({
                "text": f"This clip produced {len(fact_sheet)} citable facts across "
                        f"{len({f['category'] for f in fact_sheet})} categories.",
                "citations": [f["fact_id"] for f in fact_sheet],
            })
        return {"statements": statements}


REPORT_PROMPT = """You are writing the findings section of an analysis report \
about one video clip. Every measurement taken from the clip is listed below, and \
they are the only material you have.

Facts:
{facts}

Write 4 to 6 statements that a researcher or an operations manager would want \
first. Lead with what the clip shows rather than with what was measured. Group \
facts that belong to one finding into one statement and cite all of them. Leave \
out anything that does not change what the reader would do.

Rules:
1. Every statement must rest on the facts above and cite the fact_id of each one \
it uses. A statement citing nothing will be deleted before anyone reads it.
2. Never state a number that is not in a fact you cite.
3. One or two sentences per statement. Plain words. No markdown, no headings.

Reply with JSON only, in this exact shape:
{{"statements": [{{"text": "...", "citations": ["<fact_id>"]}}]}}"""


class MuseWonderReportWriter(ReportWriter):
    """The findings report, drafted by muse-wonder-1.0 over the fact sheet.

    Returns the same `{"statements": [{"text", "citations"}]}` shape
    `StubReportWriter` produces, so `report_verifier.py` and everything
    downstream are untouched. The verifier is load-bearing here rather than
    decorative: it drops any fact_id the model invented, and deletes outright
    any statement left with no real citation.

    `fallback` keeps the report rendering when the model cannot be reached.
    """

    def __init__(self, client=None, fallback=None):
        self.client = client
        self.fallback = fallback or StubReportWriter()

    def draft(self, fact_sheet):
        if not fact_sheet or self.client is None:
            return self.fallback.draft(fact_sheet)

        from proxemix.muse import MuseError, render_fact_sheet

        prompt = REPORT_PROMPT.format(facts=render_fact_sheet(fact_sheet))
        try:
            reply = self.client.complete_json(prompt, note="report",
                                              max_output_tokens=3000)
        except MuseError:
            return self.fallback.draft(fact_sheet)

        statements = []
        for stmt in reply.get("statements") or []:
            text = str(stmt.get("text") or "").strip()
            cites = [str(c) for c in (stmt.get("citations") or [])]
            if text:
                statements.append({"text": text, "citations": cites})
        if not statements:
            return self.fallback.draft(fact_sheet)
        return {"statements": statements}
