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


class MuseWonderReportWriter(ReportWriter):
    """Where the real model plugs in. Construction takes whatever client
    the Muse Wonder integration needs; `draft()` would send the fact
    sheet (not raw tracks — see the spec's stage 7 risk note on
    in-context table size) and ask for narrative JSON in the same
    `{"statements": [...]}` shape `StubReportWriter` already produces,
    so swapping this in requires no change to `report_verifier.py` or
    anything downstream of it.
    """

    def __init__(self, client=None):
        self.client = client

    def draft(self, fact_sheet):
        raise NotImplementedError(
            "Muse Wonder integration not wired up yet — no hackathon API key exists. "
            "Use StubReportWriter until it does."
        )
