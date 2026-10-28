"""Stage 8c — the mechanical verifier. This is the first of the spec's
three load-bearing wow mechanisms: the LLM (or its stub) never invents a
frame number, because nothing it cites reaches the reader until this
~50-line check has confirmed the fact ID exists in the fact sheet. Any
citation that doesn't exist is dropped, not fixed or guessed at.
"""


def verify_draft(draft, fact_sheet):
    """Returns (clean_statements, dropped_citations):
    clean_statements — draft['statements'] with every invalid fact_id
      removed from each statement's citation list. A statement left with
      no valid citation is removed outright: an uncited claim is exactly
      what this mechanism exists to keep out of the report, so dropping
      the citation but shipping the sentence would defeat it.
    dropped_citations — a flat list of (statement_index, fact_id) for
      every citation that did not exist in the fact sheet.
    """
    valid_ids = {f["fact_id"] for f in fact_sheet}
    clean_statements = []
    dropped = []
    for i, stmt in enumerate(draft.get("statements", [])):
        kept, bad = [], []
        for cid in stmt.get("citations", []):
            (kept if cid in valid_ids else bad).append(cid)
        for cid in bad:
            dropped.append((i, cid))
        if kept:
            clean_statements.append({"text": stmt["text"], "citations": kept})
    return clean_statements, dropped


def render_markdown(clean_statements, fact_sheet, title="Proxemix Report"):
    """Renders verified statements to markdown, with every surviving
    citation replaced by its frame/timecode link, e.g.
    `[frame 4,210-4,540 / 02:48-03:02]`.
    """
    by_id = {f["fact_id"]: f for f in fact_sheet}
    lines = [f"# {title}", ""]
    for stmt in clean_statements:
        links = " ".join(by_id[cid]["citation"] for cid in stmt["citations"] if cid in by_id)
        line = stmt["text"]
        if links:
            line = f"{line} {links}"
        lines.append(f"- {line}")
    return "\n".join(lines) + "\n"


def verify_and_render(draft, fact_sheet, title="Proxemix Report"):
    clean_statements, dropped = verify_draft(draft, fact_sheet)
    markdown = render_markdown(clean_statements, fact_sheet, title=title)
    return clean_statements, dropped, markdown
