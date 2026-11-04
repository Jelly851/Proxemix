"""Muse Wonder client — the one text-model seam for the whole product.

Three places in Proxemix were built with a model-shaped hole in them, and
all three are filled from here:

* `chat_stub.MuseWonderChatBackend` — the question box over a clip's facts.
* `report_writer.MuseReportWriter` — the drafted findings report.
* `promptable.plan_concepts` — a typed question becomes the concepts SAM hunts.

Call shape is the one `sam_video.py` proved live on 2026-10-27, with the
model swapped: POST `/v1/responses`, one user message of `input_text`,
`stream: true` with `Accept: text/event-stream`. The SSE accumulator is
reimplemented here rather than imported so this module pulls in no numpy
or pandas — the server imports it on every request.

Two deliberate differences from the SAM client:

* **The budget is counted in calls, not dollars.** SAM's per-frame prices
  are published; token prices for muse-wonder-1.0 are not in hand, so a
  dollar cap here would be a number I made up. A call ceiling is honest
  and still stops a loop.
* **Nothing reaches the model but the fact sheet.** No raw tracks, no
  frame images. The facts are small, already computed, and already carry
  the frame ranges every citation needs.

`transport(url, headers, body_bytes, timeout) -> (status, text)` is the
only seam to the network, so every test below runs with no key and no
HTTP. The key is read fresh from disk per call, never stored on the
instance, never logged, and scrubbed from any error text.
"""
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

API_URL = "https://api.meta.ai/v1/responses"
MODEL = "muse-wonder-1.0"
DEFAULT_KEY_FILE = pathlib.Path(
    os.environ.get("META_API_KEY_FILE", "C:/Users/jayde/meta-api-key.txt"))

DEFAULT_MAX_CALLS = 40
DEFAULT_TIMEOUT = 120

# muse-wonder-1.0 reasons before it answers, and the reasoning is billed and
# capped as output. Measured live 2026-11-03: a prompt whose visible answer was
# the 12 characters `{"ok": true}` spent 123 output tokens, 109 of them
# reasoning. So a tight `max_output_tokens` does not shorten the answer — it
# truncates the stream before any answer exists and returns nothing at all.
# Every ceiling below is sized for reasoning plus answer, not for the answer.
MIN_OUTPUT_TOKENS = 600

# And the default effort is `max`, which on the 14-fact report prompt spent
# **2,998 of 3,000** output tokens thinking and returned `response.incomplete`
# with no answer at all — measured 2026-11-03. `medium` answered the same
# prompt in ~1,150 characters and grouped facts into findings better than
# `low` did, so it is the default here. Raise it per call where a job is
# genuinely harder than writing five sentences.
DEFAULT_REASONING_EFFORT = "medium"


class MuseError(RuntimeError):
    pass


class CallCapExceeded(MuseError):
    pass


def _urllib_transport(url, headers, body, timeout):
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


class MuseClient:
    def __init__(self, key_path=DEFAULT_KEY_FILE, url=API_URL, model=MODEL,
                 transport=None, timeout=DEFAULT_TIMEOUT,
                 max_calls=DEFAULT_MAX_CALLS,
                 reasoning_effort=DEFAULT_REASONING_EFFORT):
        self.key_path = pathlib.Path(key_path)
        self.url = url
        self.model = model
        self.transport = transport or _urllib_transport
        self.timeout = timeout
        self.max_calls = max_calls
        self.reasoning_effort = reasoning_effort
        self.calls = []            # one dict per call: note, chars_in, chars_out

    # -- key ----------------------------------------------------------------
    def key_present(self):
        """True when the file holds something that could be a key. Never
        prints it, never returns it — length only."""
        try:
            raw = "".join(self.key_path.read_text(encoding="utf-8",
                                                  errors="replace").split())
        except OSError:
            return False
        return len(raw) >= 20

    def _key(self):
        if not self.key_path.exists():
            raise MuseError(f"no key file at {self.key_path}")
        key = "".join(self.key_path.read_text(encoding="utf-8",
                                              errors="replace").split())
        if len(key) < 20:
            raise MuseError(
                f"key file {self.key_path} holds {len(key)} characters — empty "
                "or truncated, not a key")
        return key

    # -- the stream ---------------------------------------------------------
    @staticmethod
    def _output_text(stream_text):
        """Accumulate `response.output_text.delta` out of the SSE stream.

        Falls back to the terminal `response.completed` event when no deltas
        arrived, and to the raw body when the body is not SSE at all — so a
        transport handing back plain JSON still parses.
        """
        if "data:" not in stream_text:
            return stream_text
        out, saw_completed, saw_delta = [], False, False
        for line in stream_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            etype = event.get("type")
            if etype == "response.output_text.delta":
                delta = event.get("delta")
                if delta:
                    out.append(delta)
                    saw_delta = True
            elif etype == "response.completed":
                saw_completed = True
                if not saw_delta:
                    resp = event.get("response") or {}
                    for item in resp.get("output", []) or []:
                        for c in item.get("content", []) or []:
                            if c.get("text"):
                                out.append(c["text"])
            elif etype == "response.incomplete":
                # The budget ran out before an answer existed. Named rather
                # than returned as raw SSE, because the raw body parses as
                # "no JSON object" and that error points at the wrong cause.
                usage = ((event.get("response") or {}).get("usage") or {})
                detail = (usage.get("output_tokens_details") or {})
                raise MuseError(
                    "the reply was cut off before any answer: "
                    f"{usage.get('output_tokens')} output tokens, "
                    f"{detail.get('reasoning_tokens')} of them reasoning. "
                    "Raise max_output_tokens or lower reasoning_effort.")
            elif etype == "error" or ("error" in event and event.get("error")):
                raise MuseError(
                    f"error event mid-stream: {json.dumps(event)[:600]}")
        if not out and not saw_completed:
            return stream_text
        return "".join(out)

    # -- requests -----------------------------------------------------------
    def complete(self, prompt, note="", max_output_tokens=None):
        """One prompt in, the model's text out. Raises MuseError on anything
        that is not a clean 200."""
        if len(self.calls) >= self.max_calls:
            raise CallCapExceeded(
                f"this client has already made {len(self.calls)} calls against "
                f"a {self.max_calls}-call ceiling — refused before sending. "
                "Raise max_calls deliberately if that is wrong.")
        body = {
            "model": self.model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }],
            "stream": True,
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if max_output_tokens:
            body["max_output_tokens"] = max(int(max_output_tokens),
                                            MIN_OUTPUT_TOKENS)

        key = self._key()
        headers = {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = json.dumps(body).encode()
        try:
            status, text = self.transport(self.url, headers, payload,
                                          self.timeout)
        except Exception as e:                      # network, DNS, timeout
            raise MuseError(f"{note or 'completion'}: {type(e).__name__}: "
                            f"{str(e).replace(key, '<key>')}") from None
        text = text.replace(key, "<key>")
        if status != 200:
            raise MuseError(f"{note or 'completion'}: HTTP {status} — "
                            f"{text[:600]}")
        out = self._output_text(text)
        self.calls.append({"note": note, "chars_in": len(prompt),
                           "chars_out": len(out)})
        return out

    def complete_json(self, prompt, note="", max_output_tokens=None):
        """`complete`, then parse the reply as JSON.

        Models fence JSON in markdown and prepend throat-clearing even when
        told not to, so the fence is stripped and the outermost brace-balanced
        object is taken. A reply with no parseable object raises rather than
        returning a half-answer, because every caller here has a working
        non-model fallback and a silent empty result is worse than a raised one.
        """
        raw = self.complete(prompt, note=note,
                            max_output_tokens=max_output_tokens)
        return parse_json_object(raw)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(raw):
    """The first brace-balanced JSON object in `raw`, as a dict."""
    text = _FENCE.sub("", str(raw).strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass

    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise MuseError(f"no JSON object in the reply: {str(raw)[:300]!r}")


# --- the shared fact-sheet rendering ----------------------------------------
# All three seams hand the model the same view of a clip, so it is written
# once. Each line is one fact with the ID a citation must use, its frame
# range, and the claim already phrased in the product's own words.

def render_fact_sheet(fact_sheet, max_facts=60):
    lines = []
    for fact in list(fact_sheet)[:max_facts]:
        metrics = fact.get("metrics") or {}
        bits = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:6])
        lines.append(
            f"- {fact['fact_id']} [{fact.get('category', '?')}] "
            f"frames {fact.get('frame_start')}-{fact.get('frame_end')}: "
            f"{fact.get('claim', '').strip()}"
            + (f" ({bits})" if bits else ""))
    return "\n".join(lines) if lines else "(no facts computed for this clip)"


if __name__ == "__main__":        # a no-network smoke test of the parsing
    assert parse_json_object('{"a": 1}')["a"] == 1
    assert parse_json_object('```json\n{"a": 2}\n```')["a"] == 2
    assert parse_json_object('Sure! {"a": {"b": "}"}} trailing')["a"]["b"] == "}"
    sse = ('data: {"type":"response.output_text.delta","delta":"{\\"a\\": "}\n'
           'data: {"type":"response.output_text.delta","delta":"3}"}\n'
           'data: [DONE]\n')
    assert parse_json_object(MuseClient._output_text(sse))["a"] == 3
    print("muse.py parsing: 4/4 pass")
