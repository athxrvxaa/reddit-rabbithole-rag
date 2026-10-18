"""Gradio app. Run locally: `python app.py`. Deployed as a HF Space (see deploy/)."""
from __future__ import annotations

import html
import os
import time
from collections import OrderedDict, defaultdict, deque

import gradio as gr

from rabbithole.config import get_settings
from rabbithole.generation import GeminiError, check_citations
from rabbithole.text import short

SETTINGS = get_settings()
PIPELINE = None
LOAD_ERROR = None
try:
    from rabbithole.pipeline import build_pipeline

    PIPELINE = build_pipeline(SETTINGS)
except Exception as e:  # show the problem in the UI instead of crashing the Space
    LOAD_ERROR = f"{type(e).__name__}: {e}"

N_DOCS = PIPELINE.retriever.n if PIPELINE else 0
N_THREADS = PIPELINE.retriever.manifest.get("n_threads", 0) if PIPELINE else 0
CATEGORIES = ["all"] + (PIPELINE.retriever.categories if PIPELINE else [])

# ---- tiny abuse protection (your Gemini quota is the cost) --------------------------------
RATE_LIMIT = int(os.environ.get("RR_RATE_LIMIT", "12"))  # requests per window
RATE_WINDOW = int(os.environ.get("RR_RATE_WINDOW", "600"))  # seconds
_hits: dict[str, deque] = defaultdict(deque)
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()


def _client_id(req: gr.Request | None) -> str:
    if req is None:
        return "local"
    fwd = req.headers.get("x-forwarded-for") if req.headers else None
    return (fwd.split(",")[0].strip() if fwd else getattr(req.client, "host", "unknown")) or "unknown"


def _rate_ok(cid: str) -> bool:
    now = time.time()
    q = _hits[cid]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


def sources_html(hits) -> str:
    e = lambda t: html.escape(t or "")  # noqa: E731  (Reddit text is untrusted -> always escape)
    cards = []
    for i, h in enumerate(hits, 1):
        d = h.doc
        url = d["source_url"] if str(d.get("source_url", "")).startswith("https://") else "#"  # no javascript: links
        rr = f" · rerank {h.rerank:.2f}" if h.rerank is not None else ""
        cards.append(
            f"""<div style="border:1px solid #8884;border-radius:10px;padding:12px;margin-bottom:10px;">
  <div style="font-size:12px;opacity:.7;">[{i}] {e(d['category'])}{rr} · score {d['answer_score']}</div>
  <div style="font-weight:600;margin:4px 0;">🎙️ {e(short(d['title'], 110))}</div>
  <div style="font-size:13px;"><b>Asked:</b> {e(short(d['question'], 220))}</div>
  <div style="font-size:13px;margin-top:4px;"><b>Host:</b> {e(short(d['answer'], 380))}</div>
  <div style="margin-top:6px;font-size:12px;"><a href="{e(url)}" target="_blank" rel="noopener noreferrer">🔗 view on Reddit</a></div>
</div>"""
        )
    return "".join(cards) or "<i>No sources.</i>"


def respond(query: str, top_k: int, category: str, request: gr.Request):
    query = (query or "").strip()
    if LOAD_ERROR:
        yield f"⚠️ App failed to load: `{LOAD_ERROR}`", "", ""
        return
    if not query:
        yield "Please enter a question.", "", ""
        return
    if len(query) > 400:
        yield "Please keep the question under 400 characters.", "", ""
        return
    if not _rate_ok(_client_id(request)):
        yield "Too many requests — please wait a few minutes and try again.", "", ""
        return
    key = (query.lower(), int(top_k), category)
    if key in _cache:
        _cache.move_to_end(key)
        yield _cache[key]
        return

    cat = None if category == "all" else category
    search = PIPELINE.retrieve(query, top_k=int(top_k), category=cat)
    src = sources_html(search.hits)
    status = f"retrieved in {search.timings['total_ms']:.0f} ms"
    if search.confidence == "low":
        status += " · low confidence"
    yield "_Thinking…_", src, status
    try:
        text = ""
        for text in PIPELINE.stream_answer(query, search):
            yield text, src, status
        rep = check_citations(text, len(search.hits))
        notes = []
        if rep.invalid:
            notes.append(f"⚠️ cited non-existent source(s) {rep.invalid}")
        if PIPELINE.llm and PIPELINE.llm.last_finish_reason == "MAX_TOKENS":
            notes.append("⚠️ answer may be cut off")
        if search.abstain:
            notes.append("no sufficiently relevant AMA found")
        final = (text, src, status + (" · " + " · ".join(notes) if notes else ""))
        _cache[key] = final
        if len(_cache) > 256:
            _cache.popitem(last=False)
        yield final
    except (GeminiError, RuntimeError) as err:
        yield f"⚠️ Generation failed: {err}\n\nThe retrieved sources are still shown on the right.", src, status


EXAMPLES = [
    ["What do people in prison actually do all day?", 4, "all"],
    ["What is the hardest part of being a nurse?", 4, "all"],
    ["How do you recover from addiction?", 4, "all"],
    ["What was it like growing up in a cult?", 4, "all"],
    ["What do soldiers miss most while deployed?", 4, "military"],
]

with gr.Blocks(title="Reddit RabbitHole RAG") as demo:
    gr.Markdown(
        f"""# 🕳️ Reddit RabbitHole RAG
Ask a question — get an answer **cited from real Reddit AMAs**.
*{N_DOCS:,} Q&A pairs from {N_THREADS:,} AMA threads · hybrid BM25 + dense retrieval · cross-encoder rerank · cited generation*

> These are personal anecdotes from individual Redditors, not verified facts or professional advice."""
    )
    with gr.Row():
        with gr.Column(scale=3):
            q = gr.Textbox(label="Your question", placeholder="e.g. What is it actually like to work night shifts as a nurse?", lines=2)
        with gr.Column(scale=1):
            k = gr.Slider(1, 7, value=4, step=1, label="AMA sources")
            cat = gr.Dropdown(CATEGORIES, value="all", label="Topic filter")
            btn = gr.Button("Ask the rabbit hole 🕳️", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=3):
            ans = gr.Markdown(label="Answer")
        with gr.Column(scale=2):
            srcs = gr.HTML(label="Sources")
    gr.Examples(EXAMPLES, inputs=[q, k, cat], label="Try these")
    btn.click(respond, [q, k, cat], [ans, srcs, status])
    q.submit(respond, [q, k, cat], [ans, srcs, status])

demo.queue(max_size=20, default_concurrency_limit=4)

if __name__ == "__main__":
    demo.launch()
