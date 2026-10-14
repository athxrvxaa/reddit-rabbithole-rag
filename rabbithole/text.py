"""Text cleaning + small NLP helpers (no heavy dependencies)."""
from __future__ import annotations

import hashlib
import html
import re

_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*[^)\s]*\s*\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_QUOTE_MARK = re.compile(r"^[ \t]*>+[ \t]?", re.M)
_QUOTE_LINE = re.compile(r"^[ \t]*>.*$", re.M)
_EMPHASIS = re.compile(r"\*{1,3}|~~")
_SUPERSCRIPT = re.compile(r"(?<!\w)\^+")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SP = re.compile(r"[ \t]{2,}")

_QUESTION_START = re.compile(
    r"^(what|how|why|when|where|who|whom|whose|which|did|do|does|are|is|were|was|have|has|had|"
    r"can|could|would|will|should|shall|may|tell|describe|explain|any|is there|are there|"
    r"i'?m curious|i was wondering|curious)\b",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _unescape(t: str) -> str:
    # Reddit dumps are sometimes double-escaped (&amp;gt;)
    for _ in range(3):
        n = html.unescape(t)
        if n == t:
            break
        t = n
    return t


def clean_comment(text: str, quotes: str = "marker") -> str:
    """Clean a Reddit comment body.

    quotes:
      "marker" -> keep quoted text, strip the leading '>' marker (use for questions)
      "drop"   -> remove quoted lines entirely (use for answers that re-quote the question)
      "keep"   -> leave '>' as-is
    """
    if not text:
        return ""
    t = _unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    if quotes == "drop":
        t = _QUOTE_LINE.sub("", t)
    elif quotes == "marker":
        t = _QUOTE_MARK.sub("", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _URL.sub("", t)
    t = _HEADING.sub("", t)
    t = _EMPHASIS.sub("", t)
    t = _SUPERSCRIPT.sub("", t)
    t = _ZERO_WIDTH.sub("", t)
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = _MULTI_SP.sub(" ", t)
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


def is_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(_QUESTION_START.match(t.lower()))


def truncate_sentences(text: str, max_chars: int) -> str:
    """Cut at a sentence boundary (or word boundary) instead of mid-word."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    best = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind(".\n"))
    if best >= max_chars * 0.5:
        return cut[: best + 1].strip()
    sp = cut.rfind(" ")
    if sp >= max_chars * 0.5:
        cut = cut[:sp]
    return cut.rstrip(",;:- ") + "…"


def normalize_for_hash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def content_hash(question: str, answer: str) -> str:
    key = normalize_for_hash(question)[:400] + "||" + normalize_for_hash(answer)[:600]
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def bare_id(x: str) -> str:
    """'t3_abc' / 't1_abc' -> 'abc'."""
    x = (x or "").strip()
    return x[3:] if len(x) > 3 and x[0] == "t" and x[1].isdigit() and x[2] == "_" else x


def short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
