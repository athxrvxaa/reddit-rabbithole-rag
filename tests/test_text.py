from rabbithole.text import bare_id, clean_comment, content_hash, is_question, truncate_sentences


def test_unescape_and_markdown():
    t = "&gt; quoted &amp;amp; thing\n\nSee [this link](http://x.com/a) and http://spam.com now **bold** ^super"
    out = clean_comment(t)
    assert "&gt;" not in out and "&amp;" not in out
    assert "http" not in out and "this link" in out
    assert "**" not in out and "^" not in out


def test_quote_modes():
    raw = "&gt; Do many people work?\n\nYes, most do. They work in the kitchen and laundry every day of the week."
    assert "Do many people" not in clean_comment(raw, quotes="drop")
    kept = clean_comment(raw, quotes="marker")
    assert kept.startswith("Do many people") and ">" not in kept


def test_caret_in_math_survives():
    assert "2^10" in clean_comment("that is 2^10 bytes")


def test_is_question():
    assert is_question("What is it like?")
    assert is_question("tell me about your day")
    assert is_question("Do you miss it")
    assert not is_question("Thanks for doing this AMA")
    assert not is_question("Dont worry about it")  # 'do' must not match 'dont'


def test_truncate_sentences():
    s = "First sentence here. Second sentence is long enough to count. Third part keeps going on and on"
    out = truncate_sentences(s, 62)
    assert out.endswith("count.") and len(out) <= 62
    assert truncate_sentences("word " * 40, 50).endswith("…")
    assert truncate_sentences("short", 50) == "short"


def test_bare_id_and_hash():
    assert bare_id("t3_abc") == "abc" and bare_id("t1_x1") == "x1" and bare_id("abc") == "abc"
    assert content_hash("What?", "Answer here") == content_hash("what", "ANSWER   here!")
