"""Tests for rex.

The centrepiece is differential testing: for a corpus of patterns and inputs,
rex must agree with Python's `re` on the match span and on every capture group.
An engine that is fast and subtly wrong is worse than no engine, and hand-written
expectations would only encode whatever I already believed.
"""
import re as stdlib_re
import time

import pytest

import rex
from rex import ParseError, Regex


# Patterns where rex and stdlib re should agree exactly. Deliberately includes
# the awkward cases: empty alternatives, nested groups, lazy quantifiers,
# anchors, bounded repeats, and character-class edge cases.
PATTERNS = [
    r"abc", r"a", r"", r"a|b", r"a|", r"|a",
    r"a*", r"a+", r"a?", r"a*?", r"a+?", r"a??",
    r"(a)", r"(a)(b)", r"((a))", r"(a|b)", r"(a|b)*", r"(ab)+",
    r"(?:ab)+", r"(?:a|b)c",
    r"a{2}", r"a{2,}", r"a{2,4}", r"a{0,2}", r"(ab){2,3}",
    r".", r".*", r".+", r"a.c", r".*b",
    r"[abc]", r"[^abc]", r"[a-z]", r"[a-z0-9]", r"[^a-z]", r"[]a]",
    r"[-a]", r"[a-]", r"\d", r"\D", r"\w", r"\W", r"\s", r"\S",
    r"\d+", r"[\d]+", r"[\w.]+",
    r"^abc", r"abc$", r"^abc$", r"^a|b$",
    r"\bword\b", r"\Bord\B", r"\bcat",
    r"(a+)+b", r"(a|a)*b", r"(a|aa)+c",
    r"colou?r", r"(\w+)@(\w+)\.(\w+)",
    r"a(b|c)*d", r"((a|b)(c|d))+",
    r"\.", r"\*", r"\\", r"a\-b",
    r"x*y*z*",
]

TEXTS = [
    "", "a", "b", "ab", "abc", "abcd", "aaa", "aaab", "aaac",
    "xxabbac", "ababab", "abab", "aabbcc",
    "hello world", "cat", "concat", "the cat sat", "word", "sword", "words",
    "keenan@example.com", "a.b", "a-b", "a*b", "a\\b",
    "123", "abc123def", "  spaced  ", "line1\nline2",
    "aaaaaaaaaab", "aaaaaaaaaa", "acd", "abd", "abcd" * 3,
    "x", "xyz", "zzz",
]


def spans_and_groups(m):
    if m is None:
        return None
    return (m.span(0), tuple(m.span(i) for i in range(1, (m.re.groups if hasattr(m, "re") else m.group_count) + 1)))


def _stdlib_result(pattern, text):
    m = stdlib_re.compile(pattern).search(text)
    if m is None:
        return None
    return (m.span(), tuple(m.span(i) for i in range(1, m.re.groups + 1)))


def _rex_result(pattern, text):
    m = Regex(pattern).search(text)
    if m is None:
        return None
    return ((m.start, m.end), tuple(m.span(i) or (-1, -1) for i in range(1, m.group_count + 1)))


def _normalise(result):
    """stdlib reports an unmatched group as (-1, -1); so does rex."""
    if result is None:
        return None
    span, groups = result
    return span, tuple(g if g is not None else (-1, -1) for g in groups)


# ------------------------------------------------------- differential tests

# Patterns with a quantified, nullable body. Spans agree; capture groups
# legitimately do not. See the divergence note in rex.py.
NULLABLE_LOOP_PATTERNS = [r"(a*)*b", r"(|a)*b", r"(a*)*", r"(a?)*b"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_agrees_with_stdlib_on_every_text(pattern):
    for text in TEXTS:
        try:
            expected = _normalise(_stdlib_result(pattern, text))
        except stdlib_re.error:
            pytest.skip(f"stdlib rejects {pattern!r}")
        actual = _normalise(_rex_result(pattern, text))
        assert actual == expected, (
            f"pattern={pattern!r} text={text!r}: rex={actual} stdlib={expected}"
        )


@pytest.mark.parametrize("pattern", NULLABLE_LOOP_PATTERNS)
def test_spans_agree_even_where_captures_diverge(pattern):
    """The documented divergence is confined to capture groups.

    If a nullable quantified body ever changed the overall match span, that
    would be a real bug rather than a known semantic difference. This is the
    test that would catch it.
    """
    for text in TEXTS:
        expected = _stdlib_result(pattern, text)
        actual = _rex_result(pattern, text)
        if expected is None or actual is None:
            assert (expected is None) == (actual is None), f"{pattern!r} on {text!r}"
            continue
        assert actual[0] == expected[0], (
            f"span mismatch, pattern={pattern!r} text={text!r}: "
            f"rex={actual[0]} stdlib={expected[0]}"
        )


def test_the_divergence_is_what_the_docstring_says_it_is():
    """Pins the exact known difference, so a future change to it is visible."""
    text = "aaab"
    assert stdlib_re.search(r"(a*)*b", text).span(1) == (3, 3)   # empty last iteration
    assert Regex(r"(a*)*b").search(text).span(1) == (0, 3)       # final non-empty one
    # The span is identical, which is the part that matters.
    assert stdlib_re.search(r"(a*)*b", text).span() == (0, 4)
    assert (Regex(r"(a*)*b").search(text).start,
            Regex(r"(a*)*b").search(text).end) == (0, 4)


@pytest.mark.parametrize("pattern", [
    r"\d+", r"[a-z]+", r"a+", r"(ab)+", r"\w+", r"x*", r"a|b", r".",
])
def test_findall_agrees_with_stdlib(pattern):
    for text in TEXTS:
        expected = stdlib_re.compile(pattern).findall(text)
        # stdlib findall returns groups when the pattern has any; compare only
        # group-free patterns as whole matches.
        if stdlib_re.compile(pattern).groups:
            expected = [m.group(0) for m in stdlib_re.compile(pattern).finditer(text)]
        assert Regex(pattern).findall(text) == expected, f"{pattern!r} on {text!r}"


@pytest.mark.parametrize("pattern,text", [
    (r"a", "banana"), (r"\d+", "a1b22c333"), (r"x*", "abc"), (r"", "abc"),
])
def test_finditer_spans_agree(pattern, text):
    expected = [m.span() for m in stdlib_re.finditer(pattern, text)]
    actual = [(m.start, m.end) for m in Regex(pattern).finditer(text)]
    assert actual == expected


# ------------------------------------------------------------- the point

def test_no_catastrophic_backtracking():
    """The reason this engine exists.

    (a+)+b against a run of 'a' with no 'b' is exponential in a backtracking
    engine. Here it must stay comfortably linear.
    """
    regex = Regex(r"(a+)+b")

    def elapsed(n):
        text = "a" * n
        start = time.perf_counter()
        regex.search(text)
        return time.perf_counter() - start

    small = elapsed(100)
    large = elapsed(400)

    # Four times the input should cost far less than exponential growth. A
    # generous ceiling, because timing on shared hardware is noisy.
    assert large < max(small * 40, 0.5), f"{small=:.4f}s {large=:.4f}s"


def test_stdlib_actually_struggles_where_rex_does_not():
    """Confirms the comparison is real rather than a claim about a straw man."""
    text = "a" * 26
    start = time.perf_counter()
    stdlib_re.search(r"(a+)+b", text)
    stdlib_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    Regex(r"(a+)+b").search(text)
    rex_elapsed = time.perf_counter() - start

    assert stdlib_elapsed > rex_elapsed


def test_nested_star_terminates():
    # (a*)* can loop forever without the per-position visited set.
    assert Regex(r"(a*)*b").search("aaab") is not None
    assert Regex(r"(a*)*b").search("aaa") is None


def test_empty_body_repeat_terminates():
    assert Regex(r"(|a)*b").search("b") is not None


# ------------------------------------------------------------- parse errors

@pytest.mark.parametrize("pattern", [
    r"(", r")", r"[", r"a**", r"a+*", r"a*??", r"a{2}{3}",
    r"*a", r"+a", r"?a", r"a{3,1}", r"(?P<n>a)",
])
def test_bad_patterns_rejected(pattern):
    with pytest.raises(ParseError):
        Regex(pattern)


def test_parse_error_reports_a_position():
    with pytest.raises(ParseError, match=r"\d"):
        Regex(r"(abc")


def test_lone_brace_is_a_literal():
    # Python treats an unparseable {  as a literal; so does this.
    assert Regex(r"a{b").search("a{b") is not None


# ----------------------------------------------------------------- matching

def test_match_is_anchored_at_zero():
    assert Regex(r"abc").match("abcdef") is not None
    assert Regex(r"bcd").match("abcdef") is None
    assert Regex(r"bcd").search("abcdef") is not None


def test_fullmatch_requires_the_whole_string():
    assert Regex(r"a+").fullmatch("aaa") is not None
    assert Regex(r"a+").fullmatch("aaab") is None


def test_search_from_an_offset():
    assert Regex(r"a").search("aaa", start=2).start == 2


def test_group_accessors():
    m = Regex(r"(\w+)@(\w+)").search("mail keenan@example here")
    assert m.group() == "keenan@example"
    assert m.group(1) == "keenan"
    assert m.group(2) == "example"
    assert m.groups() == ("keenan", "example")
    assert m.span(1) == (5, 11)


def test_unmatched_group_is_none():
    m = Regex(r"(a)|(b)").search("b")
    assert m.group(1) is None
    assert m.group(2) == "b"


def test_last_iteration_wins_for_repeated_groups():
    # Matches Python: a repeated group reports its final iteration.
    assert Regex(r"(a|b)+").search("abab").group(1) == stdlib_re.search(r"(a|b)+", "abab").group(1)


def test_sub_replaces_every_match():
    assert Regex(r"\d+").sub("#", "a1b22c333") == "a#b#c#"


def test_sub_with_no_matches_returns_the_original():
    assert Regex(r"z+").sub("#", "abc") == "abc"


# ------------------------------------------------------------- greediness

def test_greedy_takes_the_most():
    assert Regex(r"a.*b").search("axxbxxb").group() == "axxbxxb"


def test_lazy_takes_the_least():
    assert Regex(r"a.*?b").search("axxbxxb").group() == "axxb"


def test_lazy_bounded_repeat():
    assert Regex(r"a{2,4}?").search("aaaa").group() == "aa"


def test_alternation_prefers_the_left():
    # Not longest-match: the first alternative that can match wins.
    assert Regex(r"a|ab").search("ab").group() == "a"
    assert Regex(r"a|ab").search("ab").group() == stdlib_re.search(r"a|ab", "ab").group()


# ------------------------------------------------------------- compilation

def test_program_is_a_flat_instruction_list():
    program = Regex(r"a").program
    assert program[0].op is rex.Op.SAVE
    assert program[-1].op is rex.Op.MATCH


def test_bounded_repeat_expands_by_duplication():
    short = len(Regex(r"a{2}").program)
    long = len(Regex(r"a{8}").program)
    assert long > short


def test_dump_lists_every_instruction():
    regex = Regex(r"(a|b)*c")
    assert len(regex.dump().splitlines()) == len(regex.program)


def test_group_count_matches_capturing_groups_only():
    assert Regex(r"(a)(?:b)(c)").group_count == 2


def test_module_level_helpers():
    assert rex.search(r"\d+", "abc123").group() == "123"
    assert rex.compile(r"a+").search("aaa") is not None
