#!/usr/bin/env python3
"""A regular expression engine that cannot blow up.

Python's `re`, like Perl's and JavaScript's, is a backtracking engine. On
patterns with nested quantifiers -- `(a+)+b` is the classic -- it explores
exponentially many paths and hangs. That is not a bug in the pattern; it is a
property of the algorithm, and it is a real denial-of-service vector wherever
user input reaches a regex.

This is the other approach: compile to a small instruction set and simulate all
branches simultaneously, one input character at a time. Every position is
visited once per instruction, so matching is O(len(pattern) * len(text)) with no
worst case to discover in production.

The simulation is a Pike VM, which keeps capture groups and leftmost-first
semantics -- the same answers a backtracking engine gives, without the
backtracking.

ONE DOCUMENTED DIVERGENCE
    For a quantified group whose body can match the empty string -- `(a*)*`,
    `(|a)*` -- the *match span* agrees with Python, but the *capture group*
    may not. On `(a*)*b` against "aaab", Python reports group 1 as the empty
    match at position 3; this reports "aaa" at 0-3.

    Python gets its answer by running one extra, empty iteration of the loop
    and then detecting that it made no progress. This engine cannot: the
    per-position visited set that guarantees linear time is exactly what stops
    an instruction being re-entered at the same position, and that guarantee
    is the entire point.

    RE2 and Go's regexp diverge here in the same way and for the same reason.
    The differential tests assert spans always agree and mark this case
    explicitly rather than skipping it.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto


class Op(Enum):
    CHAR = auto()     # consume one specific character
    CLASS = auto()    # consume one character from a set
    ANY = auto()      # consume any character (not newline unless DOTALL)
    MATCH = auto()    # success
    JMP = auto()      # unconditional jump
    SPLIT = auto()    # try x first, then y -- the only source of branching
    SAVE = auto()     # record the input position in a capture slot
    ASSERT = auto()   # zero-width test: ^ $ \b \B


class Assertion(Enum):
    LINE_START = "^"
    LINE_END = "$"
    WORD_BOUNDARY = r"\b"
    NOT_WORD_BOUNDARY = r"\B"


@dataclass
class Inst:
    op: Op
    char: str = ""
    ranges: tuple[tuple[str, str], ...] = ()
    negated: bool = False
    x: int = 0
    y: int = 0
    slot: int = 0
    assertion: Assertion | None = None

    def matches_char(self, c: str) -> bool:
        if self.op is Op.CHAR:
            return c == self.char
        if self.op is Op.ANY:
            return c != "\n"
        inside = any(lo <= c <= hi for lo, hi in self.ranges)
        return inside != self.negated


# --------------------------------------------------------------------- AST

@dataclass
class Node:
    pass


@dataclass
class Empty(Node):
    pass


@dataclass
class Char(Node):
    char: str


@dataclass
class CharClass(Node):
    ranges: tuple[tuple[str, str], ...]
    negated: bool = False


@dataclass
class Any(Node):
    pass


@dataclass
class Anchor(Node):
    assertion: Assertion


@dataclass
class Concat(Node):
    parts: list[Node]


@dataclass
class Alternate(Node):
    options: list[Node]


@dataclass
class Repeat(Node):
    node: Node
    low: int
    high: int | None      # None means unbounded
    greedy: bool = True


@dataclass
class Group(Node):
    node: Node
    index: int | None     # None for a non-capturing group


class ParseError(ValueError):
    """Raised for a malformed pattern, with the offset where it went wrong."""


# ------------------------------------------------------------------ parser

WORD_RANGES = (("a", "z"), ("A", "Z"), ("0", "9"), ("_", "_"))
DIGIT_RANGES = (("0", "9"),)
SPACE_RANGES = ((" ", " "), ("\t", "\t"), ("\n", "\n"), ("\r", "\r"),
                ("\f", "\f"), ("\v", "\v"))

ESCAPES = {
    "d": (DIGIT_RANGES, False),
    "D": (DIGIT_RANGES, True),
    "w": (WORD_RANGES, False),
    "W": (WORD_RANGES, True),
    "s": (SPACE_RANGES, False),
    "S": (SPACE_RANGES, True),
}

LITERAL_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "0": "\0"}


class Parser:
    """Recursive descent over the pattern.

    Grammar, loosest binding first:
        alternate := concat ('|' concat)*
        concat    := repeat*
        repeat    := atom quantifier*
        atom      := group | class | '.' | anchor | escape | literal
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0
        self.group_count = 0

    # ---- helpers
    def peek(self) -> str:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else ""

    def eat(self) -> str:
        if self.pos >= len(self.pattern):
            raise ParseError(f"unexpected end of pattern at {self.pos}")
        self.pos += 1
        return self.pattern[self.pos - 1]

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise ParseError(f"expected {char!r} at position {self.pos}")
        self.pos += 1

    # ---- grammar
    def parse(self) -> Node:
        node = self.parse_alternate()
        if self.pos < len(self.pattern):
            raise ParseError(f"unbalanced {self.peek()!r} at position {self.pos}")
        return node

    def parse_alternate(self) -> Node:
        options = [self.parse_concat()]
        while self.peek() == "|":
            self.eat()
            options.append(self.parse_concat())
        return options[0] if len(options) == 1 else Alternate(options)

    def parse_concat(self) -> Node:
        parts: list[Node] = []
        while self.peek() and self.peek() not in "|)":
            parts.append(self.parse_repeat())
        if not parts:
            return Empty()
        return parts[0] if len(parts) == 1 else Concat(parts)

    def parse_repeat(self) -> Node:
        node = self.parse_atom()
        while self.peek() and self.peek() in "*+?{":
            if self.peek() == "{":
                bounds = self._try_bounds()
                if bounds is None:
                    break          # a bare '{' is a literal, as Python treats it
                low, high = bounds
            else:
                quantifier = self.eat()
                low, high = {"*": (0, None), "+": (1, None), "?": (0, 1)}[quantifier]

            greedy = True
            if self.peek() == "?":
                self.eat()
                greedy = False

            if isinstance(node, Anchor):
                raise ParseError(f"nothing to repeat at position {self.pos}")
            node = Repeat(node, low, high, greedy)

            # `a**` is meaningless and Python rejects it. Possessive
            # quantifiers (`a*+`) are also rejected here rather than silently
            # treated as something else -- they are not implemented.
            if self.peek() and self.peek() in "*+?":
                raise ParseError(f"multiple repeat at position {self.pos}")
            if self.peek() == "{" and self._try_bounds() is not None:
                raise ParseError(f"multiple repeat at position {self.pos}")
        return node

    def _try_bounds(self) -> tuple[int, int | None] | None:
        start = self.pos
        self.eat()                                   # '{'
        digits_low = ""
        while self.peek().isdigit():
            digits_low += self.eat()
        if not digits_low and self.peek() != ",":
            self.pos = start
            return None

        if self.peek() == "}":
            self.eat()
            n = int(digits_low)
            return n, n

        if self.peek() != ",":
            self.pos = start
            return None
        self.eat()

        digits_high = ""
        while self.peek().isdigit():
            digits_high += self.eat()
        if self.peek() != "}":
            self.pos = start
            return None
        self.eat()

        low = int(digits_low) if digits_low else 0
        high = int(digits_high) if digits_high else None
        if high is not None and high < low:
            raise ParseError(f"min repeat greater than max at position {start}")
        return low, high

    def parse_atom(self) -> Node:
        char = self.peek()

        if char == "(":
            return self.parse_group()
        if char == "[":
            return self.parse_class()
        if char == ".":
            self.eat()
            return Any()
        if char == "^":
            self.eat()
            return Anchor(Assertion.LINE_START)
        if char == "$":
            self.eat()
            return Anchor(Assertion.LINE_END)
        if char == "\\":
            return self.parse_escape()
        if char and char in "*+?":
            raise ParseError(f"nothing to repeat at position {self.pos}")
        if char == ")":
            raise ParseError(f"unbalanced ')' at position {self.pos}")

        return Char(self.eat())

    def parse_group(self) -> Node:
        self.expect("(")
        capturing = True
        if self.peek() == "?":
            self.eat()
            if self.peek() == ":":
                self.eat()
                capturing = False
            else:
                raise ParseError(f"unsupported group syntax at position {self.pos}")

        index = None
        if capturing:
            self.group_count += 1
            index = self.group_count

        node = self.parse_alternate()
        if self.peek() != ")":
            raise ParseError(f"missing ')' for group opened before {self.pos}")
        self.eat()
        return Group(node, index)

    def parse_escape(self) -> Node:
        self.expect("\\")
        char = self.eat()
        if char in ESCAPES:
            ranges, negated = ESCAPES[char]
            return CharClass(ranges, negated)
        if char == "b":
            return Anchor(Assertion.WORD_BOUNDARY)
        if char == "B":
            return Anchor(Assertion.NOT_WORD_BOUNDARY)
        if char in LITERAL_ESCAPES:
            return Char(LITERAL_ESCAPES[char])
        return Char(char)

    def parse_class(self) -> Node:
        self.expect("[")
        negated = False
        if self.peek() == "^":
            self.eat()
            negated = True

        ranges: list[tuple[str, str]] = []
        first = True
        while True:
            if not self.peek():
                raise ParseError("unterminated character class")
            if self.peek() == "]" and not first:
                self.eat()
                break
            first = False

            if self.peek() == "\\":
                self.eat()
                esc = self.eat()
                if esc in ESCAPES:
                    sub_ranges, sub_negated = ESCAPES[esc]
                    if sub_negated:
                        raise ParseError(f"negated class escape \\{esc} inside [] "
                                         f"is not supported")
                    ranges.extend(sub_ranges)
                    continue
                lo = LITERAL_ESCAPES.get(esc, esc)
            else:
                lo = self.eat()

            if self.peek() == "-" and self.pos + 1 < len(self.pattern) \
                    and self.pattern[self.pos + 1] != "]":
                self.eat()
                hi = self.eat()
                if hi == "\\":
                    hi = self.eat()
                    hi = LITERAL_ESCAPES.get(hi, hi)
                if hi < lo:
                    raise ParseError(f"bad character range {lo}-{hi}")
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))

        return CharClass(tuple(ranges), negated)


# --------------------------------------------------------------- compiler

class Compiler:
    """Thompson construction, emitted as a flat instruction list."""

    def __init__(self) -> None:
        self.program: list[Inst] = []

    def emit(self, inst: Inst) -> int:
        self.program.append(inst)
        return len(self.program) - 1

    def compile(self, node: Node, group_count: int) -> list[Inst]:
        self.emit(Inst(Op.SAVE, slot=0))          # whole-match start
        self._compile(node)
        self.emit(Inst(Op.SAVE, slot=1))          # whole-match end
        self.emit(Inst(Op.MATCH))
        return self.program

    def _compile(self, node: Node) -> None:
        if isinstance(node, Empty):
            return

        if isinstance(node, Char):
            self.emit(Inst(Op.CHAR, char=node.char))

        elif isinstance(node, Any):
            self.emit(Inst(Op.ANY))

        elif isinstance(node, CharClass):
            self.emit(Inst(Op.CLASS, ranges=node.ranges, negated=node.negated))

        elif isinstance(node, Anchor):
            self.emit(Inst(Op.ASSERT, assertion=node.assertion))

        elif isinstance(node, Concat):
            for part in node.parts:
                self._compile(part)

        elif isinstance(node, Group):
            if node.index is None:
                self._compile(node.node)
            else:
                self.emit(Inst(Op.SAVE, slot=node.index * 2))
                self._compile(node.node)
                self.emit(Inst(Op.SAVE, slot=node.index * 2 + 1))

        elif isinstance(node, Alternate):
            # Chain of SPLITs, each preferring the earlier option, which is
            # what makes alternation leftmost-first rather than longest-match.
            jumps: list[int] = []
            for i, option in enumerate(node.options):
                last = i == len(node.options) - 1
                if not last:
                    split = self.emit(Inst(Op.SPLIT))
                    self.program[split].x = len(self.program)
                self._compile(option)
                if not last:
                    jumps.append(self.emit(Inst(Op.JMP)))
                    self.program[split].y = len(self.program)
            for j in jumps:
                self.program[j].x = len(self.program)

        elif isinstance(node, Repeat):
            self._compile_repeat(node)

        else:
            raise TypeError(f"cannot compile {type(node).__name__}")

    def _compile_repeat(self, node: Repeat) -> None:
        low, high = node.low, node.high

        # Bounded repeats are expanded by duplication. {2,4} becomes the body
        # twice, then two optional bodies. Costs program size, keeps the VM
        # free of counters -- and counters would break the "each instruction
        # visited once per position" guarantee the whole design rests on.
        for _ in range(low):
            self._compile(node.node)

        if high is None:
            if low == 0:
                self._star(node.node, node.greedy)
            else:
                self._plus_tail(node.node, node.greedy)
        else:
            optional = high - low
            splits: list[int] = []
            for _ in range(optional):
                split = self.emit(Inst(Op.SPLIT))
                splits.append(split)
                if node.greedy:
                    self.program[split].x = len(self.program)
                else:
                    self.program[split].y = len(self.program)
                self._compile(node.node)
            end = len(self.program)
            for split in splits:
                if node.greedy:
                    self.program[split].y = end
                else:
                    self.program[split].x = end

    def _star(self, body: Node, greedy: bool) -> None:
        split = self.emit(Inst(Op.SPLIT))
        body_start = len(self.program)
        self._compile(body)
        jmp = self.emit(Inst(Op.JMP, x=split))
        end = len(self.program)
        if greedy:
            self.program[split].x, self.program[split].y = body_start, end
        else:
            self.program[split].x, self.program[split].y = end, body_start
        _ = jmp

    def _plus_tail(self, body: Node, greedy: bool) -> None:
        """The loop-back half of `x+`, given one copy of the body already emitted."""
        body_start = len(self.program)
        split = self.emit(Inst(Op.SPLIT))
        # Re-emit the body for subsequent iterations.
        again = len(self.program)
        self._compile(body)
        self.emit(Inst(Op.JMP, x=split))
        end = len(self.program)
        if greedy:
            self.program[split].x, self.program[split].y = again, end
        else:
            self.program[split].x, self.program[split].y = end, again
        _ = body_start


# ---------------------------------------------------------------- Pike VM

@dataclass
class Thread:
    pc: int
    saved: list[int | None]


@dataclass
class Match:
    text: str
    slots: list[int | None]
    group_count: int

    @property
    def start(self) -> int:
        return self.slots[0] if self.slots[0] is not None else -1

    @property
    def end(self) -> int:
        return self.slots[1] if self.slots[1] is not None else -1

    def span(self, group: int = 0) -> tuple[int, int] | None:
        lo, hi = self.slots[group * 2], self.slots[group * 2 + 1]
        return None if lo is None or hi is None else (lo, hi)

    def group(self, index: int = 0) -> str | None:
        span = self.span(index)
        return None if span is None else self.text[span[0]:span[1]]

    def groups(self) -> tuple[str | None, ...]:
        return tuple(self.group(i) for i in range(1, self.group_count + 1))

    def __repr__(self) -> str:
        return f"<Match span=({self.start}, {self.end}) {self.group()!r}>"


def _is_word(c: str) -> bool:
    return c.isalnum() or c == "_"


class Regex:
    """A compiled pattern. Construct once, use many times."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        parser = Parser(pattern)
        ast = parser.parse()
        self.group_count = parser.group_count
        self.program = Compiler().compile(ast, self.group_count)
        self.nslots = (self.group_count + 1) * 2

    # ---- VM internals
    def _assertion_holds(self, assertion: Assertion, text: str, pos: int) -> bool:
        if assertion is Assertion.LINE_START:
            return pos == 0 or text[pos - 1] == "\n"
        if assertion is Assertion.LINE_END:
            return pos == len(text) or text[pos] == "\n"
        before = pos > 0 and _is_word(text[pos - 1])
        after = pos < len(text) and _is_word(text[pos])
        boundary = before != after
        return boundary if assertion is Assertion.WORD_BOUNDARY else not boundary

    def _add_thread(self, threads: list[Thread], seen: set[int], pc: int,
                    saved: list[int | None], text: str, pos: int) -> None:
        """Follow every zero-width instruction now, so the run loop only ever
        deals with instructions that consume a character.

        `seen` is what makes this linear and what stops `(a*)*` looping: each
        instruction is added at most once per input position.
        """
        stack = [(pc, saved)]
        while stack:
            pc, saved = stack.pop()
            if pc in seen:
                continue
            seen.add(pc)
            inst = self.program[pc]

            if inst.op is Op.JMP:
                stack.append((inst.x, saved))
            elif inst.op is Op.SPLIT:
                # Pushed in reverse so the preferred branch is popped first.
                stack.append((inst.y, saved))
                stack.append((inst.x, saved))
            elif inst.op is Op.SAVE:
                updated = list(saved)
                updated[inst.slot] = pos
                stack.append((pc + 1, updated))
            elif inst.op is Op.ASSERT:
                if self._assertion_holds(inst.assertion, text, pos):
                    stack.append((pc + 1, saved))
            else:
                threads.append(Thread(pc, saved))

    def search(self, text: str, start: int = 0) -> Match | None:
        """Leftmost-first match at or after `start`, or None."""
        clist: list[Thread] = []
        seen: set[int] = set()
        matched: list[int | None] | None = None

        self._add_thread(clist, seen, 0, [None] * self.nslots, text, start)

        for pos in range(start, len(text) + 1):
            if not clist and matched is not None:
                break

            nlist: list[Thread] = []
            next_seen: set[int] = set()
            c = text[pos] if pos < len(text) else ""

            for thread in clist:
                inst = self.program[thread.pc]

                if inst.op is Op.MATCH:
                    matched = thread.saved
                    # Every thread after this one is lower priority, so stop.
                    # This is what makes the result leftmost-first rather than
                    # leftmost-longest.
                    break

                if c and inst.matches_char(c):
                    self._add_thread(nlist, next_seen, thread.pc + 1,
                                     thread.saved, text, pos + 1)

            # Only seed a new start while nothing has matched, which keeps the
            # match leftmost.
            if matched is None and pos < len(text):
                self._add_thread(nlist, next_seen, 0, [None] * self.nslots,
                                 text, pos + 1)

            clist = nlist

        return Match(text, matched, self.group_count) if matched else None

    def match(self, text: str) -> Match | None:
        """Anchored at position 0."""
        found = self.search(text)
        return found if found and found.start == 0 else None

    def fullmatch(self, text: str) -> Match | None:
        found = self.match(text)
        return found if found and found.end == len(text) else None

    def finditer(self, text: str):
        pos = 0
        while pos <= len(text):
            found = self.search(text, pos)
            if not found:
                return
            yield found
            # An empty match must still advance, or this loops forever.
            pos = found.end + 1 if found.end == found.start else found.end

    def findall(self, text: str) -> list[str]:
        return [m.group() for m in self.finditer(text)]

    def sub(self, replacement: str, text: str) -> str:
        out, last = [], 0
        for m in self.finditer(text):
            out.append(text[last:m.start])
            out.append(replacement)
            last = m.end
        out.append(text[last:])
        return "".join(out)

    def dump(self) -> str:
        lines = []
        for i, inst in enumerate(self.program):
            if inst.op in (Op.JMP,):
                detail = f"-> {inst.x}"
            elif inst.op is Op.SPLIT:
                detail = f"-> {inst.x} else {inst.y}"
            elif inst.op is Op.CHAR:
                detail = repr(inst.char)
            elif inst.op is Op.CLASS:
                body = ",".join(f"{lo}-{hi}" for lo, hi in inst.ranges)
                detail = ("^" if inst.negated else "") + f"[{body}]"
            elif inst.op is Op.SAVE:
                detail = f"slot {inst.slot}"
            elif inst.op is Op.ASSERT:
                detail = inst.assertion.value
            else:
                detail = ""
            lines.append(f"  {i:>3}  {inst.op.name:<7} {detail}")
        return "\n".join(lines)


def compile(pattern: str) -> Regex:      # noqa: A001 - mirrors re.compile
    return Regex(pattern)


def search(pattern: str, text: str) -> Match | None:
    return Regex(pattern).search(text)


def _demo() -> None:
    import re as stdlib_re

    print("\n  Instruction listing for (a|b)*c\n")
    print(Regex("(a|b)*c").dump())

    print("\n\n  Catastrophic backtracking, side by side")
    print("  pattern (a+)+b against a string of 'a' with no 'b'\n")
    print(f"  {'n':>4} {'stdlib re':>14} {'rex':>12}")
    print("  " + "-" * 32)

    pattern = r"(a+)+b"
    compiled_rex = Regex(pattern)
    compiled_re = stdlib_re.compile(pattern)
    give_up = False

    for n in (10, 16, 20, 24, 28):
        text = "a" * n

        if give_up:
            stdlib_result = "     (skipped)"
        else:
            t0 = time.perf_counter()
            compiled_re.search(text)
            elapsed = time.perf_counter() - t0
            stdlib_result = f"{elapsed * 1000:>11.1f}ms"
            if elapsed > 1.0:
                give_up = True

        t0 = time.perf_counter()
        compiled_rex.search(text)
        rex_elapsed = time.perf_counter() - t0

        print(f"  {n:>4} {stdlib_result:>14} {rex_elapsed * 1000:>9.2f}ms")

    print("\n  The stdlib column roughly doubles per step. This one does not.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="A backtracking-free regex engine.")
    ap.add_argument("pattern", nargs="?")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--dump", action="store_true", help="show the compiled program")
    ap.add_argument("--findall", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo or not args.pattern:
        _demo()
        return

    try:
        regex = Regex(args.pattern)
    except ParseError as exc:
        raise SystemExit(f"bad pattern: {exc}")

    if args.dump:
        print(regex.dump())
        if args.text is None:
            return

    if args.text is None:
        raise SystemExit("give some text to match against")

    if args.findall:
        for m in regex.finditer(args.text):
            print(f"  {m.start:>4}-{m.end:<4} {m.group()!r}  groups={m.groups()}")
        return

    found = regex.search(args.text)
    if not found:
        print("  no match")
        sys.exit(1)
    print(f"  match at {found.start}-{found.end}: {found.group()!r}")
    for i, g in enumerate(found.groups(), start=1):
        print(f"    group {i}: {g!r}")


if __name__ == "__main__":
    main()
