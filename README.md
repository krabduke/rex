# rex

A regular expression engine that cannot blow up.

Python's `re`, like Perl's and JavaScript's, backtracks. On patterns with
nested quantifiers it explores exponentially many paths and hangs. That is not
a bug in the pattern — it is a property of the algorithm, and it is a real
denial-of-service vector anywhere user input reaches a regex.

```
$ python3 rex.py --demo

  pattern (a+)+b against a string of 'a' with no 'b'

     n      stdlib re          rex
  --------------------------------
    10          0.1ms      0.12ms
    16          5.1ms      0.11ms
    20         78.0ms      0.13ms
    24       1227.2ms      0.20ms
    28      (skipped)      0.15ms
```

The stdlib column roughly doubles per step. This one does not.

## How

Patterns compile to a small instruction set — `CHAR`, `CLASS`, `SPLIT`, `JMP`,
`SAVE`, `ASSERT`, `MATCH` — and a Pike VM simulates every branch simultaneously,
one input character at a time. A per-position visited set means each
instruction is entered at most once per position, which is both what bounds the
work at O(pattern × text) and what stops `(a*)*` looping forever.

Capture groups and leftmost-first semantics survive: `SAVE` instructions carry
per-thread slots, and priority ordering in the thread list reproduces the same
answers a backtracking engine gives.

```
$ python3 rex.py --dump '(a|b)*c'
    0  SAVE    slot 0
    1  SPLIT   -> 2 else 9
    2  SAVE    slot 2
    3  SPLIT   -> 4 else 6
    4  CHAR    'a'
    ...
```

## Testing

The suite is mostly **differential**: for a corpus of ~70 patterns against ~35
inputs, rex must agree with Python's `re` on the match span and every capture
group. Hand-written expectations would only encode what I already believed.

## One documented divergence

For a quantified group whose body can match empty — `(a*)*`, `(|a)*` — the
match span agrees with Python but the **capture group** may not. On `(a*)*b`
against `"aaab"`, Python reports group 1 as the empty match at position 3; this
reports `"aaa"` at 0–3.

Python gets its answer by running one extra empty iteration and then noticing
it made no progress. This engine cannot: the visited set that guarantees linear
time is exactly what prevents re-entering an instruction at the same position,
and that guarantee is the whole point. RE2 and Go's `regexp` diverge identically
and for the same reason.

There is a test pinning the exact difference, and a separate test asserting the
*span* always agrees — if that ever broke it would be a real bug, not a known
difference.

## Supported

Literals, `.`, classes with ranges and negation, `\d \w \s` and negations,
`* + ? {m,n}` greedy and lazy, alternation, capturing and `(?:)` groups,
`^ $ \b \B`. `search` `match` `fullmatch` `finditer` `findall` `sub`.

Not supported: backreferences and lookaround — both need more than a finite
automaton, which is the trade that buys the linear time. Named groups,
possessive quantifiers, and Unicode property classes are simply unimplemented,
and are rejected rather than silently misread.

Stdlib only. Tests: `python3 -m pytest test_rex.py` (118 tests)
