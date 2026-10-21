# Verification against an independent reference model

The engine in `puerto_rico/` is checked game by game against a second,
independent implementation of the rules:

| File | What it is |
|---|---|
| `rule/RULES.md` | The base-game rules restated rule by rule (identifiers such as **R5.6.4**), with the line numbers of the Puerto Rico Deluxe rulebook (Rio Grande Games, 2019) they come from. |
| `rule/CONFORMANCE.md` | How the rules become a sequence of single-player decisions (**E1–E11**), how the questions the rulebook leaves open are settled (**O1–O9**), the action encoding, and the canonical state that two implementations can compare. |
| `oracle.py` | The reference model. It was written from the two documents above, without reference to any engine code, and uses only the standard library. |
| `lockstep.py` | This repo's driver: it converts the engine's state to the canonical form of CONFORMANCE.md §4.2 (`engine_canonical`) and plays seeded games in the engine and the reference model side by side. Before every decision the legal actions and the full state must agree; at the end, the scores, the winners and the rewards. |

The first three files come unchanged from the research engine this repo shares
an ancestor with (`PuertoRico_ESWA`, commit `11d6c1183d7bc145f8c0d5e811e94136a3edea74`).
`tests/test_lockstep.py` checks that they still match these SHA-256 digests, so
an edit cannot slip in unnoticed:

```
85b5a93a9e438454aabf4f67296eb16e8b4b745acdf5c4f2bb5613c295c4d96f  oracle.py
532ef1be44e25becabd49b7a37aada70e61f86a54d8791976b91127e365b816a  rule/RULES.md
cf79f22c4a62c3a4e897e30b22c8f5379840d0a6f8b42b0e264f04527868c742  rule/CONFORMANCE.md
```

Paths inside them refer to the research engine's layout: `rule/` is this
folder's `rule/`; `env/engine.py`, `env/pr_env.py` and `configs/constants.py`
correspond to `puerto_rico/engine.py`, `puerto_rico/env.py` and
`puerto_rico/constants.py` here. CONFORMANCE.md §5–§6 report the research
engine's own checks; this repository's are recorded in
[`../CHANGES.md`](../CHANGES.md). The plain-text rulebook extraction that
RULES.md cites by line is the publisher's text and is not included.

## Running it

```bash
python -m pytest tests/test_lockstep.py                       # 30 games per player count
LOCKSTEP_GAMES=500 python -m pytest tests/test_lockstep.py    # a longer run
python -m verification.lockstep --games 5000 --players 2,3 --workers 8
```

A disagreement is reported with the seed, the policy, the first field that
differs and the last few actions, so it can be replayed. The comparison cannot
catch an error shared by the specification and both implementations; the rule
tests in `tests/test_engine_rules.py` and `tests/test_decisions.py` pin the
rulebook's own wording and examples for that.
