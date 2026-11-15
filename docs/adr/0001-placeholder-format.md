# ADR-0001 — Placeholder format

- Status: **Proposed** (to be confirmed by benchmark)
- Date: 2026-11-14

## Context
Detected entities must be replaced with something the LLM can reason about and that Nona can reliably map back.

## Options
| Option | Example | Restoration | Utility | Risk |
|---|---|---|---|---|
| A. Typed token | `PERSON_1` | Easy: regex | Loses gender and language cues (French agreement: `PERSON_1 est arrivé(e)`) | Models occasionally mangle it (`Person 1`) |
| B. Bracketed token | `[[PERSON_1]]` | Easiest | Same as A | Models drop or reformat brackets, especially in Markdown |
| C. Realistic surrogate | `Ahmed Ben Salem` → `Karim Trabelsi` | Hard: models inflect, translate or transliterate names (`كريم`) | Best: natural text, gender can be preserved | Surrogate may collide with a real name in the text, and restoring is ambiguous |

## Decision
Start with **A** (`TYPE_n`, case-insensitive matching that tolerates `TYPE n`). Implement **C** behind a flag in v0.2
and let the benchmark decide (utility loss vs restoration failure rate, per language).

## Consequences
- Simple and robust restoration for v0.1.
- Gender loss in French and Arabic is a known, measured weakness. It is reported, not hidden.
