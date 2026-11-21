# ADR-0004: Causal entity linking (placeholder numbering vs linking)

- Status: **Accepted** (2026-11-21, on the S3 benchmark: see [Measured](#measured-s3-42))
- Date: 2026-11-15
- Related: [ADR-0002 stateless by default](0002-stateless-by-default.md) (this ADR replaces its "renumbering" consequence), [deep dive B](../design/04-deep-dives.md#b-entity-linking-consistency)

## Context

In stateless mode (ADR-0002), Nona re-anonymizes the **whole conversation** on every request. Two goals pull against each other:

1. **Stable placeholders.** `PERSON_1` in message 1 should still be `PERSON_1` when message 10 arrives. If earlier placeholders change, the provider's prompt cache breaks on every turn, which costs money and latency. Behaviour also gets harder to test and debug.
2. **Good linking.** `Ahmed` and `Ahmed Ben Salem` should get the same placeholder (FR-7).

The draft rule in deep dive B fails the simplest case. A mention joins an entity only if its tokens are a **subset** of the entity's tokens. So when `Ahmed` (message 1) comes before `Ahmed Ben Salem` (message 2), they become two people.

Fixing this by re-linking the whole history with hindsight breaks goal 1. A name that appears later can make an earlier mention ambiguous, and then the earlier placeholders get renumbered.

A third problem appears once restoration works. The client receives restored answers (`Ahmed Ben Salem a été relancé`) and **sends them back** in the next request's history. Nona has to catch that name again in an assistant message. If the NER model misses it on this second pass, the real name leaks upstream, even though Nona issued the placeholder itself one turn earlier.

## Options

| | Option | Linking quality | Earlier placeholders stable? | Notes |
|---|---|---|---|---|
| A | Incremental, subset-only (current draft) | ✗ fails short→long (`Ahmed` then `Ahmed Ben Salem`) | ✓ | Simple but wrong on the most common pattern |
| B | Global re-link of the whole history on every request | Best (uses hindsight) | ✗ a later name can renumber earlier ones | Each request is still internally consistent, but the prompt cache breaks |
| C | **Causal**: each message is linked using only itself and earlier messages, and earlier decisions are frozen | Good. Handles short→long by absorbing tokens | ✓ guaranteed, and testable | Can't revise an earlier merge using hindsight |
| D | Stateful vault keyed by a session id | Same as C | ✓ | Stores PII between requests. Already deferred to v0.2 by ADR-0002 |

## Decision

**Option C, causal linking at the level of whole messages, plus forward propagation of known names.**

The pipeline processes messages in order. For message *i*:

1. **Detect** spans in message *i* (detection cache, per message).
2. **Propagate forward.** Add a span for every occurrence in message *i* of a surface form (the exact text of a mention) already linked to an entity from messages before *i*. Multi-word forms are matched on the normalized text. Single-word forms are matched on word boundaries with the same capitalization, so a known name like `Amal` doesn't mask *amal* ("hope").
3. **Resolve** overlapping spans (unchanged).
4. **Link** each mention in message *i* to the entities known so far. Inside a message, the linker can look ahead: a message never changes once it's sent.
   - A mention is *compatible* with an entity of the same type if their token sets are a **subset or a superset** of each other.
   - **Exactly one** compatible entity: join it. If the mention has new tokens, the entity **absorbs** them (`{ahmed}` → `{ahmed, ben, salem}`). Its number does not change.
   - **No** compatible entity: create a new entity with the next number for its type.
   - **Two or more** compatible entities: the mention is *ambiguous*. It gets its own placeholder, and every later identical ambiguous form reuses that placeholder.
5. **Freeze.** Entities from earlier messages are never merged with each other, split, or renumbered. They can only absorb tokens.

**Restoration** uses the entity's longest surface form seen so far.

**Scope.** The proxy uses causal mode. A `linking: global` option (Option B) exists for single-document use, such as `nona scan` or the library's `anonymize()` on one text, where there's no earlier request whose placeholders need to stay stable.

### Worked example

```
m1  user       Rappelle Ahmed demain.
m2  user       Ahmed Ben Salem a envoyé le devis, son mail ahmed.bs@gmail.com
m3  user       Ahmed Karoui veut aussi un rdv.
a3  assistant  (restored) J'ai relancé Ahmed Ben Salem et Ahmed Karoui.
m4  user       Ahmed a confirmé ?
```

| Message | Causal (chosen) | Global re-link (rejected for the proxy) |
|---|---|---|
| m1 `Ahmed` | `PERSON_1` {ahmed} | Request 1: `PERSON_1`. From request 3 on, `Ahmed` is ambiguous, so the numbering changes |
| m2 `Ahmed Ben Salem` | ⊇ PERSON_1, single match → `PERSON_1`, which absorbs {ben, salem}. Email → `EMAIL_1`, linked to PERSON_1 | `PERSON_2` from request 3 on |
| m3 `Ahmed Karoui` | Not compatible with {ahmed, ben, salem} → `PERSON_2` | `PERSON_3` |
| a3 (re-sent by the client) | NER may miss the names. **Forward propagation** catches `Ahmed Ben Salem` and `Ahmed Karoui` because both are known forms → `PERSON_1`, `PERSON_2` | Same, if detected |
| m4 `Ahmed` | Compatible with PERSON_1 and PERSON_2 → ambiguous → `PERSON_3` | `PERSON_1` (ambiguous) |
| Prompt cache from request 3 on | ✓ prefix unchanged | ✗ broken from m1 |

The causal reading of m1 matches what the user could have meant when they wrote it: at that point only one Ahmed existed in the conversation.

### Implementation notes (S3, #38)

Details the decision left open, settled while implementing it in [`core/linking.py`](../../src/nona/core/linking.py):

- **Look-ahead inside a message** means PERSON mentions are linked longest first. In `Ahmed, Ahmed Karoui et Ahmed Ben Salem`, linking in reading order would let the bare `Ahmed` absorb `Karoui`; longest first makes it ambiguous, which is what it is. New entities are still numbered by first appearance.
- **Contested entities.** When two mentions in one message fit an earlier entity but not each other (`Ahmed` in m1, then `Ahmed Karoui et Ahmed Ben Salem`), neither absorbs it, because which one did would depend on word order. Both become new entities, and a later bare `Ahmed` is ambiguous.
- **An entity known only by an honorific** (`Mme`) isn't propagated: a later `Mme` is as likely someone else, and `Mme Trabelsi et Mme Karoui` must not become one person.
- **Restoration** writes the longest form with spacing collapsed, the first seen among equals, so a later `SAMI  TRABELSI` doesn't replace `Sami Trabelsi`.
- **FR-8** becomes "round trip up to linking": a mention is restored to its entity's canonical form, so the input comes back exactly only when no entity has two forms. That was already implied by "restoration uses the longest surface form"; the acceptance criterion in 01 now says so.
- **Ambiguous entities match only an identical token set.** They never take part in subset/superset checks, so an ambiguous `Ahmed` can't make a later `Ahmed Ben Salem` ambiguous too.
- **Other types** link by value: letters and digits for numbers (`22 123 456` = `22.123.456`), folded text for the rest.
- **A placeholder the user types after its number was issued** (`PHONE_1` in message 3, when `PHONE_1` was issued in message 1) can't be renumbered. It becomes an entity whose value is the literal text: it goes upstream as a fresh placeholder (`PHONE_2`) and is restored to exactly what the user wrote. Without this, restoring would silently turn the user's `PHONE_1` into the phone number. The conversation-level round-trip property test found this case.
- `anonymize(text)` is the one-message conversation, so single texts get the same linking.

### Measured (S3, #42)

`nona` system on `dev` (120 conversations, 292 turns; 40 of them multi-turn). The linking metrics are defined in [05 §6](../design/05-dataset.md#6-how-leaks-are-counted-input-to-the-leak-rate-script). They read placeholders from the payload alone and only count mentions that got one. A name that leaks in full is a detection failure, already counted by the leak rate.

| PERSON | all of `dev` | multi-turn part |
|---|---|---|
| Mentions with a placeholder | 48.8% (103/211) | 32.5% (52/160) |
| Wrong merge (entities) | 0% (0/57) | 0% (0/16) |
| Wrong split (entities with 2+ masked mentions) | 0% (0/24) | 0% (0/16) |
| Ambiguity (later mentions) | 0% (0/46) | 0% (0/36) |

Over all types: 0/170 merged, 0/51 split, 0/73 ambiguous. The 8 gold-`ambiguous` mentions (a shared first name) are skipped by these rates. Nona gave 4 of them their own placeholder, as the decision says, and missed the other 4.

**Re-sent assistant turns** leak 43.6% (48/110) of direct identifiers. None of them belongs to an entity Nona had already masked in an earlier turn: every one is a name that detection never caught. So the claim under *Consequences* holds on `dev` (an issued placeholder is never leaked back), and the remaining leaks are recall, which S4's NER addresses.

**Why Accepted:** the short→long pattern links (`Boukadida`, then `Chirine Boukadida` → one placeholder), two people sharing a first name stay apart, prefix stability and round trip are property-tested, and no issued placeholder leaks back. The rates are only as broad as detection: with half the PERSON mentions masked, they say linking makes no wrong decision on what it sees, not that every hard case is covered. They are re-measured on the frozen test set in S4.

**Found while measuring:** a bare first name before the full name in the same message (`Chirine. Chirine Boukadida…`) leaks, because forward propagation only searches earlier messages (#48).

## Consequences

- ➕ **Prefix stability is a property that can be tested.** For any conversation `msgs` and any `k`, `anonymize(msgs[:k]) == anonymize(msgs)[:k]`. This becomes a Hypothesis property test next to the round-trip test of FR-8.
- ➕ **Restored answers can't leak on the next turn**, provided the earlier mention is still in the history. The model can only produce placeholders Nona issued, and every issued entity has a surface form in an earlier message, so forward propagation re-catches the restored text. This is the main reason forward propagation is **required**, not optional.
- ➕ Forward propagation also improves recall in general. A name the NER catches once gets masked everywhere after it.
- ➖ **No hindsight.** If the user meant Ahmed Karoui in m1, that's a wrong merge: it costs utility, but nothing leaks. And a name missed in an early message stays unmasked in that message. That message already left the machine in an earlier request, so masking it now would protect little.
- ➖ **Ambiguous mentions add placeholders** (`PERSON_3` above), so the model sees an extra "person". A recency heuristic is a possible v0.2 alternative, adopted only if the benchmark shows it lowers utility loss without raising the wrong-merge rate.
- ➖ An entity matching two earlier entities at once (`Ahmed Ben Salem` after separate `Ahmed` and `Salem` entities) becomes ambiguous rather than merging both. That merge would renumber earlier messages, which this ADR forbids. Rare. Accepted.
- ➖ Stability assumes detection gives the same output for the same input. A model upgrade, a config change or a restart with different numerics can renumber placeholders. Each request is still self-consistent. **The detection-cache key must include the model revision and a hash of the detection config.**
- ➖ If the client **truncates** the history, the mention that forward propagation depends on can disappear. That case belongs to the stateful mode (ADR-0002, v0.2), and it's documented as a known leak path until then.
- 📏 **Benchmark impact.** The dataset needs **multi-turn conversations** with **conversation-level gold entity ids**, not only spans. New metrics: wrong-merge rate, wrong-split rate, ambiguity rate, and the leak rate on re-sent assistant turns.
