# ADR-0004: Causal entity linking (placeholder numbering vs linking)

- Status: **Accepted** (2026-11-21; evidence rewritten in S3.1, see [Why Accepted](#why-accepted))
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

**Restoration** uses the entity's longest surface form in the conversation sent so far (the whole request, in the proxy).

**Scope.** The proxy uses causal mode, and so does everything else: `anonymize(text)` is the one-message case of `anonymize_messages()`, so a single text gets the same linking. A `linking: global` option (Option B) for single-document use, such as `nona scan`, where no earlier request's placeholders need to stay stable, is **deferred**: it doesn't exist yet, and it will only be added if the benchmark shows hindsight linking helps there.

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
- **Contested entities.** When two mentions in one message would each add tokens to an earlier entity and don't fit each other (`Ahmed` in m1, then `Ahmed Karoui et Ahmed Ben Salem`), neither absorbs it, because which one did would depend on word order. Both become new entities, and a later bare `Ahmed` is ambiguous. A mention inside the entity's token set adds nothing, so it never contests it (#55): before that, `Sami Trabelsi ? Oui, Sami. M. Trabelsi…` after `Sami Trabelsi` gave the same person a second placeholder, and look-ahead produces exactly such bare words.
- **An entity known only by an honorific** (`Mme`) isn't propagated: a later `Mme` is as likely someone else, and `Mme Trabelsi et Mme Karoui` must not become one person.
- **Restoration** writes the longest form with spacing collapsed, the first seen among equals, so a later `SAMI  TRABELSI` doesn't replace `Sami Trabelsi`.
- **FR-8** becomes "round trip up to linking": a mention is restored to its entity's canonical form, so the input comes back exactly only when no entity has two forms. That was already implied by "restoration uses the longest surface form"; the acceptance criterion in 01 now says so.
- **Ambiguous entities match only an identical token set.** They never take part in subset/superset checks, so an ambiguous `Ahmed` can't make a later `Ahmed Ben Salem` ambiguous too.
- **Other types** link by value, after normalization (`٢٢١٢٣٤٥٦` = `22123456`, #55): letters and digits for numbers (`22 123 456` = `22.123.456`), folded text for the rest.
- **A typed placeholder is a hard boundary** (#55). The resolver cuts every span at a placeholder written in its exact shape (`PERSON_1`), so `Sami Trabelsi PERSON_1` can't become one name whose mapping holds `PERSON`. The looser shapes still reserve numbers, but don't cut spans: `card 4242…` and `RIB 0812…` read as placeholders that way.
- **Email owners** (#52, #55): an exact spelling is final; a superset owner (`wassim.ferchichi` for `Wassim`) is provisional and is dropped if the person later absorbs a token the handle doesn't spell. Owners are metadata, never text, so this doesn't touch prefix stability.
- **A placeholder the user types after its number was issued** (`PHONE_1` in message 3, when `PHONE_1` was issued in message 1) can't be renumbered. It becomes an entity whose value is the literal text: it goes upstream as a fresh placeholder (`PHONE_2`) and is restored to exactly what the user wrote. Without this, restoring would silently turn the user's `PHONE_1` into the phone number. The conversation-level round-trip property test found this case.
- `anonymize(text)` is the one-message conversation, so single texts get the same linking.
- **Restore and propagation share their boundaries** (#54, [`core/boundaries.py`](../../src/nona/core/boundaries.py)). `_` separates words, and up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`). Any other letter or digit glued on stops a match on both sides, so `PERSON_1s` or `عندPERSON_1` in a reply stays a placeholder instead of being restored: a value is only written back where it can be caught again. Spotting placeholders the *user* typed keeps the looser old rule, so `عندPERSON_2` still reserves number 2. These strict boundaries apply to **names** (PERSON, LOCATION, ORGANIZATION, CUSTOM), which only a word boundary separates from ordinary words. Structured values (emails, phones, IBANs…) are distinctive on their own, so they restore and propagate with loose boundaries: `a@example.comـ`, an email glued to an Arabic letter, round-trips. Hypothesis found that case after the first version. Before this, `restore` wrote `لكنزة` from `لPERSON_1`, but propagation needed a plain word boundary, so the name went upstream in clear on the next request. A property test now restores every placeholder with affixes and re-sends it. **Redacted values propagate too**, so a redacted card or name stays redacted when the detector misses it later. A span kept as is (an ORGANIZATION) no longer hides a name inside it, whether the name was masked in an earlier message or found in this one (look-ahead also searches the message's own full names): `Clinique Ahmed Ben Salem`. Known trade-off: a short known name can now match after a proclitic inside an ordinary word (`علي` in `فعلي`). That over-masks, but not matching would let `لعلي`, restored from `لPERSON_1`, leak.
- **Look-ahead propagation inside a message** (#48). Once a message's spans are resolved, the name words of each PERSON span are searched for in the same message, with the single-word rule of forward propagation (exact text, word boundaries), and the message is resolved again. So `Chirine. Chirine Boukadida…` masks both, when a detector finds the full name but misses the lone first name, as NER models often do. A word qualifies when its own token set is inside the mention's. So `Salem` alone doesn't qualify in `Ben Salem` (the token is `bensalem`), honorifics and particles never do, and words under 3 characters or all-lowercase Latin words are skipped. Names that are also ordinary words (`Mabrouk`, `Amal`: the gazetteer's common-word list) are never spread by look-ahead: whether each occurrence is a name is left to the gazetteer's context rules, occurrence by occurrence. In `Mabrouk ! Mabrouk Jlassi`, the first is "congratulations" (#50), and look-ahead mustn't turn it into a name because of the second. This doesn't mean the gazetteer leaves a common-word name alone: capitalized, it masks one mid-sentence (0.6) and at the start of a sentence (0.5, at the default threshold), so look-ahead would add nothing there anyway. Only spans scoring 0.8 or more give words, so a weak false positive (say, a hypothetical `Bonne Route` scored 0.55) can't spread. **The gate is on the detector's score, carried on the entity** (#55): each person keeps the tokens a confident detection vouched for (its *seeds*), and a form propagated from an earlier message spreads only those. Before, a weak span came back as a 1.0 propagated span in the next message and spread anyway. From a gazetteer run without a strong cue, only listed names are seeds: `Merci Sami Bonne journée` takes `Bonne` as a surname, and spreading it masked every `Bonne` of the conversation (restore then wrote `Sami Bonne nouvelle`). Known trade-off: an unlisted surname alone before the full name (`Haddad, Sami Haddad`) is no longer caught; after a strong cue (`je m'appelle Kenza Haddad`) it still is. Telling `Haddad` from `Bonne` needs a model (S4). Look-ahead only fills gaps: a word already covered by another span (a place, an organization) keeps its type. A word found this way is a surface form like any other, so it propagates to later messages. It only reads message *i*, so prefix stability holds.

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

This section is kept as it was measured. The S3.1 review found that the multi-turn part wasn't fully held out (the shared-first-name template used listed names) and that absorption was never exercised; the next section re-measures with both fixed.

**Found while measuring:** in `Chirine. Chirine Boukadida…` (4 dev conversations), both `Chirine` leak. The full name isn't detected: only `Boukadida` is masked, propagated from an earlier message. That's a detection gap, left to S4. #48 added the look-ahead that masks the lone first name once a detector does find the full name; it doesn't move the dev numbers yet.

### Re-measured (S3.1, #58)

`dev` was extended to 137 conversations (370 turns; 57 multi-turn), so these numbers aren't comparable with the table above. The changes: the shared-first-name template uses held-out names (with a listed copy), absorption cases where the short form *is* detected, Arabic-script conversations with clitics, and over-masking controls. The harness also changed: every copy of a turn is scored (one per request), and placed mentions that leak part of the name (`PERSON_1 Boukadida`) are reported apart and left out of merge, split and ambiguity.

| PERSON | all of `dev` | multi-turn part |
|---|---|---|
| Mentions with a placeholder | 54.0% (141/261) | 42.9% (90/210) |
| of which partly leaking (left out below) | 10.6% (15/141) | 16.7% (15/90) |
| Wrong merge (entities) | 0% (0/65) | 0% (0/24) |
| Wrong split (entities with 2+ masked mentions) | 0% (0/32) | 0% (0/24) |
| Ambiguity (later mentions) | 0% (0/61) | 0% (0/51) |

Over all types: 0/184 merged, 0/68 split, 0/97 ambiguous. Of the 9 gold-`ambiguous` mentions, Nona gave 3 their own placeholder and missed 6.

- **Absorption**, now measured: with a listed first name, `Hamza`, then `Hamza Khelifi`, is one placeholder throughout. With a held-out one, `Mme Islem` is detected, and `Islem Boukadida` later links to the same placeholder, but `Boukadida` leaks: the linking is right, and the surname is a detection gap for S4.
- **Arabic script:** the first measurement showed 3 wrong splits, all from the metric. `وسامي` became `وPERSON_1`, and the clitic was read as part of the placeholder. The metric now strips text glued outside the mention, and the splits are gone.
- **Re-sent assistant turns** leak 38.8% (54/139). None of these full leaks belongs to an entity Nona had masked earlier. 9 re-sent mentions of a masked entity leak **in part**: the placeholder goes up with a word of the name next to it that detection never caught (`PERSON_1 Boukadida`), or, for an email, with that word elsewhere in the turn.

### Why Accepted

The decision rests on **tests**, not on the benchmark:

- **Properties** (Hypothesis): prefix stability of the texts, `anonymize_messages(msgs[:k]).texts == anonymize_messages(msgs).texts[:k]`, both on scripted detectors and on the default recognizers; the conversation round trip; and restoring every placeholder with affixes glued on, then re-sending it, never lets the value through.
- **Unit tests** for every rule above: the worked example, short→long absorption, contested entities, ambiguity, honorifics, look-ahead, seeds, email owners, typed and literal placeholders.

The benchmark is **consistent with** the decision, and no more: over every mention it could place, linking makes no wrong merge, split or ambiguity decision (0/184, 0/68, 0/97 on the dev set above). That's only as broad as detection. Just over half the PERSON mentions get a placeholder, and the hard cases are thin on `dev`: absorption on a held-out name has one template, whose surname isn't even detected, and only 9 mentions are gold-ambiguous. Both are **re-measured in S4**, with NER and on the frozen `test` set. If causal linking then makes wrong decisions that hindsight would avoid, this ADR is reopened.

What the benchmark does show: **no issued placeholder is leaked back in full** on re-sent assistant turns. The claim under *Consequences* covers full leaks only. The 9 partial ones above are words that detection missed, which forward propagation can't re-catch because they were never known forms.

## Consequences

- ➕ **Prefix stability is a property that can be tested.** For any conversation `msgs` and any `k`, `anonymize_messages(msgs[:k]).texts == anonymize_messages(msgs).texts[:k]`. It holds for the **texts**, which is what goes upstream and what the prompt cache sees. Entity metadata (`canonical`, `owner`) and the mapping describe the whole conversation sent so far, so a later message can lengthen a canonical form or settle an owner. A Hypothesis property test checks it, next to the round-trip test of FR-8.
- ➕ **Restored answers can't leak in full on the next turn**, provided the earlier mention is still in the history. The model can only produce placeholders Nona issued, and every issued entity has a surface form in an earlier message, so forward propagation re-catches the restored text. A word of the name that detection never caught can still go up next to the placeholder (a partial leak): it was never a known form. This is the main reason forward propagation is **required**, not optional.
- ➕ Forward propagation also improves recall in general. A name the NER catches once gets masked everywhere after it.
- ➖ **No hindsight.** If the user meant Ahmed Karoui in m1, that's a wrong merge: it costs utility, but nothing leaks. And a name missed in an early message stays unmasked in that message. That message already left the machine in an earlier request, so masking it now would protect little.
- ➖ **Ambiguous mentions add placeholders** (`PERSON_3` above), so the model sees an extra "person". A recency heuristic is a possible v0.2 alternative, adopted only if the benchmark shows it lowers utility loss without raising the wrong-merge rate.
- ➖ A mention matching two earlier entities at once (`Ahmed Karoui` after separate `Ahmed` and `Karoui` entities) becomes ambiguous rather than merging both. That merge would renumber earlier messages, which this ADR forbids. Rare. Accepted.
- ➖ Stability assumes detection gives the same output for the same input. A model upgrade, a config change or a restart with different numerics can renumber placeholders. Each request is still self-consistent. **The detection-cache key must include the model revision and a hash of the detection config.** Done in #40: the key is `sha256(fingerprint + text)`, the fingerprint covering the nona version and every recognizer's type and configuration (and the model revision from S4).
- ➖ If the client **truncates** the history, the mention that forward propagation depends on can disappear. That case belongs to the stateful mode (ADR-0002, v0.2), and it's documented as a known leak path until then.
- 📏 **Benchmark impact.** The dataset needs **multi-turn conversations** with **conversation-level gold entity ids**, not only spans. New metrics: wrong-merge rate, wrong-split rate, ambiguity rate, and the leak rate on re-sent assistant turns.
