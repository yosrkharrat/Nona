from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona.bench.dataset import Conversation, parse_conversation
from nona.bench.generate import MULTI_TURN_TEMPLATES, TEMPLATES, generate
from nona.bench.metrics import (
    Linking,
    MentionResult,
    Outcome,
    Rate,
    Report,
    changed_ranges,
    classify,
    compact,
    contains_word,
    evaluate,
    identifying_tokens,
    link,
    normalize,
    placeholders,
)
from nona.bench.systems import IdentitySystem
from nona.core import EntityType

P = EntityType.PERSON

# --------------------------------------------------------------------------- normalization


def test_normalize_folds_case_width_digits_and_spaces() -> None:
    assert normalize("  Sami\u00a0 TRABELSI\n") == "sami trabelsi"
    assert normalize("٩٨ ٧٦٥ ٤٣٢") == "98 765 432"
    assert normalize("۰۱۲") == "012"  # Persian digits too
    assert normalize("ＡＢＣ１２") == "abc12"  # noqa: RUF001 - full-width forms on purpose (NFKC)


def test_compact_keeps_only_letters_and_digits() -> None:
    assert compact("+216 22.123-456") == compact("21622123456")


@pytest.mark.parametrize(
    ("haystack", "needle", "found"),
    [
        ("appelle sami demain", "sami", True),
        ("appelle samir demain", "sami", False),
        ("appelle usami", "sami", False),
        ("sami", "sami", True),
        ("(sami)", "sami", True),
        ("نبعث لسامي الفلوس", "سامي", True),  # proclitic ل
        ("وبسامي", "سامي", True),  # two proclitics
        ("كتابسامي", "سامي", False),  # not a clitic: part of another word
        ("mail: a.b@example.com.", "a.b@example.com", True),
        ("x", "", False),
    ],
)
def test_contains_word_respects_word_boundaries(haystack: str, needle: str, found: bool) -> None:
    assert contains_word(haystack, needle) is found


# --------------------------------------------------------------------------- classify


@pytest.mark.parametrize(
    ("mention", "etype", "payload", "outcome"),
    [
        ("Sami Trabelsi", P, "rappelle Sami Trabelsi demain", Outcome.LEAKED),
        ("Sami Trabelsi", P, "rappelle SAMI  trabelsi demain", Outcome.LEAKED),
        ("Sami Trabelsi", P, "rappelle PERSON_1 demain", Outcome.PROTECTED),
        ("Sami Trabelsi", P, "rappelle PERSON_1 Trabelsi demain", Outcome.PARTIAL),
        ("Sami", P, "rappelle Samir demain", Outcome.PROTECTED),
        ("Ahmed Ben Salem", P, "PERSON_1 Ben PERSON_2", Outcome.PROTECTED),  # `ben` alone identifies no one
        ("22 123 456", EntityType.PHONE, "tel 22.123.456", Outcome.LEAKED),
        ("+216 22 123 456", EntityType.PHONE, "tel 22123456", Outcome.LEAKED),  # without the country code
        ("+33 6 12 34 56 78", EntityType.PHONE, "tel 06.12.34.56.78", Outcome.LEAKED),
        ("06 12 34 56 78", EntityType.PHONE, "tel +33612345678", Outcome.LEAKED),
        ("22 123 456", EntityType.PHONE, "tel 22 PHONE_1 456", Outcome.PARTIAL),
        ("+216 22 123 456", EntityType.PHONE, "tel +216 PHONE_1", Outcome.PROTECTED),  # country code only
        ("٩٨ ٧٦٥ ٤٣٢", EntityType.PHONE, "nomrou 98765432", Outcome.LEAKED),
        ("ahmed.bs@example.com", EntityType.EMAIL, "write to EMAIL_1", Outcome.PROTECTED),
        ("ahmed.bs@example.com", EntityType.EMAIL, "write to EMAIL_1, not example.com", Outcome.PROTECTED),
        ("ahmed.bs@example.com", EntityType.EMAIL, "write to ahmed@EMAIL_1", Outcome.PARTIAL),
        ("TN59 1000 6035 1835 9847 8831", EntityType.IBAN, "iban TN59 IBAN_1", Outcome.PROTECTED),
        ("TN59 1000 6035 1835 9847 8831", EntityType.IBAN, "TN5910006035183598478831", Outcome.LEAKED),
        ("00806360", EntityType.NATIONAL_ID, "CIN 00 806 360", Outcome.LEAKED),
    ],
)
def test_classify(mention: str, etype: EntityType, payload: str, outcome: Outcome) -> None:
    assert classify(mention, etype, payload) is outcome


def test_identifying_tokens_drop_short_and_non_identifying_parts() -> None:
    assert identifying_tokens("Ahmed Ben Salem", P) == ["ahmed", "salem"]
    assert identifying_tokens("a.bensalem@example.org", EntityType.EMAIL) == ["bensalem"]
    assert identifying_tokens("+216 22 123 456", EntityType.PHONE) == ["123", "456"]


# --------------------------------------------------------------------------- over-masking


def test_changed_ranges_of_identical_text_is_empty() -> None:
    assert changed_ranges("Rappelle Sami au 22 123 456.", "Rappelle Sami au 22 123 456.") == []


def test_changed_ranges_finds_each_replacement() -> None:
    original = "Rappelle Sami au 22 123 456 demain."
    payload = "Rappelle PERSON_1 au PHONE_1 demain."
    spans = [original[s:e] for s, e in changed_ranges(original, payload)]
    assert spans == ["Sami", "22 123 456"]


def test_changed_ranges_finds_deletions() -> None:
    original = "Envoie le fichier à Sami ce soir"
    assert [original[s:e] for s, e in changed_ranges(original, "Envoie le fichier ce soir")] == ["à Sami"]


# --------------------------------------------------------------------------- placeholders


def _spans(text: str, *parts: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        start = text.index(part)
        out.append((start, start + len(part)))
    return out


def test_placeholders_reads_what_replaced_each_span() -> None:
    text = "Ahmed Ben Salem a appelé Sami au 22 123 456."
    payload = "PERSON_1 a appelé PERSON_2 au PHONE_1."
    spans = _spans(text, "Ahmed Ben Salem", "Sami", "22 123 456")
    assert placeholders(text, payload, spans) == ["PERSON_1", "PERSON_2", "PHONE_1"]


def test_placeholders_of_an_unchanged_or_deleted_span_is_none() -> None:
    text = "Rappelle Sami demain."
    spans = _spans(text, "Sami")
    assert placeholders(text, text, spans) == [None]
    assert placeholders(text, "Rappelle demain.", spans) == [None]


def test_placeholders_keeps_the_replaced_part_of_a_span() -> None:
    text = "Mme Trabelsi est là."
    assert placeholders(text, "Mme PERSON_1 est là.", _spans(text, "Mme Trabelsi")) == ["PERSON_1"]


def test_placeholders_pairs_adjacent_spans_token_by_token() -> None:
    text = "Merci Sami Kais"
    assert placeholders(text, "Merci PERSON_1 PERSON_2", _spans(text, "Sami", "Kais")) == [
        "PERSON_1",
        "PERSON_2",
    ]


def test_placeholders_ignores_a_rewritten_neighbour() -> None:
    text = "Merci M. Sami"
    assert placeholders(text, "Merci Monsieur PERSON_1", _spans(text, "Sami")) == ["PERSON_1"]
    text = "Ahmed Ben Salem demain"
    assert placeholders(text, "PERSON_1 DATE_1", _spans(text, "Ahmed Ben Salem")) == ["PERSON_1"]


def test_placeholders_pairs_adjacent_spans_of_different_lengths() -> None:
    text = "Merci Sami Ben Ali"
    assert placeholders(text, "Merci PERSON_1 PERSON_2", _spans(text, "Sami", "Ben Ali")) == [
        "PERSON_1",
        "PERSON_2",
    ]


def test_placeholders_does_not_attribute_a_replacement_shared_by_two_spans() -> None:
    text = "Merci Sami Kais"
    assert placeholders(text, "Merci PERSON_1", _spans(text, "Sami", "Kais")) == [None, None]


def test_placeholders_keeps_bracketed_redactions() -> None:
    text = "Carte 4111 1111 1111 1111 ok"
    assert placeholders(text, "Carte [CARD] ok", _spans(text, "4111 1111 1111 1111")) == ["[ CARD ]"]


# --------------------------------------------------------------------------- linking


def _m(
    entity: str,
    placeholder: str | None,
    *,
    ambiguous: bool = False,
    public: bool = False,
    outcome: Outcome = Outcome.PROTECTED,
) -> MentionResult:
    return MentionResult("c", 0, "user", "fr", P, public, entity, outcome, entity, ambiguous, placeholder)


def test_link_of_consistent_placeholders_is_clean() -> None:
    result = link([_m("e1", "PERSON_1"), _m("e2", "PERSON_2"), _m("e1", "PERSON_1")])
    assert result == Linking(Rate(3, 3), Rate(0, 2), Rate(0, 1), Rate(0, 1), Rate(0, 3))


def test_link_counts_a_wrong_merge_for_every_entity_involved() -> None:
    result = link([_m("e1", "PERSON_1"), _m("e2", "PERSON_1"), _m("e3", "PERSON_2")])
    assert result.wrong_merge == Rate(2, 3)
    assert result.wrong_split == Rate(0, 0)


def test_link_counts_a_wrong_split() -> None:
    # `Ahmed` → PERSON_1, then `Ahmed Ben Salem` → PERSON_2: option A of ADR-0004.
    result = link([_m("e1", "PERSON_1"), _m("e1", "PERSON_2"), _m("e2", "PERSON_3")])
    assert result.wrong_split == Rate(1, 1)
    assert result.ambiguity == Rate(1, 1)  # PERSON_2 belongs to no other entity: an extra person
    assert result.wrong_merge == Rate(0, 2)


def test_link_tells_ambiguity_from_borrowing_another_entity() -> None:
    result = link([_m("e1", "PERSON_1"), _m("e2", "PERSON_2"), _m("e1", "PERSON_2")])
    assert result.ambiguity == Rate(0, 1)  # e1 took e2's placeholder: a merge, not an extra person
    assert result.wrong_merge == Rate(2, 2)
    assert result.wrong_split == Rate(1, 1)


def test_link_does_not_call_a_placeholder_another_entity_uses_ambiguous() -> None:
    # e1's second mention borrows PERSON_3, which e2 uses later on: a merge, not an extra person.
    result = link([_m("e2", "PERSON_2"), _m("e1", "PERSON_1"), _m("e2", "PERSON_3"), _m("e1", "PERSON_3")])
    assert result.ambiguity == Rate(0, 2)  # PERSON_3 is shared by both entities, so it's no one's extra
    assert result.wrong_merge == Rate(2, 2)


def test_link_skips_ambiguous_public_and_unplaced_mentions() -> None:
    result = link(
        [
            _m("e1", "PERSON_1"),
            _m("e1", "PERSON_3", ambiguous=True),
            _m("e2", "PERSON_1", public=True),
            _m("e1", None),
        ]
    )
    assert result == Linking(Rate(1, 2), Rate(0, 1), Rate(0, 0), Rate(0, 0), Rate(0, 1))


def test_report_links_within_a_conversation_and_a_type() -> None:
    other = MentionResult("d", 0, "user", "fr", P, False, "x", Outcome.PROTECTED, "e1", False, "PERSON_1")
    phone = MentionResult(
        "c", 0, "user", "fr", EntityType.PHONE, False, "x", Outcome.PROTECTED, "e9", False, "PERSON_1"
    )
    report = Report((_m("e1", "PERSON_1"), other, phone), Rate(0, 0), Rate(0, 0))
    assert report.linking().wrong_merge == Rate(0, 3)
    assert report.linking((P,)).wrong_merge == Rate(0, 2)


# --------------------------------------------------------------------------- evaluate


def _conv() -> Conversation:
    return parse_conversation(
        {
            "id": "c1",
            "languages": ["fr"],
            "source": "handwritten",
            "entities": [
                {"id": "e1", "type": "PERSON", "value": "Sami Trabelsi"},
                {"id": "e2", "type": "PHONE", "value": "22 123 456", "owner": "e1"},
                {"id": "e3", "type": "PERSON", "value": "Kais Saied", "public": True},
            ],
            "turns": [
                {
                    "role": "user",
                    "text": "Rappelle Sami Trabelsi au 22 123 456, commande 48213377.",
                    "mentions": [
                        {"text": "Sami Trabelsi", "entity": "e1"},
                        {"text": "22 123 456", "entity": "e2"},
                    ],
                    "distractors": [{"text": "48213377", "note": "order"}],
                },
                {
                    "role": "assistant",
                    "text": "C'est fait, Sami a été rappelé. Kais Saied parle ce soir.",
                    "mentions": [{"text": "Sami", "entity": "e1"}, {"text": "Kais Saied", "entity": "e3"}],
                },
            ],
        }
    )


def test_identity_leaks_everything_and_masks_nothing() -> None:
    conv = _conv()
    report = Report.from_results([evaluate(conv, IdentitySystem().payloads(conv))])
    assert report.headline == Rate(2, 2)
    assert report.assistant == Rate(1, 1)  # the public figure is excluded
    assert report.masked_tokens.hits == 0
    assert report.masked_distractors == Rate(0, 1)


def test_scores_a_partially_good_system() -> None:
    conv = _conv()
    payloads = [
        "Rappelle PERSON_1 au 22 123 456, commande PHONE_2.",  # phone missed, order number over-masked
        "C'est fait, PERSON_1 a été rappelé. PERSON_2 parle ce soir.",
    ]
    report = Report.from_results([evaluate(conv, payloads)])
    assert report.headline == Rate(1, 2)
    assert report.assistant == Rate(0, 1)
    assert report.masked_distractors == Rate(1, 1)
    non_pii = report.masked_tokens
    assert non_pii.hits == 1  # `48213377`; the public figure's tokens are mentions, so not counted
    # Rappelle au commande 48213377 / C est fait a été rappelé parle ce soir
    assert non_pii.total == 13


def test_evaluate_records_each_mentions_placeholder() -> None:
    conv = _conv()
    payloads = [
        "Rappelle PERSON_1 au PHONE_1, commande 48213377.",
        "C'est fait, PERSON_2 a été rappelé. Kais Saied parle ce soir.",
    ]
    report = Report.from_results([evaluate(conv, payloads)])
    assert [m.placeholder for m in report.mentions] == ["PERSON_1", "PHONE_1", "PERSON_2", None]
    linking = report.linking()
    assert linking.placed == Rate(3, 3)  # the public figure is out of scope
    assert linking.wrong_split == Rate(1, 1)
    assert linking.ambiguity == Rate(1, 1)


def test_wrong_number_of_payloads_is_an_error() -> None:
    with pytest.raises(ValueError, match="1 payloads for 2 turns"):
        evaluate(_conv(), ["only one"])


def test_rate_formatting() -> None:
    assert str(Rate(1, 4)) == "25.0% (1/4)"
    assert str(Rate(0, 0)) == "—"
    assert Rate(0, 0).value is None


# --------------------------------------------------------------------------- properties on generated data


def _perfect(conv: Conversation) -> Sequence[str]:
    """Replaces every gold mention with a typed placeholder and touches nothing else."""
    out: list[str] = []
    numbers = {e.id: i for i, e in enumerate(conv.entities, start=1)}
    for turn in conv.turns:
        text = turn.text
        for m in sorted(turn.mentions, key=lambda m: m.start, reverse=True):
            entity = conv.entity(m.entity)
            text = f"{text[: m.start]}{entity.type.value}_{numbers[entity.id]}{text[m.end :]}"
        out.append(text)
    return out


def _mask_all(conv: Conversation) -> Sequence[str]:
    return [" ".join(f"X_{i}" for i, _ in enumerate(turn.text.split())) for turn in conv.turns]


seeds = st.integers(min_value=0, max_value=2**32 - 1)


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_identity_leaks_every_generated_mention(seed: int) -> None:
    convs = generate(len(TEMPLATES), seed=seed, prefix="g")
    report = Report.from_results(evaluate(c, IdentitySystem().payloads(c)) for c in convs)
    assert all(m.outcome is Outcome.LEAKED for m in report.mentions)
    assert report.masked_tokens.hits == 0
    assert report.masked_distractors.hits == 0


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_a_perfect_system_leaks_nothing_and_over_masks_nothing(seed: int) -> None:
    convs = generate(len(TEMPLATES), seed=seed, prefix="g")
    report = Report.from_results(evaluate(c, _perfect(c)) for c in convs)
    assert [m for m in report.mentions if m.outcome is not Outcome.PROTECTED] == []
    assert report.masked_tokens.hits == 0
    assert report.masked_distractors.hits == 0


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_masking_everything_over_masks_everything(seed: int) -> None:
    convs = generate(len(TEMPLATES), seed=seed, prefix="g")
    report = Report.from_results(evaluate(c, _mask_all(c)) for c in convs)
    assert report.headline.hits == 0
    assert report.masked_tokens.value == 1.0
    assert report.masked_distractors.value == 1.0


ALL_TEMPLATES = (*TEMPLATES, *MULTI_TURN_TEMPLATES)


def _generated(seed: int) -> list[Conversation]:
    return generate(len(ALL_TEMPLATES), seed=seed, prefix="g", templates=ALL_TEMPLATES)


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_a_perfect_system_links_perfectly(seed: int) -> None:
    report = Report.from_results(evaluate(c, _perfect(c)) for c in _generated(seed))
    linking = report.linking()
    assert linking.placed.value == 1.0
    assert linking.wrong_merge.hits == linking.wrong_split.hits == linking.ambiguity.hits == 0
    assert linking.wrong_split.total > 0


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_a_fresh_placeholder_per_mention_splits_every_entity(seed: int) -> None:
    def fresh(conv: Conversation) -> Sequence[str]:
        out: list[str] = []
        for i, turn in enumerate(conv.turns):
            text = turn.text
            for k, m in enumerate(sorted(turn.mentions, key=lambda m: m.start, reverse=True)):
                text = f"{text[: m.start]}X_{i}_{k}{text[m.end :]}"
            out.append(text)
        return out

    linking = Report.from_results(evaluate(c, fresh(c)) for c in _generated(seed)).linking()
    assert linking.wrong_split.value == 1.0
    assert linking.ambiguity.value == 1.0
    assert linking.wrong_merge.hits == 0


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_one_placeholder_for_everything_merges_every_entity_with_company(seed: int) -> None:
    def redact(conv: Conversation) -> Sequence[str]:
        out: list[str] = []
        for turn in conv.turns:
            text = turn.text
            for m in sorted(turn.mentions, key=lambda m: m.start, reverse=True):
                text = f"{text[: m.start]}REDACTED{text[m.end :]}"
            out.append(text)
        return out

    convs = _generated(seed)
    report = Report.from_results(evaluate(c, redact(c)) for c in convs)
    linking = report.linking()
    assert linking.wrong_split.hits == 0
    # Every entity is merged, except one that is alone of its type in its conversation.
    alone = sum(
        1
        for c in convs
        for t in {e.type for e in c.entities}
        if len(
            {
                m.entity
                for m in report.mentions
                if m.conversation == c.id and m.type is t and not m.public and not m.ambiguous
            }
        )
        == 1
    )
    assert linking.wrong_merge.total - linking.wrong_merge.hits == alone


# --------------------------------------------------------------------------- harness validity (#58)


@pytest.mark.parametrize("payload", ["S a m i a appelé", "S.a.m.i a appelé", "s-a-m-i"])
def test_a_name_spelled_out_is_obfuscated(payload: str) -> None:
    assert classify("Sami", P, payload) is Outcome.OBFUSCATED


def test_zero_width_characters_dont_hide_a_name() -> None:
    assert classify("Sami", P, "Sa\u200bmi a appelé") is Outcome.LEAKED


def test_spelled_out_check_keeps_word_boundaries() -> None:
    assert classify("Ali", P, "une totalisation") is Outcome.PROTECTED  # `ali` inside a word isn't the name


def test_lowercasing_is_not_masking() -> None:
    assert changed_ranges("Bonjour Monsieur", "bonjour monsieur") == []


def test_a_partly_leaking_placed_mention_is_reported_apart() -> None:
    # `Chirine PERSON_1`: the surname propagated, the first name in clear. It doesn't count as linking.
    result = link([_m("e1", "PERSON_1"), _m("e1", "PERSON_2", outcome=Outcome.PARTIAL)])
    assert result.partly_leaking == Rate(1, 2)
    assert result.wrong_split == Rate(0, 0)  # the partial mention is left out, so no split


def test_deleted_mentions_are_reported() -> None:
    c = parse_conversation(
        {
            "id": "c", "languages": ["fr"], "source": "handwritten", "tags": [],
            "entities": [{"id": "e1", "type": "PERSON", "value": "Sami"}],
            "turns": [{"role": "user", "text": "Sami arrive.",
                       "mentions": [{"text": "Sami", "entity": "e1", "start": 0, "end": 4}]}],
        }
    )  # fmt: skip
    report = Report.from_results([evaluate(c, [" arrive."])])
    assert report.headline == Rate(0, 1)
    assert report.deleted == Rate(1, 1)


def test_held_out_rate_counts_names_outside_the_gazetteer_only() -> None:
    c = parse_conversation(
        {
            "id": "c", "languages": ["fr"], "source": "handwritten", "tags": [],
            "entities": [{"id": "e1", "type": "PERSON", "value": "Sami Trabelsi"},
                         {"id": "e2", "type": "PERSON", "value": "Chiheb Drira"}],
            "turns": [{"role": "user", "text": "Sami Trabelsi et Chiheb Drira.",
                       "mentions": [{"text": "Sami Trabelsi", "entity": "e1", "start": 0, "end": 13},
                                    {"text": "Chiheb Drira", "entity": "e2", "start": 17, "end": 29}]}],
        }
    )  # fmt: skip
    report = Report.from_results([evaluate(c, ["Sami Trabelsi et Chiheb Drira."])])
    assert report.held_out == Rate(1, 1)
