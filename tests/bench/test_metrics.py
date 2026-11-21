from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona.bench.dataset import Conversation, parse_conversation
from nona.bench.generate import TEMPLATES, generate
from nona.bench.metrics import (
    Outcome,
    Rate,
    Report,
    changed_ranges,
    classify,
    compact,
    contains_word,
    evaluate,
    identifying_tokens,
    normalize,
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
