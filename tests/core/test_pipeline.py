from __future__ import annotations

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from nona import Nona as TopLevelNona
from nona.bench.generate import TEMPLATES, generate
from nona.core import Anonymized, DetectorError, EntityType, Mapping, Nona, NormalizedDoc, Policy, Span

ALL_REVERSIBLE = {t: Policy.PSEUDONYMIZE for t in EntityType}


def test_top_level_import() -> None:
    assert TopLevelNona is Nona


def test_end_to_end_example_from_the_api_doc() -> None:
    nona = Nona()
    result = nona.anonymize("Appelle Ahmed au +216 22 123 456 ou ahmed.bs@example.com")
    assert result.text == "Appelle PERSON_1 au PHONE_1 ou EMAIL_1"
    assert nona.restore("PERSON_1 confirmé, j'écris à EMAIL_1.", result.mapping) == (
        "Ahmed confirmé, j'écris à ahmed.bs@example.com."
    )


def test_cin_beats_phone_on_the_same_digits() -> None:
    result = Nona().anonymize("Ma CIN est 98353374.")
    assert result.text == "Ma CIN est NATIONAL_ID_1."


def test_iban_beats_the_rib_inside_it() -> None:
    result = Nona().anonymize("IBAN TN59 1000 6035 1835 9847 8831 merci")
    assert result.text == "IBAN IBAN_1 merci"


def test_cards_are_redacted_by_default() -> None:
    result = Nona().anonymize("carte 4242 4242 4242 4242")
    assert result.text == "carte [CARD]"
    assert len(result.mapping) == 0


def test_allowlist_and_denylist() -> None:
    nona = Nona(allowlist=["support@example.com"], denylist=["Projet Jasmin"])
    result = nona.anonymize("Écris à support@example.com et à sami@example.com au sujet du Projet Jasmin.")
    assert result.text == "Écris à support@example.com et à EMAIL_1 au sujet du CUSTOM_1."


def test_a_term_cannot_be_both_allowed_and_denied() -> None:
    with pytest.raises(ValueError, match="both allowed and denied"):
        Nona(allowlist=["Acme"], denylist=["ACME"])


def test_min_score_is_validated() -> None:
    with pytest.raises(ValueError, match="min_score"):
        Nona(min_score=1.5)


class Broken:
    name = "broken"
    supported_types = frozenset({EntityType.PERSON})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        raise RuntimeError(f"boom on {doc.original}")


def test_fails_closed_when_a_recognizer_crashes() -> None:
    """ADR-0003: no partly masked text; the error names the detector, never the text."""
    with pytest.raises(DetectorError) as info:
        Nona([*Nona()._recognizers, Broken()]).anonymize("Sami 22 123 456")
    assert info.value.detector == "broken"
    assert "Sami" not in str(info.value)


# --------------------------------------------------------------------------- properties

PIECES = [
    *"abcSAMI 0123456789+@.-_:/,\n",
    "PERSON_1",
    "PHONE_2",
    "٢",
    "ـ",
    "é",
    "ﬁ",
    "22 123 456",
    "22.123.456",
    "@example.com",
]
texts = st.lists(st.sampled_from(PIECES), max_size=40).map("".join)


def canonical_form(text: str, result: Anonymized) -> str:
    """`text` with every pseudonymized mention written as its entity's canonical form."""
    parts, pos = [], 0
    for span, entity in sorted(
        ((m, e) for e in result.entities for m in e.mentions), key=lambda me: me[0].start
    ):
        parts += [text[pos : span.start], entity.canonical]
        pos = span.end
    return "".join([*parts, text[pos:]])


@given(texts)
@example("PERSON_1@example.com")  # an email whose local part looks like a placeholder
@example("M.٢PERSON_1")  # #80: a piece glued to a typed placeholder
@example("a+822 123 4560")  # #80: a phone glued to a letter
@example("M.Saaـ")  # #80: a name glued to a tatweel
@example("a@example.com@example.com@example.com")  # overlapping occurrences of a known email
@example("0/ 22 123 456")  # a phone found with a space in front
@example("S+3222 123 456")  # a phone glued to a capital
def test_restore_inverts_anonymize_on_arbitrary_text(text: str) -> None:
    """FR-8: restore(anonymize(x)) == x up to linking. Each mention comes back as its entity's canonical form,
    so x itself comes back when no entity has two forms (`22 123 456` and `22.123.456` are one phone)."""
    nona = Nona(policies=ALL_REVERSIBLE)
    result = nona.anonymize(text)
    restored = nona.restore(result.text, result.mapping)
    assert restored == canonical_form(text, result)
    if all(len({m.text_in(text) for m in e.mentions}) == 1 for e in result.entities):
        assert restored == text


@given(texts)
def test_anonymize_is_deterministic(text: str) -> None:
    assert Nona().anonymize(text) == Nona().anonymize(text)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_round_trip_and_no_structured_value_left_on_generated_data(seed: int) -> None:
    nona = Nona(policies=ALL_REVERSIBLE)
    structured = {EntityType.EMAIL, EntityType.PHONE, EntityType.IP_ADDRESS, EntityType.IBAN, EntityType.RIB,
                  EntityType.CARD, EntityType.NATIONAL_ID}  # fmt: skip
    for conv in generate(len(TEMPLATES), seed=seed, prefix="g"):
        for turn in conv.turns:
            result = nona.anonymize(turn.text)
            assert nona.restore(result.text, result.mapping) == canonical_form(turn.text, result)
            for m in turn.mentions:
                if conv.entity(m.entity).type in structured:
                    assert m.text not in result.text, f"{m.text!r} left in {result.text!r}"


def test_a_structured_value_glued_to_a_letter_round_trips() -> None:
    """A value found glued to an Arabic letter is restored too (loose boundaries for structured types)."""
    n = Nona()
    for text in ["a@example.comـ", "رقمي22123456"]:
        result = n.anonymize(text)
        assert result.text != text
        assert n.restore(result.text, result.mapping) == text


@pytest.mark.parametrize(
    "text",
    [
        "M.٢PERSON_1",  # #80, found by Hypothesis
        "esmi سميرPERSON_1",
        "PERSON_1محمد m3a",
        "SamiPERSON_1 w Sami",  # the user's `PERSON_1` is never taken for Sami's
        "PERSON_1Sami w Sami",
    ],
)
def test_a_name_glued_to_a_typed_placeholder_round_trips(text: str) -> None:
    """#80: no detection cuts through a typed placeholder, so a name glued to one gets its own placeholder,
    glued to it, and restore puts the name back."""
    n = Nona(policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert result.text != text
    assert n.restore(result.text, result.mapping) == text


@pytest.mark.parametrize("text", ["M. Samiـ", "M. ـSami", "Mme Ksouriـ w Ksouri", "M.ـSb"])
def test_a_name_glued_to_a_tatweel_round_trips(text: str) -> None:
    """#80: normalization drops a tatweel, so a name's span takes in the ones at its edges, and restore puts
    the name back (each mention as its entity's canonical form: `Ksouriـ` twice)."""
    n = Nona(policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert "PERSON" in result.text
    assert n.restore(result.text, result.mapping) == canonical_form(text, result)


def test_a_phone_starting_with_a_sign_round_trips_glued_to_a_letter() -> None:
    """#80: `+216…` needs no boundary on its left; restore writes it back where propagation finds it again."""
    n = Nona()
    first = n.anonymize("a+216 22 123 456")
    assert first.text == "aPHONE_1"
    restored = n.restore(first.text, first.mapping)
    assert restored == "a+216 22 123 456"
    assert "22 123 456" not in n.anonymize_messages(["a+216 22 123 456", restored]).messages[1].text


def _conversation() -> tuple[Nona, Mapping]:
    n = Nona(policies=ALL_REVERSIBLE)
    mapping = n.anonymize_messages(["Sami w Amal Trabelsi", "phone +216 22 123 456"]).mapping
    assert [mapping.value(p) for p in ("PERSON_1", "PERSON_2", "PHONE_1")] == [
        "Sami",
        "Amal Trabelsi",
        "+216 22 123 456",
    ]
    return n, mapping


@pytest.mark.parametrize(
    "answer",
    [
        "PERSON_1PERSON_2",  # `SamiAmal Trabelsi` would leak `Sami` and `Amal` on the next request
        "xـPERSON_2 ok",  # found on the normalized view, where the tatweel is gone: `xamal trabelsi`
        "Telephone 1: see",  # not a placeholder: glued, only the exact shape counts
        "smartphone_1",
        "PERSON_1abc",
    ],
)
def test_restore_writes_a_value_only_where_propagation_finds_it_again(answer: str) -> None:
    n, mapping = _conversation()
    assert n.restore(answer, mapping) == answer


def test_restore_writes_a_glued_value_that_propagation_finds_again() -> None:
    n, mapping = _conversation()
    assert n.restore("aPHONE_1, PERSON_1PERSON_9", mapping) == "a+216 22 123 456, SamiPERSON_9"


def test_a_glued_placeholder_the_user_typed_in_a_later_message_round_trips() -> None:
    """Its number was issued in an earlier message: it's a literal of its own, never that entity."""
    n = Nona(policies=ALL_REVERSIBLE)
    conv = n.anonymize_messages(["Amal is here", "SamiPERSON_1 w Sami"])
    assert n.restore(conv.messages[1].text, conv.mapping) == "SamiPERSON_1 w Sami"


@pytest.mark.parametrize(
    ("messages", "last"),
    [
        (["أمين", "أمينـة جات"], "أمينـة جات"),  # a tatweel stretches a word: Amina isn't Amin
        (["phone 22 123 456", "ref_2022 123 456"], "ref_2022 123 456"),  # not after a real placeholder
        (["Sami w Ahmed", "Samiphone_1 ok covid_19Ahmed"], "Samiphone_1 ok covid_19Ahmed"),
    ],
)
def test_propagation_stays_out_of_longer_words(messages: list[str], last: str) -> None:
    conv = Nona(policies=ALL_REVERSIBLE).anonymize_messages(messages)
    assert conv.messages[-1].text == last


def test_a_literal_is_restored_only_where_it_is_found_again_as_one() -> None:
    """`xperson 1` would go upstream as `person 1`, read as the person issued `PERSON_1`."""
    n = Nona(policies=ALL_REVERSIBLE)
    conv = n.anonymize_messages(["Sami Trabelsi est la", "person 1 hi"])
    assert conv.mapping.value("PERSON_2") == "person 1"
    assert n.restore("xPERSON_2, PERSON_2", conv.mapping) == "xPERSON_2, person 1"


def test_a_glued_placeholder_is_reserved() -> None:
    """`SamiPERSON_1`: the user's `PERSON_1` isn't a typed placeholder there; its number is never issued."""
    assert Nona().anonymize("SamiPERSON_1 w Sami").text == "PERSON_2PERSON_1 w PERSON_2"


def test_arabic_look_ahead_spreads_listed_names_only() -> None:
    """No capitals in Arabic: an unlisted word of a name (`الثاني`, the second) must not spread."""
    assert Nona().anonymize("رقم محمد الثاني. الطابق الثاني").text == "رقم PERSON_1 الثاني. الطابق الثاني"
