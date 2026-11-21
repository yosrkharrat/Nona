from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona import Nona as TopLevelNona
from nona.bench.generate import TEMPLATES, generate
from nona.core import DetectorError, EntityType, Nona, NormalizedDoc, Policy, Span

ALL_REVERSIBLE = {t: Policy.PSEUDONYMIZE for t in EntityType}


def test_top_level_import() -> None:
    assert TopLevelNona is Nona


def test_end_to_end_example_from_the_api_doc() -> None:
    nona = Nona()
    result = nona.anonymize("Appelle Ahmed au +216 22 123 456 ou ahmed.bs@example.com")
    assert result.text == "Appelle Ahmed au PHONE_1 ou EMAIL_1"  # names come in S3/S4
    assert nona.restore("PHONE_1 confirmé, j'écris à EMAIL_1.", result.mapping) == (
        "+216 22 123 456 confirmé, j'écris à ahmed.bs@example.com."
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
    "@example.com",
]
texts = st.lists(st.sampled_from(PIECES), max_size=40).map("".join)


@given(texts)
def test_restore_inverts_anonymize_on_arbitrary_text(text: str) -> None:
    """FR-8: restore(anonymize(x)) == x when every type is reversible."""
    nona = Nona(policies=ALL_REVERSIBLE)
    result = nona.anonymize(text)
    assert nona.restore(result.text, result.mapping) == text


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
            assert nona.restore(result.text, result.mapping) == turn.text
            for m in turn.mentions:
                if conv.entity(m.entity).type in structured:
                    assert m.text not in result.text, f"{m.text!r} left in {result.text!r}"
