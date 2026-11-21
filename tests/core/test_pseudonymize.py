from __future__ import annotations

import pytest

from nona.core import EntityType, Mapping, Span, restore
from nona.core.pseudonymize import placeholder_regex, pseudonymize

P, PH, E = EntityType.PERSON, EntityType.PHONE, EntityType.EMAIL


def spans_for(text: str, *items: tuple[str, EntityType, int]) -> list[Span]:
    """(substring, type, occurrence) → spans."""
    out: list[Span] = []
    for sub, etype, occurrence in items:
        start = -1
        for _ in range(occurrence + 1):
            start = text.index(sub, start + 1)
        out.append(Span(start, start + len(sub), etype, 1.0, "test"))
    return out


def test_placeholders_are_numbered_by_first_appearance_per_type() -> None:
    text = "Sami appelle Rim au 22 123 456, puis Sami rappelle."
    result = pseudonymize(
        text, spans_for(text, ("Sami", P, 0), ("Rim", P, 0), ("22 123 456", PH, 0), ("Sami", P, 1))
    )
    assert result.text == "PERSON_1 appelle PERSON_2 au PHONE_1, puis PERSON_1 rappelle."
    assert [(e.placeholder, e.canonical, len(e.mentions)) for e in result.entities] == [
        ("PERSON_1", "Sami", 2),
        ("PERSON_2", "Rim", 1),
        ("PHONE_1", "22 123 456", 1),
    ]


def test_numbers_already_in_the_input_are_never_issued() -> None:
    text = "Sami a écrit PERSON_1 et person 2 dans son message."
    result = pseudonymize(text, spans_for(text, ("Sami", P, 0)))
    assert result.text == "PERSON_3 a écrit PERSON_1 et person 2 dans son message."
    assert restore(result.text, result.mapping) == text


def test_policies_redact_and_keep() -> None:
    text = "Carte 4242 4242 4242 4242 chez Acme, mail a@example.com"
    spans = spans_for(
        text,
        ("4242 4242 4242 4242", EntityType.CARD, 0),
        ("Acme", EntityType.ORGANIZATION, 0),
        ("a@example.com", E, 0),
    )
    result = pseudonymize(text, spans)  # defaults: CARD redact, ORGANIZATION keep
    assert result.text == "Carte [CARD] chez Acme, mail EMAIL_1"
    assert [r.type for r in result.redacted] == [EntityType.CARD]
    assert result.mapping.placeholders == ("EMAIL_1",)
    everything = pseudonymize(text, spans, {})
    assert everything.text == "Carte CARD_1 chez ORGANIZATION_1, mail EMAIL_1"


def test_overlapping_spans_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        pseudonymize("abcdef", [Span(0, 4, P, 1.0, "a"), Span(2, 6, P, 1.0, "b")])


@pytest.mark.parametrize(
    "answer",
    ["PERSON_1", "person_1", "PERSON 1", "Person1", "**PERSON_1**", "PERSON_1's"],
)
def test_restore_tolerates_mangled_placeholders(answer: str) -> None:
    mapping = Mapping({"PERSON_1": "Sami"})
    assert restore(answer, mapping) == answer.replace(answer.strip("*").removesuffix("'s"), "Sami")


@pytest.mark.parametrize("answer", ["PERSON_2", "PERSON_10", "XPERSON_1", "PERSON_11", "PHONE_1"])
def test_restore_leaves_unissued_or_glued_placeholders(answer: str) -> None:
    assert restore(answer, Mapping({"PERSON_1": "Sami"})) == answer


def test_restore_handles_types_with_underscores() -> None:
    mapping = Mapping({"IP_ADDRESS_1": "192.0.2.1", "NATIONAL_ID_2": "08123456"})
    assert restore("IP_ADDRESS_1, ip address 1, NATIONAL ID 2", mapping) == "192.0.2.1, 192.0.2.1, 08123456"


def test_restore_with_an_empty_mapping_is_identity() -> None:
    assert restore("PERSON_1", Mapping({})) == "PERSON_1"


def test_mapping_never_prints_its_values() -> None:
    mapping = Mapping({"PERSON_1": "Sami Trabelsi"})
    assert "Sami" not in repr(mapping)
    assert "Sami" not in str(mapping)
    assert "Sami" not in repr(pseudonymize("Sami", [Span(0, 4, P, 1.0, "t")]).mapping)


def test_placeholder_regex_prefers_the_longest_type_name() -> None:
    m = placeholder_regex([EntityType.IP_ADDRESS, EntityType.CUSTOM]).search("IP_ADDRESS_3")
    assert m is not None
    assert (m.group(1), m.group(2)) == ("IP_ADDRESS", "3")
