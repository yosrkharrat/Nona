from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona.bench.generate import TEMPLATES, generate
from nona.core import EntityType, Nona, normalize
from nona.core.recognizers import (
    EmailRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    Recognizer,
    default_recognizers,
)
from nona.core.recognizers.phone import CUE_SCORE


def found(recognizer: Recognizer, text: str) -> list[str]:
    """The original substrings a recognizer reports."""
    return [span.text_in(text) for span in recognizer.detect(normalize(text))]


# --------------------------------------------------------------------------- email


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("write to ahmed.bs@example.com today", ["ahmed.bs@example.com"]),
        ("Mail: Sami.Trabelsi@Example.ORG.", ["Sami.Trabelsi@Example.ORG"]),  # case kept, final dot excluded
        ("a+tag@example.net", ["a+tag@example.net"]),
        ("first_last@sub.example.co.uk", ["first_last@sub.example.co.uk"]),
        ("<rim.jlassi@example.com>", ["rim.jlassi@example.com"]),
        ("mailto:hajer@example.com", ["hajer@example.com"]),
        ("x@example.com,y@example.org", ["x@example.com", "y@example.org"]),
        ("سامي@example.com", ["سامي@example.com"]),
        ("ena 3andi mail: med92@example.com", ["med92@example.com"]),
        ("user-name@example-mail.com", ["user-name@example-mail.com"]),
        ("ＡＨＭＥＤ@example.com", ["ＡＨＭＥＤ@example.com"]),  # noqa: RUF001 - full-width on purpose
    ],
)
def test_email_found(text: str, expected: list[str]) -> None:
    assert found(EmailRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no email here",
        "@handle on twitter",
        "ahmed@",
        "ahmed@localhost",
        "a..b@example",
        "price 3@5 dinars",
        "example.com",
        "user@-example.com",
        "user@example.c",
        "user@@example.com",
        "ahmed.@example.com",
    ],
)
def test_email_not_found(text: str) -> None:
    assert found(EmailRecognizer(), text) == []


# --------------------------------------------------------------------------- phone


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nomrou 22 123 456", ["22 123 456"]),
        ("tel 22.123.456 ok", ["22.123.456"]),
        ("a3tih 22123456", ["22123456"]),
        ("+216 22 123 456", ["+216 22 123 456"]),
        ("appelle le 00216 22 123 456", ["00216 22 123 456"]),
        ("+21622123456", ["+21622123456"]),
        ("bureau: 71 234 567", ["71 234 567"]),
        ("Rappelle au 06 12 34 56 78 avant 18h", ["06 12 34 56 78"]),
        ("06.12.34.56.78", ["06.12.34.56.78"]),
        ("+33 6 12 34 56 78", ["+33 6 12 34 56 78"]),
        ("+33612345678", ["+33612345678"]),
        ("رقمي ٢٢ ١٢٣ ٤٥٦", ["٢٢ ١٢٣ ٤٥٦"]),  # Eastern Arabic digits, reported in the original script
        ("22 123 456 w 98 765 432", ["22 123 456", "98 765 432"]),
    ],
)
def test_phone_found(text: str, expected: list[str]) -> None:
    assert found(PhoneRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "le 27/07/2025",
        "carte 4111 1111 1111 1111",
        "iban TN59 1000 6035 1835 9847 8831",
        "cin 00806360",
        "prix 950,020 dt",
        "REF-48213",
        "12 500 dinars",
        "en 2026",
        "06 12 34",
        "version 1.2.3",
        "rdv à 10:30",
        "06 39 98 12 34 56 78 90",
    ],
)
def test_phone_not_found(text: str) -> None:
    assert found(PhoneRecognizer(), text) == []


def test_phone_regions_are_configurable() -> None:
    assert found(PhoneRecognizer(regions=("FR",)), "nomrou 22 123 456") == []
    assert found(PhoneRecognizer(regions=("FR",)), "+216 22 123 456") == ["+216 22 123 456"]
    with pytest.raises(ValueError, match="region"):
        PhoneRecognizer(regions=())


# --------------------------------------------------------------------------- IP


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("server 192.0.2.1 is down", ["192.0.2.1"]),
        ("logins from 203.0.113.57.", ["203.0.113.57"]),
        ("198.51.100.241, 198.51.100.7", ["198.51.100.241", "198.51.100.7"]),
        ("ip=10.0.0.1;", ["10.0.0.1"]),
        ("dns 8.8.8.8", ["8.8.8.8"]),
        ("(192.0.2.44)", ["192.0.2.44"]),
        ("v6 2001:db8::1 ok", ["2001:db8::1"]),
        ("2001:DB8:0:0:8:800:200C:417A", ["2001:DB8:0:0:8:800:200C:417A"]),
        ("link-local fe80::1", ["fe80::1"]),
        ("loopback ::1", ["::1"]),
        ("[2001:db8:85a3::8a2e:370:7334]", ["2001:db8:85a3::8a2e:370:7334"]),
    ],
)
def test_ip_found(text: str, expected: list[str]) -> None:
    assert found(IpRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "256.1.1.1",
        "version 1.2.3",
        "1.2.3.4.5",
        "192.168.1",
        "rdv à 10:30:00",
        "::",
        "0.0.0.0",  # noqa: S104 - test data, not a bind address
        "1.2.3.4a",
        "01.02.03.04",
        "abc::def::1",
        "le 2026-11-21",
    ],
)
def test_ip_not_found(text: str) -> None:
    assert found(IpRecognizer(), text) == []


# --------------------------------------------------------------------------- all recognizers


def test_default_recognizers_cover_their_types() -> None:
    recognizers = default_recognizers()
    assert [r.name for r in recognizers] == [
        "regex:email",
        "phonenumbers",
        "regex:ip",
        "checksum:iban",
        "checksum:rib",
        "checksum:card",
        "context:cin",
    ]
    covered = frozenset().union(*(r.supported_types for r in recognizers))
    assert covered == TYPES


def test_spans_carry_type_score_and_detector() -> None:
    text = "Sami: 22 123 456, sami@example.com, 192.0.2.9"
    spans = [s for r in default_recognizers() for s in r.detect(normalize(text))]
    assert {(s.type, s.detector, s.score) for s in spans} == {
        (EntityType.PHONE, "phonenumbers", 1.0),
        (EntityType.EMAIL, "regex:email", 1.0),
        (EntityType.IP_ADDRESS, "regex:ip", 1.0),
    }


TYPES = {
    EntityType.EMAIL,
    EntityType.PHONE,
    EntityType.IP_ADDRESS,
    EntityType.IBAN,
    EntityType.RIB,
    EntityType.CARD,
    EntityType.NATIONAL_ID,
}


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_every_generated_structured_value_is_found_exactly(seed: int) -> None:
    """On generated data, each gold mention of these types is reported with exactly its gold offsets."""
    recognizers = default_recognizers()
    for conv in generate(len(TEMPLATES), seed=seed, prefix="g"):
        for turn in conv.turns:
            spans = {(s.start, s.end, s.type) for r in recognizers for s in r.detect(normalize(turn.text))}
            for m in turn.mentions:
                etype = conv.entity(m.entity).type
                if etype in TYPES:
                    assert (m.start, m.end, etype) in spans, f"{etype} {m.text!r} missed in {turn.text!r}"


# --------------------------------------------------------------------------- Tunisian conventions (#33)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("+216 (0)71 234 567", ["+216 (0)71 234 567"]),  # the French "(0)" habit
        ("00216 (0) 22 123 456", ["00216 (0) 22 123 456"]),
        ("+33 (0)6 12 34 56 78", ["+33 (0)6 12 34 56 78"]),
        ("Tél : 71 234 567\nFax : 71 234 568", ["71 234 567", "71 234 568"]),  # a fax is a fixed line
        ("فاكس 71 234 567", ["71 234 567"]),
        ("(+216) 71 234 567", ["(+216) 71 234 567"]),
        ("216 71 234 567", ["216 71 234 567"]),
        ("00 216 71 234 567", ["00 216 71 234 567"]),
        ("71 23 45 67", ["71 23 45 67"]),  # in pairs
        ("71-234-567", ["71-234-567"]),
        ("nomrou 36 123 456", ["36 123 456"]),  # a 3x mobile range
        ("el fixe 30 123 456", ["30 123 456"]),  # a 3x fixed range
        ("bureau 79 123 456", ["79 123 456"]),
    ],
)
def test_tunisian_formats_found(text: str, expected: list[str]) -> None:
    assert found(PhoneRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "appelez le numéro vert 80 100 200",  # toll-free
        "jeu concours 88 123 456",  # premium rate
        "service client 81 101 234",  # shared cost
        "service client 82 101 234",
        "+216 (0)49 123 456",  # 49 is not an assigned range
        "mobile 49 123 456",
    ],
)
def test_service_and_unassigned_numbers_not_found(text: str) -> None:
    assert found(PhoneRecognizer(), text) == []


def test_service_numbers_can_be_included() -> None:
    recognizer = PhoneRecognizer(include_service_numbers=True)
    assert found(recognizer, "numéro vert 80 100 200, jeu 88 123 456") == ["80 100 200", "88 123 456"]


# --------------------------------------------------------------------------- phone: order-cue context rule


def scores(text: str) -> list[tuple[str, float]]:
    return [(s.text_in(text), s.score) for s in PhoneRecognizer().detect(normalize(text))]


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("Ma commande n° 24176705 n'est toujours pas arrivée", "24176705"),
        ("Commande N° 24176705", "24176705"),  # matched on the normalized (folded) view
        ("Order 24176705 never arrived", "24176705"),
        ("order #24176705", "24176705"),
        ("Réf: 71 234 567", "71 234 567"),
        ("reference number: 24176705", "24176705"),
        ("facture numéro 24176705", "24176705"),
        ("colis 24176705", "24176705"),
        ("الطلبية رقم 24176705", "24176705"),
    ],
)
def test_order_cue_lowers_the_score(text: str, number: str) -> None:
    assert scores(text) == [(number, CUE_SCORE)]
    assert Nona().anonymize(text).text == text  # below the default threshold


@pytest.mark.parametrize(
    "text",
    [
        "Envoie la facture au 22 123 456",  # the cue isn't adjacent: a real phone, and a miss would leak
        "commande, appelle-moi au 22 123 456",
        "le n° 22 123 456",  # a connector alone is not a cue
        "réf +216 22 123 456",  # international format is never an order number
        "order 0021622123456",
        "Tél : 22 123 456",
    ],
)
def test_phone_without_adjacent_order_cue_keeps_full_score(text: str) -> None:
    assert [score for _, score in scores(text)] == [1.0]


def test_order_cue_applies_to_its_own_number_only() -> None:
    assert scores("Facture 71 234 567, Tél 22 123 456") == [("71 234 567", CUE_SCORE), ("22 123 456", 1.0)]
