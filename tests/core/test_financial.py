from __future__ import annotations

import pytest

from nona.core import EntityType, normalize
from nona.core.recognizers import CardRecognizer, IbanRecognizer, Recognizer, RibRecognizer

TN_IBAN = "TN59 1000 6035 1835 9847 8831"
TN_RIB = "10 006 0351835984788 31"


def rib(body: str) -> str:
    """A valid Tunisian RIB from 18 digits: append the key that makes it divisible by 97."""
    return body + f"{97 - int(body) * 100 % 97:02d}"


def found(recognizer: Recognizer, text: str) -> list[str]:
    return [span.text_in(text) for span in recognizer.detect(normalize(text))]


# --------------------------------------------------------------------------- IBAN


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"iban {TN_IBAN} merci", [TN_IBAN]),  # the word after it is not a group
        ("TN5910006035183598478831", ["TN5910006035183598478831"]),
        ("FR14 2004 1010 0505 0001 3M02 606.", ["FR14 2004 1010 0505 0001 3M02 606"]),
        ("GB82 WEST 1234 5698 7654 32", ["GB82 WEST 1234 5698 7654 32"]),
        ("DE89370400440532013000", ["DE89370400440532013000"]),
        (TN_IBAN.lower(), [TN_IBAN.lower()]),
        ("TN59-1000-6035-1835-9847-8831", ["TN59-1000-6035-1835-9847-8831"]),
        ("(IBAN: DE89 3704 0044 0532 0130 00)", ["DE89 3704 0044 0532 0130 00"]),
        (f"{TN_IBAN} ou DE89370400440532013000", [TN_IBAN, "DE89370400440532013000"]),
        (f"الايبان {TN_IBAN}", [TN_IBAN]),
        (
            "IBAN FR14 2004 1010 0505 0001 3M02 606 pour le remboursement",
            ["FR14 2004 1010 0505 0001 3M02 606"],
        ),
    ],
)
def test_iban_found(text: str, expected: list[str]) -> None:
    assert found(IbanRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "TN58 1000 6035 1835 9847 8831",  # wrong check digits
        "TN59 1000 6035 1835 9847 8832",  # wrong RIB key
        "TN59 1000 6035",  # truncated
        "TN59",
        "AB12 3456 7890 1234 56",
        "5910006035183598478831",  # no country code
        "ISBN 978-3-16-148410-0",
        "xTN5910006035183598478831",  # glued to a word
        "FR14 2004 1010 0505 0001 3M02 607",
        "no iban here",
    ],
)
def test_iban_not_found(text: str) -> None:
    assert found(IbanRecognizer(), text) == []


# --------------------------------------------------------------------------- RIB


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"RIB : {TN_RIB}", [TN_RIB]),
        ("10006035183598478831", ["10006035183598478831"]),
        ("10-006-0351835984788-31", ["10-006-0351835984788-31"]),
        ("rib 10006 03518 35984 78831 svp", ["10006 03518 35984 78831"]),
        ("١٠٠٠٦٠٣٥١٨٣٥٩٨٤٧٨٨٣١", ["١٠٠٠٦٠٣٥١٨٣٥٩٨٤٧٨٨٣١"]),
        (f"compte {rib('080011234567890123')}.", [rib("080011234567890123")]),
        (rib("250250000000123456"), [rib("250250000000123456")]),
        (rib("000000000000000000"), [rib("000000000000000000")]),
        (f"vire sur {rib('111112222222222222')} w {rib('333334444444444444')}",
         [rib("111112222222222222"), rib("333334444444444444")]),
        (f"El RIB mte3i {rib('070071234500000099')}, ab3ath", [rib("070071234500000099")]),
    ],
)  # fmt: skip
def test_rib_found(text: str, expected: list[str]) -> None:
    assert found(RibRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "10 006 0351835984788 32",  # wrong key
        "1000603518359847883",  # 19 digits
        "100060351835984788310",  # 21 digits
        "TN5910006035183598478831",  # inside a compact IBAN
        "22 123 456",
        "4111 1111 1111 1111",
        "le 27/07/2025",
        "commande 98353374",
        "x10006035183598478831",
        "no rib here",
    ],
)
def test_rib_not_found(text: str) -> None:
    assert found(RibRecognizer(), text) == []


def test_the_rib_inside_a_spaced_tunisian_iban_is_reported_too() -> None:
    """A TN IBAN is TN59 + the RIB, so both recognizers fire; the resolver (#17) keeps the longer IBAN."""
    assert found(RibRecognizer(), TN_IBAN) == ["1000 6035 1835 9847 8831"]


# --------------------------------------------------------------------------- card


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("carte 4242 4242 4242 4242", ["4242 4242 4242 4242"]),
        ("4242-4242-4242-4242", ["4242-4242-4242-4242"]),
        ("visa:4111111111111111,", ["4111111111111111"]),
        ("amex 3782 822463 10005", ["3782 822463 10005"]),
        ("5555 5555 5555 4444 exp 12/27", ["5555 5555 5555 4444"]),  # the expiry is not a group
        ("٤٢٤٢٤٢٤٢٤٢٤٢٤٢٤٢", ["٤٢٤٢٤٢٤٢٤٢٤٢٤٢٤٢"]),
        ("2223 0031 2200 3222", ["2223 0031 2200 3222"]),
        ("discover 6011 1111 1111 1117", ["6011 1111 1111 1117"]),
        ("3056 9300 0902 0004", ["3056 9300 0902 0004"]),
        ("Ma carte 5105-1051-0510-5100 tbloquat", ["5105-1051-0510-5100"]),
    ],
)
def test_card_found(text: str, expected: list[str]) -> None:
    assert found(CardRecognizer(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "4242 4242 4242 4241",  # Luhn fails
        "0000 0000 0000 0000",  # no card network starts with 0
        "1000 0000 0000 0009",  # nor with 1
        "4242 4242 4242",  # 12 digits
        "10006035183598478831",  # a RIB: 20 digits
        "22 123 456",
        "le 27/07/2025",
        "commande 98353374",
        "x4242424242424242",
        "no card here",
    ],
)
def test_card_not_found(text: str) -> None:
    assert found(CardRecognizer(), text) == []


def test_runs_are_capped_so_long_digit_texts_stay_cheap() -> None:
    text = " ".join(["1234"] * 2000)
    assert found(CardRecognizer(), text) == []  # returns quickly; no run grows past 42 characters


def test_types_and_detectors() -> None:
    recognizers = (IbanRecognizer(), RibRecognizer(), CardRecognizer())
    assert [(r.name, set(r.supported_types)) for r in recognizers] == [
        ("checksum:iban", {EntityType.IBAN}),
        ("checksum:rib", {EntityType.RIB}),
        ("checksum:card", {EntityType.CARD}),
    ]
