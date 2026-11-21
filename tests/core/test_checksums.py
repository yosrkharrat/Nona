from __future__ import annotations

import pytest

from nona.bench.generate import TEST_CARDS
from nona.core.checksums import compact, iban_check_digits, iban_valid, luhn_valid, tn_rib_valid


@pytest.mark.parametrize("number", TEST_CARDS)
def test_every_test_card_is_luhn_valid(number: str) -> None:
    assert luhn_valid(number)


@pytest.mark.parametrize("number", ["4242424242424241", "1234567812345678", "42"])
def test_luhn_rejects_bad_numbers(number: str) -> None:
    assert not luhn_valid(number)


@pytest.mark.parametrize(
    "iban",
    ["FR14 2004 1010 0505 0001 3M02 606", "GB82 WEST 1234 5698 7654 32", "DE89370400440532013000"],
)
def test_iban_validator_accepts_published_examples(iban: str) -> None:
    assert iban_valid(iban)


@pytest.mark.parametrize("iban", ["FR15 2004 1010 0505 0001 3M02 606", "GB82", "DE89 3704 0044 0532 0130 0!"])
def test_iban_validator_rejects_bad_ibans(iban: str) -> None:
    assert not iban_valid(iban)


@pytest.mark.parametrize("number", ["4111-1111-1111-1111", "4111 1111 1111 1111", "378282246310005"])
def test_luhn_accepts_separators(number: str) -> None:
    assert luhn_valid(number)


def test_luhn_rejects_letters() -> None:
    assert not luhn_valid("4111 1111 1111 111a")


def test_iban_check_digits_round_trip() -> None:
    bban = "10006035183598478831"
    assert iban_check_digits("TN", bban) == "59"
    assert iban_valid("TN59" + bban)
    assert iban_valid("tn59 1000-6035 1835 9847 8831")  # case and separators don't matter


@pytest.mark.parametrize(
    "iban", ["1259 0000 0000 0000 0000 00", "TNAB10006035183598478831", "TN59" + "1" * 31]
)
def test_iban_rejects_bad_shapes(iban: str) -> None:
    assert not iban_valid(iban)


def test_tn_rib() -> None:
    assert tn_rib_valid("10 006 0351835984788 31")
    assert not tn_rib_valid("10 006 0351835984788 32")
    assert not tn_rib_valid("1000603518359847883")  # 19 digits


def test_compact() -> None:
    assert compact("TN59 1000-6035\t1835") == "TN59100060351835"
