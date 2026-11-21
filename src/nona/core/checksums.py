"""Checksums that separate real identifiers from look-alike numbers (deep dive A, layer 1).

Shared by the recognizers and by the benchmark generator, so the data and the detector agree on what valid
means.
"""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[\s-]")


def compact(value: str) -> str:
    """Drop spaces and dashes: `TN59 1000-6035` → `TN5910006035`."""
    return _SEPARATORS.sub("", value)


def luhn_valid(number: str) -> bool:
    """Payment cards (ISO/IEC 7812). Needs 12+ digits and nothing but digits and separators."""
    digits = compact(number)
    if len(digits) < 12 or not digits.isdigit():
        return False
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _alnum_to_int(s: str) -> int:
    """IBAN digit expansion: letters become 10..35 (ISO 13616)."""
    return int("".join(str(int(c, 36)) for c in s))


def iban_check_digits(country: str, bban: str) -> str:
    return f"{98 - _alnum_to_int(bban + country + '00') % 97:02d}"


def iban_valid(iban: str) -> bool:
    """ISO 13616: 2 letters, 2 check digits, 11 to 30 alphanumerics, mod 97 of the rearranged number is 1."""
    value = compact(iban).upper()
    if not 15 <= len(value) <= 34 or not value.isascii() or not value.isalnum():
        return False
    if not (value[:2].isalpha() and value[2:4].isdigit()):
        return False
    return _alnum_to_int(value[4:] + value[:4]) % 97 == 1


def tn_rib_valid(rib: str) -> bool:
    """A Tunisian RIB is 20 digits whose last two are a key that makes the whole number divisible by 97."""
    digits = compact(rib)
    return len(digits) == 20 and digits.isascii() and digits.isdigit() and int(digits) % 97 == 0
