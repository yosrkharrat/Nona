from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona.bench.generate import TEMPLATES, generate
from nona.core import EntityType, normalize
from nona.core.recognizers import CinRecognizer


def found(text: str) -> list[str]:
    return [span.text_in(text) for span in CinRecognizer().detect(normalize(text))]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mon numéro de CIN est le 08123456.", ["08123456"]),
        ("CIN: 08123456", ["08123456"]),
        ("C.I.N 08123456", ["08123456"]),
        ("ma carte d'identité porte le numéro 12345678", ["12345678"]),
        ("08123456 c'est ma cin", ["08123456"]),  # context after the number
        ("El CIN mte3i 00806360, 7achti bih", ["00806360"]),
        ("رقم بطاقة التعريف 08123456", ["08123456"]),
        ("بطاقة ١٢٣٤٥٦٧٨", ["١٢٣٤٥٦٧٨"]),  # Eastern Arabic digits, reported as written
        ("ب.ت.و عدد 08123456", ["08123456"]),
        ("national ID 08123456", ["08123456"]),
        ("cin one two three four 08123456", ["08123456"]),  # context exactly 5 tokens away
        ("CIN 08123456 et 01234567", ["08123456", "01234567"]),
        # A cue then two numbers: the second is left out when its own label sits right before it (#100).
        ("رقم بطاقة التعريف 07311842 و رقم الفاتورة 52930417", ["07311842"]),
        ("رقم بطاقة التعريف 07311842 والفاتورة 52930417", ["07311842"]),
        ("رقم بطاقة التعريف 07311842 للفاتورة 52930417", ["07311842"]),
        ("CIN 07311842, facture 52930417", ["07311842"]),
        ("CIN 07311842 puis la facture 52930417", ["07311842"]),
        ("CIN 07311842, commande n° 52930417", ["07311842"]),
        ("retrait du colis : CIN 07311842, commande 52930417", ["07311842"]),
        ("commande 52930417 et 07311842 c'est ma CIN", ["07311842"]),  # out of the window, as before
        # Nothing else is dropped: a label away from the number, or one before the cue, changes nothing.
        ("ma CIN pour la commande : 08123456", ["08123456"]),
        ("Mon tel 22123456 et la CIN pour la commande 07311842", ["22123456", "07311842"]),
        ("tel 22123456, CIN du colis : 07311842", ["22123456", "07311842"]),
        ("Pour le colis, 07311842 c'est ma CIN et 22123456 mon numéro", ["07311842", "22123456"]),
        ("CIN 08123456, commande passée avec CIN 01234567", ["08123456", "01234567"]),
        ("CIN 08123456 / 08123457", ["08123456", "08123457"]),
        ("CIN 08123456, 08123457 et 08123458", ["08123456", "08123457", "08123458"]),
        ("CIN 08123456 ; 01234567", ["08123456", "01234567"]),
        ("CIN 08123456 ou bien 01234567", ["08123456", "01234567"]),
        ("CIN 08123456 (ancienne 01234567)", ["08123456", "01234567"]),
        ("cin 08123456 خاطر 01234567", ["08123456", "01234567"]),
        ("بطاقة ٠٨١٢٣٤٥٦ و ٠٨١٢٣٤٥٧", ["٠٨١٢٣٤٥٦", "٠٨١٢٣٤٥٧"]),
    ],
)
def test_cin_found(text: str, expected: list[str]) -> None:
    assert found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "commande 98353374",
        "Ma commande n° 98353374 n'est toujours pas arrivée",
        "cin one two three four five 08123456",  # 6 tokens away
        "national id a b c d 08123456",  # a two-word cue must fit in the window whole
        "08123456 a b c d national id",
        "CIN 0812345",  # 7 digits
        "CIN 081234567",  # 9 digits
        "carte 4242 4242 4242 4242",
        "CIN 12/03/2025",
        "CIN 22.123.456",  # grouped like a phone, not a CIN
        "cin12345678",  # glued to the context word
        "prix 12345678,50 dinars, carte",  # a decimal number
        "no id here 08123456",
    ],
)
def test_cin_not_found(text: str) -> None:
    assert found(text) == []


def test_long_input_stays_fast() -> None:
    """Cues, labels and numbers on every line: detection stays linear-ish, not cubic."""
    text = "paiement carte, commande 52930417, ref 71820394, CIN 07311842\n" * 2000
    started = time.perf_counter()
    assert "07311842" in found(text)
    assert time.perf_counter() - started < 2.0


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_generated_order_numbers_are_never_reported(seed: int) -> None:
    """Distractors (8-digit order numbers, prices, dates) must not be reported as CINs."""
    for conv in generate(len(TEMPLATES), seed=seed, prefix="g"):
        for turn in conv.turns:
            reported = {(s.start, s.end) for s in CinRecognizer().detect(normalize(turn.text))}
            gold = {
                (m.start, m.end)
                for m in turn.mentions
                if conv.entity(m.entity).type is EntityType.NATIONAL_ID
            }
            assert reported == gold, turn.text
