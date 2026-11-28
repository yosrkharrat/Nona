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
        ("رقم بطاقة التعريف 07311842 ورقم الطلب 52930417", ["07311842"]),
        ("رقم بطاقة التعريف 07311842، رقم الوصل 52930417", ["07311842"]),
        ("رقم بطاقة التعريف 07311842، رقم طلب 52930417", ["07311842"]),
        ("رقم بطاقة التعريف 07311842 برقم الطلب 52930417", ["07311842"]),
        ("رقم بطاقة التعريف 07311842 وفاتورة 52930417", ["07311842"]),
        ("CIN 07311842, numéro de  suivi 52930417", ["07311842"]),  # two spaces
        ("رقم بطاقة التعريف 07311842، رقم\tالطلب 52930417", ["07311842"]),  # noqa: RUF001
        ("CIN 07311842, numéro de suivi 52930417", ["07311842"]),
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
        ("رقم بطاقة التعريف ٠٨١٢٣٤٥٦ وصل ٠١٢٣٤٥٦٧", ["٠٨١٢٣٤٥٦", "٠١٢٣٤٥٦٧"]),  # وصل: also "arrived"
        ("بطاقة 08123456 طلب 01234567", ["08123456", "01234567"]),  # bare طلب: also "request"
        ("رقم بطاقة التعريف 08123456، في الطلب القديم 01234567", ["08123456", "01234567"]),  # "the request"
        ("CIN 08123456, suivi 01234567", ["08123456", "01234567"]),
        ("بطاقة 08123456 رقم اطلب 01234567", ["08123456", "01234567"]),  # "number, ask for"
        # A two-word label wrapped onto the next line is missed: it over-masks, never leaks.
        ("بطاقة التعريف 08123456 وعدد\n\nطلب 01234567", ["08123456", "01234567"]),
        ("رقم بطاقة التعريف 07311842، رقم\nالطلب 52930417", ["07311842", "52930417"]),  # noqa: RUF001
        ("رقم بطاقة التعريف 07311842، رقم\rالطلب 52930417", ["07311842", "52930417"]),  # noqa: RUF001
        ("CIN 07311842, numéro de\nsuivi 52930417", ["07311842", "52930417"]),
        ("CIN 07311842, numéro de\u2028suivi 52930417", ["07311842", "52930417"]),
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
    """Cues, labels and numbers on every line: 10x the input takes about 10x the time, not 100x or 1000x."""
    line = "paiement carte, commande 52930417, ref 71820394, CIN 07311842\n"

    def best_of_3(lines: int) -> float:
        doc = normalize(line * lines)
        times = []
        for _ in range(3):
            started = time.perf_counter()
            CinRecognizer().detect(doc)
            times.append(time.perf_counter() - started)
        return min(times)

    assert "07311842" in found(line * 10)
    small, large = best_of_3(1000), best_of_3(10000)
    assert large < 40 * small  # linear is 10x; quadratic would be 100x
    assert large < 10.0  # a generous ceiling (about 0.1 s locally) for a slowdown that is linear


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
