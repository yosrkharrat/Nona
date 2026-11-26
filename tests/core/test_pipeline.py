from __future__ import annotations

import re

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from nona import Nona as TopLevelNona
from nona.bench.generate import TEMPLATES, generate
from nona.core import Anonymized, DetectorError, EntityType, Nona, NormalizedDoc, Policy, Span
from nona.core.normalize import DROPPED, normalize

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


def _glued_to_ascii(result: Anonymized) -> bool:
    """Whether an issued placeholder touches an ASCII letter or digit in the anonymized text, or a character
    that normalizes to one (`٢`, `ﬁ`), past what normalization drops (`aـPHONE_1`): propagation looks for
    multi-word values on the normalized view (#87). A structured value may be followed by a letter (loose
    boundaries), so only a digit counts after one."""

    def ascii_word(c: str, digits_only: bool) -> bool:
        return any(x.isascii() and (x.isdigit() if digits_only else x.isalnum()) for x in normalize(c).text)

    for e in result.entities:
        name = e.type in {EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.CUSTOM}
        for m in re.finditer(re.escape(e.placeholder) + r"(?!\d)", result.text):
            before = result.text[: m.start()].rstrip("".join(DROPPED))[-1:]
            after = result.text[m.end() :].lstrip("".join(DROPPED))[:1]
            if any(ascii_word(c, False) for c in before) or any(ascii_word(c, not name) for c in after):
                return True
    return False


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
@example("M.Saaـ")  # #80: a name glued to a tatweel
@example("xphone" + "1" * 5000)  # #80: `int()` refuses more than 4300 digits
def test_restore_inverts_anonymize_on_arbitrary_text(text: str) -> None:
    """FR-8: restore(anonymize(x)) == x up to linking. Each mention comes back as its entity's canonical form,
    so x itself comes back when no entity has two forms (`22 123 456` and `22.123.456` are one phone).

    Except where detection leaves a placeholder glued to an ASCII letter or digit, also once normalized
    (`M.٢PERSON_1` → `M.PERSON_2PERSON_1`, `aPHONE_1`, `٢PHONE_1`): restore doesn't write a value there, so
    it stays masked (#87). That is judged on the anonymized text, without the boundaries restore uses, so a
    restore regression still fails.
    """
    nona = Nona(policies=ALL_REVERSIBLE)
    result = nona.anonymize(text)
    assume(not _glued_to_ascii(result))
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
    ["M. Samiـ", "M. ـSami", "Mme Ksouriـ w Ksouri", "M.ـSb", "M.Saaـ", " .ـلSami", "قول ـلسامي"],
)
def test_a_name_glued_to_a_tatweel_round_trips(text: str) -> None:
    """#80: normalization drops a tatweel, so the name's span leaves it out; up to two tatweels on either
    side, also around a proclitic, are a boundary, and restore puts the name back."""
    n = Nona(policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert "PERSON" in result.text
    assert n.restore(result.text, result.mapping) == text


@pytest.mark.parametrize(
    ("messages", "last"),
    [
        (["سامي", "قلت وـبـسامي"], "قلت وـبـPERSON_1"),  # two proclitics and two tatweels
        (["السيدة القصوري جات", "قول لـــالقصوري"], "قول لـــPERSON_1"),  # any number of tatweels
        (["السيدة القصوري جات", "قول القصُوري"], "قول PERSON_1"),  # a short vowel inside the name
        (["أمين", "تَأمين جات"], "تَأمين جات"),  # a short vowel before it: `تأمين` (insurance) isn't Amin
        (["السيد لطفي جا", "قلت ولطفي و للطفي"], "قلت وPERSON_1 و لPERSON_1"),  # a name starting with `ل`
        (["سامي", "قلت لِسامي"], "قلت لِPERSON_1"),  # a short vowel on the proclitic
        (["أمين", "أمينَة جات"], "أمينَة جات"),  # a short vowel inside a word: Amina isn't Amin
        (["Mme Zgolliـ est la", "rani m3a Zgolli"], "rani m3a PERSON_1"),
        (["السيدة القصوريـ جات", "شفت القصوري اليوم"], "شفت PERSON_1 اليوم"),
        (["السيدة القصوري جات", "قول لـالقصوري"], "قول لـPERSON_1"),  # a proclitic, then a tatweel
        (["السيدة القصوري جات", "قول القصـوري"], "قول PERSON_1"),  # a tatweel inside the name
        (["أمين", "أمينـة جات"], "أمينـة جات"),  # a tatweel inside a word: Amina isn't Amin
    ],
)
def test_propagation_and_tatweels(messages: list[str], last: str) -> None:
    conv = Nona(policies=ALL_REVERSIBLE).anonymize_messages(messages)
    assert conv.messages[-1].text == last


@pytest.mark.parametrize(
    "answer", ["قلت سَـلPERSON_1 اليوم", "™PERSON_1", " ٢PHONE_1", "ok PERSON_1\u0301 la", "aـPHONE_1"]
)
def test_a_multi_word_value_is_restored_only_where_the_normalized_view_finds_it(answer: str) -> None:
    """Propagation looks for a multi-word value on the normalized view, where `سَـل` is `سل` glued to the name,
    `™` is `tm` and `٢` is `2`, and where a combining mark would compose with the name's last letter (`í`):
    restore leaves the placeholder there."""
    n = Nona(policies=ALL_REVERSIBLE)
    first = n.anonymize_messages(["سامي الطرابلسي, phone +216 22 123 456"])
    assert first.mapping.value("PERSON_1") == "سامي الطرابلسي"
    assert n.restore(answer, first.mapping) == answer


def test_a_one_word_name_before_a_combining_mark_round_trips() -> None:
    """Propagation finds a one-word name on the original text, where a decomposed accent after it is no
    letter; only a multi-word value, found on the normalized view, would compose with it."""
    n = Nona(policies=ALL_REVERSIBLE)
    conv = n.anonymize_messages(["Mme Sami est la", "ok Sami\u0301x"])
    assert conv.messages[1].text == "ok PERSON_1\u0301x"
    assert n.restore(conv.messages[1].text, conv.mapping) == "ok Sami\u0301x"


def test_a_phone_before_a_combining_mark_round_trips() -> None:
    n = Nona(policies=ALL_REVERSIBLE)
    mapping = n.anonymize_messages(["phone +216 22 123 456"]).mapping
    assert n.restore("x PHONE_1\u0301٢ y", mapping) == "x +216 22 123 456\u0301٢ y"


@pytest.mark.parametrize("text", ["xphone" + "1" * 5000, "phone " + "1" * 5000, "xPERSON_" + "1" * 5000])
def test_a_type_word_before_a_long_digit_run_is_no_placeholder(text: str) -> None:
    """#80: placeholder numbers have at most 6 digits, as issued; `int()` refuses more than 4300."""
    n = Nona(policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert n.restore(result.text, result.mapping) == canonical_form(text, result)
    assert n.restore(text, result.mapping) == text


@pytest.mark.parametrize(
    "text",
    [
        "M.٢PERSON_1",
        "esmi سميرPERSON_1",
        "a+822 123 4560",
        "a::-@example.com",
        "٢+2522 123 456",  # glued to `2` on the normalized view
        "aـ+2522 123 456",  # glued to `a` past a tatweel
    ],
)
def test_a_placeholder_detection_leaves_glued_stays_masked(text: str) -> None:
    """#87: restore doesn't write a value where propagation couldn't find it again: the glued placeholder
    stays in the reply, and no value went upstream."""
    n = Nona(policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert result.entities
    assert not any(m.text_in(text) in result.text for e in result.entities for m in e.mentions)
    restored = n.restore(result.text, result.mapping)
    assert any(re.search(re.escape(e.placeholder) + r"(?!\d)", restored) for e in result.entities)


def test_arabic_look_ahead_spreads_listed_names_only() -> None:
    """No capitals in Arabic: an unlisted word of a name (`الثاني`, the second) must not spread."""
    assert Nona().anonymize("رقم محمد الثاني. الطابق الثاني").text == "رقم PERSON_1 الثاني. الطابق الثاني"
