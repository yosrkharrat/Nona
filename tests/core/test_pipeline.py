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


_GLUED_VALUES = ["22 123 456", "+216 22 123 456", "a@x.com", "sami.ksouri@example.tn", "10.1.2.3"]
_TYPED = ["PERSON_1", "PHONE_1", "EMAIL_2", "LOCATION_12"]


@settings(max_examples=60, deadline=None)
@given(
    st.sampled_from(_GLUED_VALUES),
    st.sampled_from(["", "_", "-", "__", "-_"]),
    st.sampled_from(_TYPED),
    st.sampled_from(["tel ", "mail: ", "", "x\n"]),
)
@example("22 123 456", "", "PERSON_1", "tel ")  # the issue's examples
@example("a@x.com", "", "PERSON_1", "mail ")
@example("a@x.com", "_", "PERSON_1", "mail ")  # the e-mail's own guard refuses `_` and `-` after it
def test_a_value_glued_before_a_typed_placeholder_is_masked(
    value: str, sep: str, typed: str, prefix: str
) -> None:
    """#85: a typed placeholder glued after a structured value hid it from its recognizer, so the value went
    upstream in clear on first mention. The typed placeholder stays, its number is never issued, and the
    answer restores to the text exactly."""
    n = Nona()
    text = f"{prefix}{value}{sep}{typed} merci"
    result = n.anonymize(text)
    assert value not in result.text
    assert typed in result.text
    assert all(e.placeholder != typed for e in result.entities)
    assert n.restore(result.text, result.mapping) == text


def test_a_value_glued_before_a_typed_placeholder_is_masked_when_re_sent() -> None:
    """The restored text goes upstream masked again in the next message."""
    n = Nona()
    first = "mail a@x.comEMAIL_1 et tel 22123456PHONE_1"
    conv = n.anonymize_messages([first, first])
    assert [m.text for m in conv.messages] == ["mail EMAIL_2EMAIL_1 et tel PHONE_2PHONE_1"] * 2


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("PERSON_1@example.com", "EMAIL_1"),  # the first pass's e-mail is kept, not cut at the placeholder
        ("PERSON_1a@x.com", "EMAIL_1"),
        ("SamiPERSON_1", "SamiPERSON_1"),  # names aren't searched again: strict boundaries, #87
    ],
)
def test_the_second_pass_only_adds(text: str, masked: str) -> None:
    assert Nona().anonymize(text).text == masked


_AR_PHONE = "\u0662\u0662\u0661\u0662\u0663\u0664\u0665\u0666"  # 22123456 in Arabic-Indic digits


@pytest.mark.parametrize(
    ("messages", "masked", "restored"),
    [
        # masking the card puts `]` before the typed PERSON_1: restore would write Sami there
        (
            ["Sami est la", "carte 4242424242424242PERSON_1 merci"],
            "carte [CARD]PERSON_2 merci",
            "carte [CARD]PERSON_1 merci",
        ),
        (["mail b@y.com", f"EMAIL_1{_AR_PHONE} merci"], "EMAIL_2PHONE_1 merci", "EMAIL_1PHONE_1 merci"),
        # glued to a digit once masked, as on main: restore doesn't touch it
        (["tel 22123456", "x 22123457PHONE_1"], "x PHONE_2PHONE_1", None),
        (["ip 10.9.9.9", "ip PERSON_1_10.1.2.3 ok"], "ip PERSON_1_IP_ADDRESS_2 ok", None),
    ],
)
def test_a_typed_placeholder_next_to_a_masked_value_never_restores_to_another_value(
    messages: list[str], masked: str, restored: str | None
) -> None:
    """Re-review of #92: once the glued value is masked, restore may find the typed placeholder. If its
    number was issued earlier, it gets a fresh one and comes back as typed."""
    n = Nona()
    conv = n.anonymize_messages(messages)
    assert conv.messages[-1].text == masked
    expected = messages[-1] if restored is None else restored
    assert n.restore(masked, conv.mapping) == expected


class Counting:
    """A custom structured recognizer that counts its passes."""

    name = "counting"
    supported_types = frozenset({EntityType.CUSTOM})

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        self.calls += 1
        return [
            Span(m.start(), m.end(), EntityType.CUSTOM, 1.0, self.name)
            for m in re.finditer(r"(?<![a-z0-9])emp-\d{5}(?![a-z0-9])", doc.text)
        ]


@pytest.mark.parametrize(
    ("text", "calls", "masked"),
    [
        ("id EMP-12345PERSON_1", 2, "id CUSTOM_1PERSON_1"),  # a custom ID is searched again
        ("PERSON_1 EMP-12345 PERSON_1", 1, "PERSON_1 CUSTOM_1 PERSON_1"),  # nothing glued, one pass
        ("Merci PERSON_1. (PERSON_1), EMP-12345", 1, "Merci PERSON_1. (PERSON_1), CUSTOM_1"),
        ("pas de placeholder", 1, "pas de placeholder"),
    ],
)
def test_the_second_pass_runs_only_when_a_placeholder_is_glued(text: str, calls: int, masked: str) -> None:
    recognizer = Counting()
    n = Nona([*Nona()._recognizers, recognizer], policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert result.text == masked
    assert recognizer.calls == calls
    # `CUSTOM_1PERSON_1`: a masked value glued to a placeholder stays masked in the reply (#87)
    expected = text if calls == 1 else masked
    assert n.restore(result.text, result.mapping) == expected


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        (
            "tel +216 22 123 456-sami.ksouri@example.tn-PERSON_1 ok",
            "tel PHONE_1-PERSON_2.ksouri@example.tn-PERSON_1 ok",
        ),
        ("x+216 22 123 456_12345678b.c@y.tnPERSON_1", "xPHONE_1_12345678b.c@y.tnPERSON_1"),
    ],
)
def test_the_second_pass_never_cuts_into_a_first_pass_span(text: str, masked: str) -> None:
    """Re-review of #92: the e-mail found once the placeholder is blanked overlapped the phone, won the
    resolver and left `+216 22 123` in clear. Output as on main."""
    assert Nona().anonymize(text).text == masked


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


def test_a_multi_word_name_after_a_tatweel_and_a_mark_round_trips() -> None:
    """A tatweel is a base character: the mark after it composes with nothing."""
    n = Nona(policies=ALL_REVERSIBLE)
    mapping = n.anonymize_messages(["Mme Sami Ksouri est la"]).mapping
    assert n.restore("ok PERSON_1ـ\u0301 x", mapping) == "ok Sami Ksouriـ\u0301 x"


@pytest.mark.parametrize("text", ["İBAN 1", "RİB 12", "LOCATİON_1", "PER\u017fON_1", "PHONE 1 ok"])
def test_a_placeholder_with_a_letter_regex_case_folding_takes_for_ascii(text: str) -> None:
    """`İ` matches `I` case-insensitively but `upper()` keeps it: no crash, in anonymize or restore."""
    n = Nona(policies=ALL_REVERSIBLE)
    result = n.anonymize(text)
    assert n.restore(text, n.anonymize_messages(["Mme Sami est la"]).mapping) is not None
    assert n.restore(result.text, result.mapping) == canonical_form(text, result)


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
        "PERSON_1\u0662\u0662\u0661\u0662\u0663\u0664\u0665\u0666",  # Arabic-Indic phone after one, #85
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
