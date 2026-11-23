from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core.canonical import (
    HONORIFICS,
    canonical_tokens,
    compatible,
    glued_parts,
    handle_matches,
    handle_names,
    is_honorific,
    name_words,
)


def toks(*words: str) -> frozenset[str]:
    return frozenset(words)


@pytest.mark.parametrize(
    "surface",
    ["M. Ben Salem", "ben salem", "BEN SALEM", "Ben-Salem", "Bensalem", "Mme Ben Salem", "Dr. ben  salem"],
)
def test_surname_forms_are_one_token(surface: str) -> None:
    assert canonical_tokens(surface) == toks("bensalem")


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Ahmed Ben Salem", toks("ahmed", "bensalem")),
        ("Ahmed", toks("ahmed")),
        ("Ahmad", toks("ahmed")),
        ("Mohammed Trabelsi", toks("mohamed", "trabelsi")),
        ("Med Trabelsi", toks("mohamed", "trabelsi")),
        ("Muhammad", toks("mohamed")),
        ("Meriem", toks("mariem")),
        ("Hélène Dupont", toks("helene", "dupont")),
        ("Mohamed El Abed", toks("mohamed", "elabed")),
        ("si Moncef", toks("monsef")),
        ("khalti Fatma", toks("fatma")),
        ("3amm Hédi", toks("hedi")),
        ("3ali", toks("ali")),
        ("m7ammed", toks("mhammed")),
        ("Sami O'Neil", toks("sami", "o", "neil")),
        ("سامي", toks("sami")),
        ("السيد أحمد بن سالم", toks("ahmed", "bensalem")),
        ("فاطمة", toks("fatma")),
        ("Ahmed Ben", toks("ahmed", "ben")),  # a trailing particle stays a token
        ("Ben", toks("ben")),
    ],
)
def test_canonical_tokens(surface: str, expected: frozenset[str]) -> None:
    assert canonical_tokens(surface) == expected


@pytest.mark.parametrize("surface", ["M.", "Mme", "Dr", "si", "السيد", "", "  "])
def test_honorific_alone_has_no_tokens(surface: str) -> None:
    assert canonical_tokens(surface) == frozenset()


def test_cross_script_and_spelling_variants_are_compatible() -> None:
    assert compatible(canonical_tokens("سامي"), canonical_tokens("Sami Trabelsi"))
    assert compatible(canonical_tokens("Mohammed"), canonical_tokens("Med Ben Salem"))


def test_compatible_is_subset_or_superset() -> None:
    a, ab, ac = toks("ahmed"), toks("ahmed", "bensalem"), toks("ahmed", "karoui")
    assert compatible(a, ab)
    assert compatible(ab, a)
    assert compatible(ab, ab)
    assert not compatible(ab, ac)
    assert not compatible(frozenset(), a)  # nothing to link on


@given(st.text(max_size=40))
def test_canonical_tokens_is_total_and_honorific_free(text: str) -> None:
    tokens = canonical_tokens(text)
    assert not tokens & HONORIFICS
    assert all(t and not t.isdigit() for t in tokens)


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz -.", max_size=30))
def test_case_and_honorific_prefix_dont_change_tokens(text: str) -> None:
    assert canonical_tokens(text.upper()) == canonical_tokens(text)
    assert canonical_tokens("M. " + text) == canonical_tokens(text)


# --------------------------------------------------------------------------- name words (look-ahead)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Chirine Boukadida", ["Chirine", "Boukadida"]),
        ("Mme Chirine Boukadida", ["Chirine", "Boukadida"]),  # the honorific names no one
        ("Ahmed Ben Salem", ["Ahmed"]),  # `Salem` alone is not `bensalem`, and `Ben` is a particle
        ("Ahmed Ben-Salem", ["Ahmed"]),
        ("SAMI TRABELSI", ["SAMI", "TRABELSI"]),
        ("amal trabelsi", []),  # lowercase Latin words are as likely ordinary words
        ("Sami", []),  # the whole mention is already a span
        ("Al Jaziri", []),  # `al` is a particle: the surname is `aljaziri`
        ("Mohamed Ali", ["Mohamed", "Ali"]),
        ("Ali Ali", ["Ali"]),
        ("سامي الطرابلسي", ["سامي", "الطرابلسي"]),  # no case in Arabic script
        ("Che\u0301rine Boukadida", ["Che\u0301rine", "Boukadida"]),  # NFD accent, as macOS may send it
        ("مُحَمَّد الطرابلسي", ["مُحَمَّد", "الطرابلسي"]),  # harakat
        ("Sami O'Brien", ["Sami", "O'Brien"]),  # an elision stays whole: no bare `Brien`
        ("Jean-Pierre Dupont", ["Jean", "Pierre", "Dupont"]),
        ("PERSON_1 Trabelsi", ["Trabelsi"]),  # a typed placeholder is not a name word
    ],
)
def test_name_words(surface: str, expected: list[str]) -> None:
    assert name_words(surface) == expected


@given(st.text(max_size=40))
def test_name_words_are_words_of_the_mention_inside_its_tokens(text: str) -> None:
    whole = canonical_tokens(text)
    for word in name_words(text):
        assert word in text
        assert canonical_tokens(word) <= whole
        assert len(word) >= 3


# --------------------------------------------------------------------------- handles (deep dive B, step 6)


@pytest.mark.parametrize(
    ("handle", "name"),
    [
        ("ahmed.bs", "Ahmed Ben Salem"),
        ("abensalem", "Ahmed Ben Salem"),
        ("a.bensalem", "Ahmed Ben Salem"),
        ("bensalem.ahmed", "Ahmed Ben Salem"),  # any order
        ("ahmed.salem", "Ahmed Ben Salem"),  # the surname without its particle
        ("ahmed_ben_salem", "Ahmed Ben Salem"),
        ("ahmed", "Ahmed Ben Salem"),  # one name part in full is enough
        ("med.bs", "Mohamed Ben Salem"),  # spelling variant
        ("sami.trabelsi.1990", "Sami Trabelsi"),  # a year is not a name part
        ("sami92", "Sami Trabelsi"),  # trailing digits aren't Arabizi letters
        ("3ali.t", "Ali Trabelsi"),  # Arabizi `3` before a letter is read
        ("@strabelsi", "Sami Trabelsi"),  # a social handle
        ("Sami.Trabelsi", "Sami Trabelsi"),
        ("wassim.ferchichi", "Wassim"),  # says more than the known name: allowed, it covers every token
        ("ahmed+devis", "Ahmed Ben Salem"),  # a plus tag is not part of the name
        ("92sami", "Sami Trabelsi"),  # a leading number is not Arabizi
        ("7amza.jlassi", "Hamza Jlassi"),  # a single leading Arabizi digit is a letter
    ],
)
def test_handle_matches(handle: str, name: str) -> None:
    assert handle_matches(handle, canonical_tokens(name), glued_parts(name))


@pytest.mark.parametrize(
    ("handle", "name"),
    [
        ("ab", "Ahmed Ben Salem"),  # initials only
        ("o.ayari", "Olfa"),  # only an initial of the known name
        ("contact", "Sami Trabelsi"),
        ("ahmedkaroui", "Ahmed Ben Salem"),  # `karoui` contradicts nothing known, but it's unseparated
        ("sami.jlassi", "Sami Trabelsi"),  # contradicts the known surname
        ("ahmed.ahmed", "Ahmed Ben Salem"),  # a token is used once
        ("samir", "Sami Trabelsi"),  # not `sami` + `r`: `r` is no one's initial here
        ("", "Sami Trabelsi"),
        ("1990", "Sami Trabelsi"),
        ("sami", "Mme"),  # nothing to match: an honorific alone has no tokens
        ("y.trabelsi", "Elyes Trabelsi"),  # `Elyes` only starts like the particle `el`: no `yes`, no `y`
        ("kadida", "Chirine Boukadida"),  # `Boukadida` is not `bou` + `kadida`
        ("jamin", "Benjamin Martin"),
    ],
)
def test_handle_does_not_match(handle: str, name: str) -> None:
    assert not handle_matches(handle, canonical_tokens(name), glued_parts(name))


def test_glued_parts_are_only_the_glued_tokens() -> None:
    assert glued_parts("Ahmed Ben Salem") == {"bensalem": ("ben", "salem")}
    assert glued_parts("Elyes Boukadida") == {}
    assert glued_parts("Ali el Abed") == {"elabed": ("el", "abed")}


def test_superset_needs_the_option_and_few_pieces() -> None:
    wassim = canonical_tokens("Wassim")
    assert not handle_matches("wassim.ferchichi", wassim, superset=False)
    assert handle_matches("wassim.ferchichi", wassim)
    assert not handle_matches(".".join(["wassim", *"bcdefgh"]), wassim)  # 8 pieces: no subset search


def test_a_long_handle_is_fast() -> None:
    import time

    start = time.perf_counter()
    assert not handle_matches(".".join("a" * 40), canonical_tokens("Sami"))
    assert not handle_matches("a" * 500, canonical_tokens("Amal Ben Ali"))
    assert time.perf_counter() - start < 0.5


def test_handle_names_finds_a_name_part_written_in_full() -> None:
    ahmed = canonical_tokens("Ahmed Karoui")
    assert handle_names("sami_ahmed", ahmed)
    assert not handle_names("sami.a", ahmed)  # an initial isn't enough


names = st.lists(
    st.sampled_from(["sami", "trabelsi", "ahmed", "bensalem", "nour", "jlassi", "karoui"]),
    min_size=1,
    max_size=3,
    unique=True,
)


@given(names, st.data())
def test_a_handle_written_from_a_persons_tokens_matches_them(tokens: list[str], data: st.DataObject) -> None:
    parts = [data.draw(st.sampled_from([t, t[0]])) for t in tokens]
    full = data.draw(st.integers(0, len(tokens) - 1))
    parts[full] = tokens[full]  # at least one part in full
    order = data.draw(st.permutations(range(len(parts))))
    separator = data.draw(st.sampled_from(["", ".", "_", "-"]))
    handle = separator.join(parts[i] for i in order)
    assert handle_matches(handle, frozenset(tokens))


def test_m_elision_and_abdel_glue_into_one_name() -> None:
    assert canonical_tokens("M'hamed") == {"mhamed"}  # not the honorific `M` + `hamed`
    assert canonical_tokens("M'barek") == canonical_tokens("Mbarek")
    assert not compatible(canonical_tokens("M'hamed"), canonical_tokens("Hamed"))
    assert (
        canonical_tokens("Abdel Kader") == canonical_tokens("Abd El Kader") == canonical_tokens("Abdelkader")
    )
    assert canonical_tokens("M. Ben Salem") == {"bensalem"}  # `M.` with a dot is still the honorific
    assert canonical_tokens("d'Ahmed") == {"d", "ahmed"}


def test_abdel_alone_is_a_first_name() -> None:
    assert canonical_tokens("Abdel Trabelsi") == {"abdel", "trabelsi"}
    assert compatible(canonical_tokens("Abdel Ben Salem"), canonical_tokens("M. Ben Salem"))
    assert canonical_tokens("Abdou Rahman") == {"abdourahman"}


# --------------------------------------------------------------------------- variants (#55)

VARIANT_PAIRS = [
    ("Syrine", "Cyrine"), ("Yassine", "Yacine"), ("Sofiene", "Sofien"), ("Jlassi", "Jelassi"),
    ("Labidi", "Laabidi"), ("Chedly", "Chedli"), ("Wassim", "Ouassim"), ("Qais", "Kais"), ("Mahdi", "Mehdi"),
    ("كريم", "Karim"), ("مهدي", "Mehdi"), ("ياسين", "Yassine"), ("خالد", "Khaled"), ("حمزة", "Hamza"),
    ("أيوب", "Ayoub"), ("أمل", "Amel"), ("الطرابلسي", "Trabelsi"), ("بن علي", "Ben Ali"),
]  # fmt: skip


@pytest.mark.parametrize(("a", "b"), VARIANT_PAIRS)
def test_spelling_variants_and_scripts_unify(a: str, b: str) -> None:
    assert canonical_tokens(a) == canonical_tokens(b)


def test_every_arabic_list_name_has_a_latin_form() -> None:
    from nona.core.recognizers.gazetteer_data import NAMES_AR

    assert [n for n in NAMES_AR if not all(t.isascii() for t in canonical_tokens(n))] == []


def test_merged_list_spellings_agree_on_being_ordinary_words() -> None:
    """Merged keys share the gazetteer's common-word status, so two listed spellings may only merge if the
    common-word list treats them alike: merging `Rim` (gazelle) with `Rym` would demote the name `Rym`."""
    import unicodedata

    from nona.core.canonical import VARIANTS
    from nona.core.recognizers import gazetteer_data as data

    def fold(word: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", word.casefold()) if not unicodedata.combining(c)
        )

    listed = {fold(w) for w in data.FIRST_NAMES_TN + data.FIRST_NAMES_FR_EN + data.SURNAMES_TN}
    common = {fold(w) for w in data.COMMON_WORDS}
    pairs = [(v, c) for v, c in VARIANTS.items() if v in listed and c in listed]
    assert pairs  # the check covers something
    assert [(v, c) for v, c in pairs if (v in common) != (c in common)] == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [("Mme", True), ("Dr.", True), ("M.", True), ("السيدة", True), ("khalti", True), ("MONSIEUR", True),
     ("Islem", False), ("Ben", False), ("12", False), ("بالسيدة", False)],
)  # fmt: skip
def test_is_honorific(word: str, expected: bool) -> None:
    """One written word. A glued clitic (`بالسيدة`) isn't recognized: the word is just not trimmed."""
    assert is_honorific(word) is expected
