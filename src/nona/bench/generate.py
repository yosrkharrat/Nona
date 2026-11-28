"""Template generator for the benchmark dataset (docs/design/05-dataset.md, sections 3 and 5).

Templates are written by hand. The generator fills their slots with synthetic values from a seeded RNG, so
the gold labels are exact by construction: the generator knows where it put every value. Output goes through
`parse_conversation`, so a generated conversation is held to the same rules as a hand-written one.

Usage. `bench/data/dev.jsonl` starts with the v0 set, the multi-turn set (19 templates three times each)
and the S4 set (35 templates three times each); `bench/data/test.jsonl` starts with the test set (61
templates five times each). Hand-written and LLM-drafted conversations follow in both files (05 §7). A test
pins every generated part:
    python -m nona.bench.generate --seed 20261120 --count 80 --prefix dev > bench/data/dev.jsonl
    python -m nona.bench.generate --set multi-turn --seed 2 --count 57 --prefix dev --start 81 \
        >> bench/data/dev.jsonl
    python -m nona.bench.generate --set dev-s4 --seed 4 --count 105 --prefix dev --start 138 \
        >> bench/data/dev.jsonl
    python -m nona.bench.generate --set test --seed 20261128 --count 305 --prefix test > bench/data/test.jsonl

Slot syntax inside a template turn: `{name}` or `{name.form}`. Each slot name is declared once per template
with a kind (see `KINDS`). Person slots have the forms `first`, `last` and `lower` (the first name in lower
case, as Arabizi writes it) besides the full name. A kind may name an owner, `phone:tn@p1`, which links the
value to person `p1` (deep dive B, step 6). A person kind may share another person's first name,
`person:tn2:m=p1`: two people called Ahmed. A mention written `{p1.first?}` is *ambiguous* (05 §3): even a
human can't tell which of the two is meant, so linking metrics skip it. Kinds starting with `~` are
distractors: they look like personal data but aren't, and they're recorded for over-masking analysis.

The `tn2` and `ar2` pools hold Tunisian names, in Latin and Arabic script, that are deliberately **not** in
the gazetteer (`nona.core.recognizers.gazetteer_data`), so the benchmark measures detection, not list overlap
(a test keeps it that way). `tn3` and `ar3` are held out the same way and used by the `test` split only, so
no name of the frozen split is seen while tuning on `dev`. Templates that use listed names (`tn`, `ar`) say
so in their name (`-listed`) or are in the v0 set, and the report's held-out rate leaves them out.

Determinism: the same seed and the same templates give the same output on the same Python minor version.
`random.Random` methods other than `random()` may change between Python versions, so the committed files,
not the seed, are the source of truth.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import phonenumbers

from nona.bench.dataset import KNOWN_TAGS, LANGUAGES, Conversation, dumps_jsonl, parse_conversation
from nona.core import EntityType
from nona.core.checksums import iban_check_digits

# --------------------------------------------------------------------------- seed lists

# Common first names and surnames, combined at random. Public figures are excluded twice: their surnames are
# kept out of the lists, and any full name in PUBLIC_FIGURES is re-drawn (05 §5).
FIRST_NAMES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "tn": {
        "m": ("Ahmed", "Sami", "Mohamed", "Yassine", "Amine", "Mehdi", "Karim", "Hamza", "Skander", "Wassim",
              "Oussama", "Bilel", "Aymen", "Anis"),
        "f": ("Ines", "Rim", "Salma", "Mariem", "Sarra", "Olfa", "Asma", "Nesrine", "Houda", "Emna", "Fatma",
              "Hajer", "Ghada", "Rania"),
    },
    "fr": {
        "m": ("Julien", "Thomas", "Nicolas", "Antoine", "Hugo", "Lucas", "Mathieu", "Pierre"),
        "f": ("Camille", "Claire", "Léa", "Sophie", "Chloé", "Manon", "Élodie", "Margaux"),
    },
    "en": {
        "m": ("James", "Daniel", "Ryan", "Ethan", "Oliver", "Samuel"),
        "f": ("Emily", "Olivia", "Grace", "Hannah", "Megan", "Lucy"),
    },
    # Held out from the gazetteer (see the module docstring).
    "tn2": {
        "m": ("Ghaith", "Mootez", "Achraf", "Nidhal", "Rayen", "Chiheb", "Hazem", "Wael", "Bacem", "Zakaria",
              "Oussema", "Montaser"),
        "f": ("Chaima", "Khouloud", "Ranim", "Mayssa", "Chirine", "Islem", "Ahlem", "Rihem", "Nadhra",
              "Molka", "Yasmine", "Asmahen"),
    },
    # Arabic script (#58): `ar` is in the gazetteer lists, `ar2` is held out like `tn2` (tests check both).
    "ar": {
        "m": ("أحمد", "محمد", "سامي", "يوسف", "ياسين", "هشام", "وليد", "طارق"),
        "f": ("فاطمة", "مريم", "ليلى", "سلمى", "زينب", "سارة", "منى", "نادية"),
    },
    "ar2": {
        "m": ("غيث", "معتز", "أشرف", "نضال", "ريان", "شهاب", "حازم", "زكرياء"),
        "f": ("شيماء", "خلود", "رنيم", "ميساء", "شيرين", "إسلام", "أحلام", "ملك"),
    },
    # Held out too, and used by the `test` split only (#74): the frozen split measures detection on names
    # that never appear in `dev`, so tuning on `dev` can't learn them.
    "tn3": {
        "m": ("Iyed", "Dhafer", "Ghazi", "Louay", "Mortadha", "Talel", "Seifeddine", "Sadri", "Amenallah",
              "Mehrez", "Rami"),
        "f": ("Tasnim", "Hadil", "Oumaima", "Ichraf", "Sondes", "Rabeb", "Arwa", "Nihel", "Soumaya",
              "Henda"),
    },
    "ar3": {
        "m": ("إياد", "ظافر", "غازي", "لؤي", "مرتضى", "طلال", "صدري", "محرز"),
        "f": ("تسنيم", "هديل", "أميمة", "سندس", "رباب", "أروى", "نهال", "سمية"),
    },
}  # fmt: skip
SURNAMES: Mapping[str, tuple[str, ...]] = {
    "tn": ("Ben Salem", "Trabelsi", "Gharbi", "Mejri", "Ayari", "Sassi", "Dridi", "Khelifi", "Chaabane",
           "Ben Amor", "Mansouri", "Jlassi", "Ferchichi", "Ouertani", "Belhadj", "Hamdi", "Zouari", "Ksouri",
           "Masmoudi", "Ben Romdhane"),
    "fr": ("Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Lefèvre", "Garnier", "Rousseau", "Fontaine",
           "Chevalier", "Girard", "Mercier"),
    "en": ("Carter", "Hughes", "Bennett", "Foster", "Walsh", "Porter", "Reed", "Hayes", "Turner", "Collins"),
    "tn2": ("Bouhlel", "Chtioui", "Zghal", "Ben Hmida", "Ghariani", "Drira", "Kammoun", "Haddar", "Belkhodja",
            "Sellami", "Lahmar", "Mnif", "Ben Fraj", "Hachicha", "Jaouadi", "Ghorbel", "Boukadida"),
    "ar": ("الطرابلسي", "الغربي", "الجلاصي", "المنصوري", "الخليفي", "الماجري"),
    "ar2": ("بوهلال", "الشتيوي", "الزغل", "دريرة", "كمون", "الحشيشة", "الغرياني", "المنيف"),
    "tn3": ("Chelbi", "Ghedira", "Mzoughi", "Feki", "Chakroun", "Frikha", "Elloumi", "Hentati", "Baklouti",
            "Dammak", "Ouni", "Khiari", "Sahnoun", "Ben Ayed", "Ben Jemaa", "Ben Hamouda"),
    "ar3": ("الشلبي", "غديرة", "المزوغي", "الفقي", "شقرون", "فريخة", "اللومي", "الهنتاتي", "البقلوطي",
            "الدماق"),
}  # fmt: skip
# Cities for LOCATION values, some of them outside the gazetteer's list.
CITIES: tuple[str, ...] = (
    "Sfax", "Sousse", "Bizerte", "Nabeul", "Monastir", "Kairouan", "La Marsa", "Menzel Temime", "El Jem",
    "Teboulba", "Sbeitla", "Tinja", "Ksour Essef", "Bekalta",
)  # fmt: skip
# The same in Arabic script (#74), as Tunisians write them.
CITIES_AR: tuple[str, ...] = (
    "صفاقس", "سوسة", "بنزرت", "نابل", "المنستير", "القيروان", "المرسى", "قليبية", "الحمامات", "توزر", "قابس",
    "المهدية", "باجة", "منزل تميم",
)  # fmt: skip
PUBLIC_FIGURES = frozenset(
    {
        "Kais Saied", "Leila Trabelsi", "Nabil Karoui", "Hamma Hammami", "Ons Jabeur", "Mohamed Bouazizi",
        "Rached Ghannouchi", "Moncef Marzouki", "Hichem Mechichi", "Youssef Chahed", "Mehdi Jomaa",
        "Najla Bouden", "Habib Bourguiba", "Abir Moussi", "Emmanuel Macron",
    }
)  # fmt: skip

EMAIL_DOMAINS = ("example.com", "example.org", "example.net")  # RFC 2606
IP_PREFIXES = ("192.0.2", "198.51.100", "203.0.113")  # RFC 5737 documentation ranges

# Published test card numbers (Stripe and Adyen test docs). All Luhn-valid, none can be charged.
TEST_CARDS = (
    "4242424242424242", "4000056655665556", "4111111111111111", "4012888888881881",
    "5555555555554444", "5200828282828210", "5105105105105100", "2223003122003222",
    "378282246310005", "371449635398431", "6011111111111117", "3056930009020004",
)  # fmt: skip


# --------------------------------------------------------------------------- filled values


@dataclass(frozen=True, slots=True)
class _Value:
    type: EntityType | None  # None for a distractor
    forms: Mapping[str, str]  # "" is the default form
    owner: str | None = None  # slot name of the owning person
    note: str = ""  # distractors only


def _fold(name: str) -> str:
    """Lower-case ASCII for e-mail local parts: `Léa Ben Salem` → `leabensalem`."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if c.isascii() and c.isalnum()).lower()


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _group(s: str, sizes: Sequence[int], sep: str) -> str:
    out: list[str] = []
    i = 0
    for size in sizes:
        out.append(s[i : i + size])
        i += size
    if i < len(s):
        out.append(s[i:])
    return sep.join(part for part in out if part)


def _valid_phone(national: str, region: str) -> bool:
    """Valid for its region per libphonenumber: no unassigned or overseas ranges (05 §5)."""
    try:
        return phonenumbers.is_valid_number(phonenumbers.parse(national, region))
    except phonenumbers.NumberParseException:
        return False


# Tunisian numbering plan, as libphonenumber encodes it (deep dive A): 8 digits, no trunk 0.
# Mobile: 2x, 5x, 9x, 40-48 and a few 3x ranges. Fixed: 7x and 30-32. Fax lines are ordinary fixed lines.
_TN_TYPES = {phonenumbers.PhoneNumberType.MOBILE: "mobile", phonenumbers.PhoneNumberType.FIXED_LINE: "fixed"}
_TN_FIRST_DIGITS = {"mobile": "2222555599994443", "fixed": "77777773"}


def _tn_national(rng: random.Random, kind: str) -> str:
    """A valid 8-digit Tunisian number of the given kind, drawn from the whole numbering plan."""
    while True:
        national = rng.choice(_TN_FIRST_DIGITS[kind]) + _digits(rng, 7)
        try:
            number = phonenumbers.parse(national, "TN")
        except phonenumbers.NumberParseException:  # pragma: no cover - 8 digits always parse
            continue
        if phonenumbers.is_valid_number(number) and _TN_TYPES.get(phonenumbers.number_type(number)) == kind:
            return national


def _format_tn(rng: random.Random, national: str) -> str:
    """The ways Tunisian numbers are actually written, the official `XX XXX XXX` included."""
    spaced = _group(national, (2, 3, 3), " ")
    return rng.choice(
        (
            spaced,
            _group(national, (2, 3, 3), "."),
            _group(national, (2, 3, 3), "-"),
            _group(national, (2, 2, 2, 2), " "),  # in pairs
            national,
            f"+216 {spaced}",
            f"(+216) {spaced}",
            f"00216 {spaced}",
            f"00 216 {spaced}",
            f"216 {spaced}",
            f"+216{national}",
            f"+216 (0){spaced}",  # the French "(0)" habit: wrong for Tunisia, but written
        )
    )


def _phone_tn(rng: random.Random) -> str:
    return _format_tn(rng, _tn_national(rng, "mobile" if rng.random() < 0.75 else "fixed"))


def _phone_tn_fixed(rng: random.Random) -> str:
    """Office lines and fax numbers (a fax is a fixed line)."""
    return _format_tn(rng, _tn_national(rng, "fixed"))


def _service_tn(rng: random.Random) -> str:
    """Business service numbers: toll-free 8010, shared cost 8110/8210, premium 88. Not personal data."""
    national = rng.choice(("8010", "8110", "8210", "88" + _digits(rng, 2))) + _digits(rng, 4)
    return _group(national, (2, 3, 3), " ")


def _phone_fr(rng: random.Random) -> str:
    while True:
        national = "0" + rng.choice("67") + _digits(rng, 8)
        if _valid_phone(national, "FR"):  # rejects overseas ranges like 06 94 (French Guiana, +594)
            break
    pairs = _group(national, (2, 2, 2, 2, 2), " ")
    dotted = _group(national, (2, 2, 2, 2, 2), ".")
    international = "+33 " + _group(national[1:], (1, 2, 2, 2, 2), " ")
    return rng.choice((pairs, dotted, national, international))


def _tn_rib(rng: random.Random) -> str:
    body = _digits(rng, 18)  # bank (2), branch (3), account (13)
    key = 97 - (int(body) * 100) % 97
    return body + f"{key:02d}"


def _rib(rng: random.Random) -> str:
    rib = _tn_rib(rng)
    return rng.choice((rib, _group(rib, (2, 3, 13, 2), " ")))


def _iban_tn(rng: random.Random) -> str:
    rib = _tn_rib(rng)
    iban = "TN" + iban_check_digits("TN", rib) + rib  # always TN59, because the RIB is divisible by 97
    return rng.choice((iban, _group(iban, (4,) * 6, " ")))


def _iban_fr(rng: random.Random) -> str:
    bank, branch, account = _digits(rng, 5), _digits(rng, 5), _digits(rng, 11)
    key = 97 - (89 * int(bank) + 15 * int(branch) + 3 * int(account)) % 97  # French RIB key
    bban = f"{bank}{branch}{account}{key:02d}"
    iban = "FR" + iban_check_digits("FR", bban) + bban
    return rng.choice((iban, _group(iban, (4,) * 7, " ")))


def _card(rng: random.Random) -> str:
    number = rng.choice(TEST_CARDS)
    sizes = (4, 6, 5) if len(number) == 15 else (4, 4, 4, 4)
    return rng.choice((number, _group(number, sizes, " "), _group(number, sizes, "-")))


def _cin(rng: random.Random) -> str:
    return rng.choice("01") + _digits(rng, 7)


_EASTERN = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _eastern(make: Callable[[random.Random], str]) -> Callable[[random.Random], str]:
    """The same value written with Eastern Arabic digits (٢٢ ١٢٣ ٤٥٦), as some Arabic-script messages do."""
    return lambda rng: make(rng).translate(_EASTERN)


def _ip(rng: random.Random) -> str:
    return f"{rng.choice(IP_PREFIXES)}.{rng.randint(1, 254)}"


def _order(rng: random.Random) -> str:
    return rng.choice("23456789") + _digits(rng, 7)  # 8 digits, like a CIN, but an order number


def _price(rng: random.Random) -> str:
    return f"{rng.randint(2, 950)},{rng.randint(0, 99) * 10:03d}"


def _ref(rng: random.Random) -> str:
    return f"{rng.choice(('REF', 'CMD', 'INV'))}-{rng.randint(10000, 99999)}"


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2023, 2026)}"


_STRUCTURED: Mapping[str, tuple[EntityType, Callable[[random.Random], str]]] = {
    "phone:tn": (EntityType.PHONE, _phone_tn),
    "phone:tn-fixed": (EntityType.PHONE, _phone_tn_fixed),
    "phone:fr": (EntityType.PHONE, _phone_fr),
    "phone:tn-east": (EntityType.PHONE, _eastern(_phone_tn)),
    "iban:tn": (EntityType.IBAN, _iban_tn),
    "iban:fr": (EntityType.IBAN, _iban_fr),
    "rib": (EntityType.RIB, _rib),
    "card": (EntityType.CARD, _card),
    "cin": (EntityType.NATIONAL_ID, _cin),
    "cin:east": (EntityType.NATIONAL_ID, _eastern(_cin)),
    "ip": (EntityType.IP_ADDRESS, _ip),
}
_DISTRACTORS: Mapping[str, tuple[str, Callable[[random.Random], str]]] = {
    "~order": ("order number, not an ID", _order),
    "~service": ("business service number (toll-free, shared cost or premium), not personal", _service_tn),
    "~price": ("price", _price),
    "~ref": ("reference number", _ref),
    "~date": ("date", _date),
    # Ordinary words that are also names (deep dive A): hope, light, high, year, doing.
    "~word:amal": ("ordinary word that is also a name (hope)", lambda rng: "amal"),
    "~word:nour": ("ordinary word that is also a name (light)", lambda rng: "nour"),
    "~word:3ali": ("ordinary word that is also a name (high)", lambda rng: "3ali"),
    "~word:sana": ("ordinary word that is also a name (year)", lambda rng: "sana"),
    "~word:3amel": ("ordinary word that is also a name (doing)", lambda rng: "3amel"),
    # Over-masking controls (#58): words that fold to a name, Darija expressions, a brand after a cue, and
    # Arabic expressions holding a name-like word.
    "~word:marie": ("French word that folds to a name (married)", lambda rng: "marié"),
    "~word:claire": ("French word that is also a name (clear)", lambda rng: "claire"),
    "~word:pierre": ("French word that is also a name (stone)", lambda rng: "pierre"),
    "~word:grace": ("French word that folds to a name (thanks to)", lambda rng: "Grâce"),
    "~word:mabrouk": ("Darija expression that is also a name (congratulations)", lambda rng: "Mabrouk"),
    "~word:sahbi": ("Darija word that is also a name (my friend)", lambda rng: "sahbi"),
    "~word:saber": ("Darija word that is also a name (patient)", lambda rng: "saber"),
    "~brand:ooredoo": ("brand after a Darija cue word (m3a)", lambda rng: "Ooredoo"),
    "~word:karim-ar": ("Arabic word that is also a name (generous, in Ramadan kareem)", lambda rng: "كريم"),
    "~word:amine-ar": ("Arabic word that is also a name (amen)", lambda rng: "آمين"),
    "~word:amal-ar": ("Arabic word that is also a name (hope)", lambda rng: "أمل"),
    # #74: the common words of #50 used as words, in fixed expressions and after the cues' look-alikes.
    "~price:east": ("price in Eastern Arabic digits", _eastern(_price)),
    "~word:karim": ("Darija word that is also a name (generous, in ramdhan karim)", lambda rng: "karim"),
    "~word:amine": ("Darija word that is also a name (amen)", lambda rng: "amine"),
    "~word:farah": ("Darija word that is also a name (joy)", lambda rng: "farah"),
    "~word:chams": ("Darija word that is also a name (sun)", lambda rng: "chams"),
    "~word:kmar": ("Darija word that is also a name (moon)", lambda rng: "kmar"),
    "~word:sana-ar": ("Arabic word that is also a name (year)", lambda rng: "سنة"),
    "~word:mabrouk-ar": ("Arabic word that is also a name (congratulations)", lambda rng: "مبروك"),
    "~word:rahma-ar": ("Arabic word that is also a name (mercy)", lambda rng: "رحمة"),
    "~word:nour-ar": ("Arabic word that is also a name (light, in nour 3iniya)", lambda rng: "نور"),
    "~word:jari-ar": ("Arabic word that is also a kin cue (current, of an account)", lambda rng: "جاري"),
    "~word:farah-ar": ("Arabic word that is also a name (joy)", lambda rng: "فرح"),
    "~word:hayet-ar": ("Arabic word that is also a name (life)", lambda rng: "حياة"),
    "~word:chams-ar": ("Arabic word that is also a name (sun)", lambda rng: "شمس"),
    "~word:olivier": ("French word that is also a name (olive tree)", lambda rng: "olivier"),
    "~word:rose": ("French word that is also a name (rose)", lambda rng: "rose"),
    "~word:grace-en": ("English word that is also a name (grace period)", lambda rng: "grace"),
    "~word:jack-en": ("English word that is also a name (car jack)", lambda rng: "jack"),
    "~word:bill-en": ("English word that is also a name (invoice)", lambda rng: "bill"),
}
_PERSON = re.compile(r"person:(tn3|tn2|tn|fr|en|ar3|ar2|ar)(?::([mf]))?(?:=(\w+))?")
KINDS = (
    frozenset(_STRUCTURED)
    | frozenset(_DISTRACTORS)
    | {"email", "city", "city:ar"}
    | {f"person:{pool}{g}" for pool in FIRST_NAMES for g in ("", ":m", ":f")}
)
PERSON_FORMS = frozenset({"first", "last", "lower"})


# --------------------------------------------------------------------------- templates


@dataclass(frozen=True, slots=True)
class Template:
    name: str
    languages: tuple[str, ...]
    turns: tuple[tuple[str, str], ...]  # (role, text with {slots})
    slots: Mapping[str, str] = field(default_factory=dict[str, str])
    tags: tuple[str, ...] = ()  # besides the ones derived automatically (see `derived_tags`)

    def derived_tags(self) -> tuple[str, ...]:
        tags = list(self.tags)
        if "aeb-Latn" in self.languages:
            tags.append("arabizi")
        if len(self.languages) > 1:
            tags.append("code-switching")
        if len(self.turns) > 1:
            tags.append("multi-turn")
        if not any(not kind.startswith("~") for kind in self.slots.values()):
            tags.append("no-pii")
        return tuple(dict.fromkeys(tags))


_SLOT = re.compile(r"\{(\w+)(?:\.(\w+))?(\?)?\}")


def _split_kind(kind: str) -> tuple[str, str | None]:
    """`phone:tn@p1` → (`phone:tn`, `p1`). A person's `=p1` (shared first name) is removed from the base."""
    base, _, owner = kind.partition("@")
    if base.startswith("person:"):
        base = base.partition("=")[0]
    return base, owner or None


def _shares(kind: str) -> str | None:
    """`person:tn2:m=p1` → `p1`: the slot whose first name this person shares."""
    match = _PERSON.fullmatch(kind)
    return match.group(3) if match else None


def check_template(t: Template) -> None:
    """Raise ValueError if the template can't produce a valid conversation."""
    if not t.languages or any(lang not in LANGUAGES for lang in t.languages):
        raise ValueError(f"{t.name}: bad languages {t.languages}")
    unknown_tags = set(t.derived_tags()) - KNOWN_TAGS
    if unknown_tags:
        raise ValueError(f"{t.name}: unknown tags {sorted(unknown_tags)}")
    used: set[str] = set()
    for role, text in t.turns:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{t.name}: bad role {role!r}")
        for m in _SLOT.finditer(text):
            slot, form, ambiguous = m.group(1), m.group(2), m.group(3)
            if slot not in t.slots:
                raise ValueError(f"{t.name}: slot {{{slot}}} is not declared")
            base, _ = _split_kind(t.slots[slot])
            if form is not None and (not base.startswith("person:") or form not in PERSON_FORMS):
                raise ValueError(f"{t.name}: slot {{{slot}.{form}}}: unknown form for {base}")
            if ambiguous and not base.startswith("person:"):
                raise ValueError(f"{t.name}: slot {{{slot}?}}: only a person mention can be ambiguous")
            used.add(slot)
        if "{" in _SLOT.sub("", text) or "}" in _SLOT.sub("", text):
            raise ValueError(f"{t.name}: malformed slot in {text!r}")
    if used != set(t.slots):
        raise ValueError(f"{t.name}: declared but unused slots {sorted(set(t.slots) - used)}")
    for slot, kind in t.slots.items():
        base, owner = _split_kind(kind)
        if base not in KINDS:
            raise ValueError(f"{t.name}: slot {slot}: unknown kind {base!r}")
        if owner is not None:
            if base.startswith(("person:", "~")):
                raise ValueError(f"{t.name}: slot {slot}: {base} can't have an owner")
            if not t.slots.get(owner, "").startswith("person:"):
                raise ValueError(f"{t.name}: slot {slot}: owner {owner!r} is not a person slot")
        shared = _shares(kind)
        if shared is not None:
            target = t.slots.get(shared, "")
            if shared == slot or not target.startswith("person:"):
                raise ValueError(
                    f"{t.name}: slot {slot}: shares a first name with {shared!r}, not a person slot"
                )
            if _shares(target) is not None:
                raise ValueError(
                    f"{t.name}: slot {slot}: shares with {shared!r}, which shares too (no chains)"
                )
            mine, theirs = _PERSON.fullmatch(kind), _PERSON.fullmatch(target)
            if mine and theirs and mine.group(2) and theirs.group(2) != mine.group(2):
                raise ValueError(f"{t.name}: slot {slot}: shares a first name across genders with {shared!r}")


def _tpl(
    name: str, languages: str, slots: Mapping[str, str], *turns: str | tuple[str, str], tags: str = ""
) -> Template:
    return Template(
        name=name,
        languages=tuple(languages.split("+")),
        turns=tuple(("user", t) if isinstance(t, str) else t for t in turns),
        slots=slots,
        tags=tuple(tags.split()),
    )


A = "assistant"

# v0 templates (build plan S1): mostly structured identifiers, some names, and no-PII controls.
# Arabic script and Darija come in S4 with the full dataset.
TEMPLATES: tuple[Template, ...] = (
    # ---- French
    _tpl("fr-intro-phone", "fr", {"p1": "person:tn", "ph": "phone:tn@p1"},
         "Bonjour, je m'appelle {p1}, vous pouvez me joindre au {ph}."),
    _tpl("fr-callback", "fr", {"p1": "person:fr", "ph": "phone:fr@p1"},
         "Rappelle {p1.first} au {ph} avant 18h, c'est pour le devis."),
    _tpl("fr-iban-refund", "fr", {"ib": "iban:fr"},
         "Pour le remboursement, voici mon IBAN : {ib}. Merci d'avance !"),
    _tpl("fr-rib-transfer", "fr", {"p1": "person:tn", "rib": "rib@p1"},
         "Tu peux faire le virement sur ce RIB : {rib} ? C'est le compte de {p1}."),
    _tpl("fr-email-cc", "fr", {"p1": "person:tn", "em": "email@p1"},
         "Envoie le compte rendu à {em} et mets {p1.first} en copie."),
    _tpl("fr-card-dispute", "fr", {"cd": "card", "d": "~date"},
         "Ma carte {cd} a été débitée deux fois le {d}, comment contester ?"),
    _tpl("fr-cin-form", "fr", {"p1": "person:tn", "cin": "cin"},
         "Pour compléter le dossier, je m'appelle {p1} et mon numéro de CIN est le {cin}."),
    _tpl("fr-server-ip", "fr", {"ip": "ip"},
         "Le serveur {ip} ne répond plus depuis ce matin, tu peux vérifier ?"),
    _tpl("fr-doctor", "fr", {"p1": "person:tn"},
         "Dr {p1.last} a confirmé le rendez-vous de demain matin.", tags="honorific"),
    _tpl("fr-two-people", "fr", {"p1": "person:tn", "p2": "person:fr", "ph": "phone:tn@p2"},
         "{p1.first} et {p2.first} arrivent demain ; {p2.first} a laissé son numéro : {ph}."),
    _tpl("fr-appointment", "fr", {"p1": "person:tn", "em": "email@p1", "ph": "phone:tn@p1"},
         "J'ai un rendez-vous avec {p1} jeudi à 10h.",
         (A, "Très bien. Voulez-vous que je prépare un message de confirmation pour {p1.first} ?"),
         "Oui, envoie-le à {em}, et si besoin son portable c'est le {ph}."),
    _tpl("fr-order-late", "fr", {"o": "~order", "pr": "~price"},
         "Ma commande n° {o} n'est toujours pas arrivée, elle coûtait {pr} DT."),
    _tpl("fr-signature", "fr", {"p1": "person:tn", "tel": "phone:tn-fixed@p1", "fax": "phone:tn-fixed@p1"},
         "Cordialement,\n{p1}\nTél : {tel}\nFax : {fax}"),
    _tpl("fr-service", "fr", {"sv": "~service"},
         "Pour toute réclamation, appelez le service client au {sv}, du lundi au samedi."),
    _tpl("fr-summary", "fr", {},
         "Peux-tu résumer ce texte en trois points et proposer un titre accrocheur ?"),
    # ---- English
    _tpl("en-intro-phone", "en", {"p1": "person:en", "ph": "phone:tn@p1"},
         "Hi, this is {p1}. Please call me back on {ph} when you get a chance."),
    _tpl("en-wire", "en", {"p1": "person:tn", "ib": "iban:tn@p1"},
         "Can you send the deposit to IBAN {ib}? The account holder is {p1}."),
    _tpl("en-card-refund", "en", {"cd": "card", "r": "~ref"},
         "I was charged twice on my card {cd} for order {r}. Can you refund one of them?"),
    _tpl("en-forward", "en", {"em": "email"},
         "Forward the signed contract to {em}, they're expecting it today."),
    _tpl("en-failed-logins", "en", {"ip": "ip", "ip2": "ip"},
         "Our logs show repeated failed logins from {ip} and {ip2} since midnight."),
    _tpl("en-support", "en", {"p1": "person:en", "em": "email@p1", "ph": "phone:fr@p1"},
         "My name is {p1} and I can't log into my account.",
         (A, "Sorry to hear that, {p1.first}. Could you confirm the email address on the account?"),
         "It's {em}. My phone number is {ph} if that helps.",
         (A, "Thanks {p1.first}, I've sent a reset link to {em}."),
         "Got it, thanks!"),
    _tpl("en-office", "en", {"p1": "person:tn", "tel": "phone:tn-fixed@p1", "fax": "phone:tn-fixed@p1"},
         "You can reach {p1.first} at the office on {tel}, or send the forms by fax to {fax}."),
    _tpl("en-decline", "en", {},
         "Write a short, polite reply declining the meeting invitation for Friday."),
    _tpl("en-invoice", "en", {"pr": "~price", "r": "~ref"},
         "The invoice total is {pr} TND and the reference is {r}; please add it to the report."),
    # ---- Tunisian Arabizi
    _tpl("aeb-intro-phone", "aeb-Latn", {"p1": "person:tn", "ph": "phone:tn@p1"},
         "3aslema, ena {p1} w hedha nomrou mte3i {ph}."),
    _tpl("aeb-callback", "aeb-Latn", {"p1": "person:tn:m", "ph": "phone:tn@p1"},
         "Kallem {p1.first} 3al {ph}, rahou yestanna fik mel sbe7."),
    _tpl("aeb-rib", "aeb-Latn", {"rib": "rib"},
         "Ab3athli el flous 3al RIB hedha: {rib}, w ba3d 9olli."),
    _tpl("aeb-email", "aeb-Latn", {"p1": "person:tn:m", "em": "email@p1"},
         "Chnowa el mail mta3 {p1.first}?",
         (A, "Ma 3andich el mail mte3ou."),
         "Houwa {em}, ab3athlou el fichier."),
    _tpl("aeb-si", "aeb-Latn", {"p1": "person:tn:m"},
         "Si {p1.first} 9alli bech yfout 3lina ba3d el 3asr.", tags="honorific"),
    _tpl("aeb-cin", "aeb-Latn", {"cin": "cin"},
         "El CIN mte3i {cin}, 7achti bih bech nkammel el inscription."),
    _tpl("aeb-beach", "aeb-Latn", {},
         "Ya5i chnowa l a5bar? Ghodwa n7ebou nemchiw lel b7ar ken el ta9s mezyen."),
    _tpl("aeb-service", "aeb-Latn", {"sv": "~service"},
         "Kallem el service client 3al {sv}, ykhdmou 24/24."),
    _tpl("aeb-price", "aeb-Latn", {"pr": "~price"},
         "Soum el kilo wsel {pr} dinar, ghali barcha!"),
    # ---- Arabizi mixed with French
    _tpl("mix-transfer", "aeb-Latn+fr", {"ib": "iban:tn"},
         "Rani 3malt le virement 3al IBAN {ib}, confirme-moi ki ywsel."),
    _tpl("mix-appointment", "aeb-Latn+fr", {"p1": "person:tn:m", "ph": "phone:tn@p1"},
         "3andi rendez-vous m3a {p1} ghodwa à 15h.",
         (A, "D'accord, voulez-vous que je rappelle le rendez-vous à {p1.first} ?"),
         "Ey, nomrou {ph}, ab3athlou SMS."),
    _tpl("mix-card", "aeb-Latn+fr", {"cd": "card"},
         "Ma carte {cd} tbloquat fel distributeur, chnowa na3mel ?"),
)  # fmt: skip


# Multi-turn set (S3, ADR-0004): 4+ turns, for linking. Assistant turns hold real names, as a client re-sends
# restored answers. Most people come from the held-out `tn2` pool.
MULTI_TURN_TEMPLATES: tuple[Template, ...] = (
    _tpl("mt-fr-short-then-full", "fr", {"p1": "person:tn2:m", "ph": "phone:tn@p1"},
         "Rappelle {p1.first} demain matin, c'est urgent.",
         (A, "D'accord, je rappellerai {p1.first} demain matin. Vous avez son numéro ?"),
         "Oui, c'est {p1}, au {ph}.",
         (A, "C'est noté : {p1}, au {ph}."),
         "Et dis à {p1.first} que la réunion est déplacée à 15h."),
    _tpl("mt-fr-two-same-first-name", "fr",
         {"p1": "person:tn2:m", "p2": "person:tn2:m=p1", "em": "email@p1"},
         "{p1} et {p2} viennent tous les deux à la réunion de jeudi.",
         (A, "Très bien, je préviens {p1} et {p2}."),
         "{p1.first?} m'a dit qu'il serait en retard.",
         (A, "Vous parlez de {p1} ou de {p2} ?"),
         "De {p1}. Son mail, c'est {em}.",
         tags="ambiguous-reference"),
    _tpl("mt-en-doctor", "en", {"p1": "person:tn2", "ph": "phone:tn@p1"},
         "Please book a meeting with Dr {p1.last} next week.",
         (A, "Sure. Which day suits Dr {p1.last}?"),
         "Tuesday. {p1.first} said mornings are better, the number is {ph}.",
         (A, "Booked {p1} for Tuesday morning; I'll confirm on {ph}."),
         tags="honorific"),
    _tpl("mt-aeb-lowercase", "aeb-Latn",
         {"p1": "person:tn2:m", "ph": "phone:tn@p1", "em": "email@p1", "w": "~word:amal"},
         "kallem {p1.lower} 9bal el 3asr, 3andi {w} yji ghodwa.",
         (A, "Behi. Chnowa nomrou {p1.first} ?"),
         "{ph}, w ken ma jewbekch ab3ath l {em}.",
         (A, "Ok, ab3athtlou 3al {em}."),
         "y3aychek, {p1.lower} rajel behi.",
         tags="common-word-name"),
    _tpl("mt-fr-moved-city", "fr", {"p1": "person:tn2:m", "c1": "city", "rib": "rib@p1"},
         "{p1} a déménagé à {c1}, il faut mettre à jour son dossier.",
         (A, "D'accord. Quelle est la nouvelle adresse de {p1.first} à {c1} ?"),
         "Il t'envoie l'adresse par mail. Et voici son nouveau RIB : {rib}.",
         (A, "Le dossier de {p1} est à jour, avec le RIB {rib}."),
         "Merci. Préviens {p1.first} quand c'est fait."),
    _tpl("mt-en-colleague", "en",
         {"p1": "person:tn2", "p2": "person:tn2:f", "em": "email@p2", "ph": "phone:fr@p1"},
         "Hi, I'm {p1}. I'm writing about my colleague {p2}.",
         (A, "Hello {p1.first}, how can I help with {p2.first}'s account?"),
         "{p2.first} can't log in. Her email is {em}.",
         (A, "I've sent {p2.first} a reset link at {em}."),
         "Thanks. If anything else comes up, call me on {ph}."),
    _tpl("mt-mix-two-same-first-name", "aeb-Latn+fr",
         {"p1": "person:tn2:m", "p2": "person:tn2:m=p1", "ph": "phone:tn@p1"},
         "3andi rendez-vous m3a {p1} w {p2} lundi.",
         (A, "D'accord, je note {p1} et {p2} pour lundi."),
         "{p1.first?} 9alli bech yjib el contrat.",
         (A, "Lequel, {p1} ou {p2} ?"),
         "{p1}, nomrou {ph}.",
         tags="ambiguous-reference"),
    # The same with names from the gazetteer's lists: linking on names that are detected, reported apart.
    _tpl("mt-mix-two-same-first-name-listed", "aeb-Latn+fr",
         {"p1": "person:tn:m", "p2": "person:tn:m=p1", "ph": "phone:tn@p1"},
         "3andi rendez-vous m3a {p1} w {p2} lundi.",
         (A, "D'accord, je note {p1} et {p2} pour lundi."),
         "{p1.first?} 9alli bech yjib el contrat.",
         (A, "Lequel, {p1} ou {p2} ?"),
         "{p1}, nomrou {ph}.",
         tags="ambiguous-reference"),
    # Absorption with the short form detected (#58): a strong cue on a held-out first name, and a listed one.
    _tpl("mt-fr-cue-then-full", "fr", {"p1": "person:tn2:f", "ph": "phone:tn@p1"},
         "Mme {p1.first} a appelé ce matin.",
         (A, "Que voulait Mme {p1.first} ?"),
         "{p1} veut déplacer son rendez-vous. Son numéro est le {ph}.",
         (A, "Je rappelle {p1} au {ph}."),
         "Merci. Dis à {p1.first} que jeudi convient.",
         tags="honorific"),
    _tpl("mt-fr-listed-short-then-full", "fr", {"p1": "person:tn:m", "ph": "phone:tn@p1"},
         "Rappelle {p1.first} demain matin, c'est urgent.",
         (A, "D'accord, je rappellerai {p1.first} demain matin. Vous avez son numéro ?"),
         "Oui, c'est {p1}, au {ph}.",
         (A, "C'est noté : {p1}, au {ph}."),
         "Et dis à {p1.first} que la réunion est déplacée à 15h."),
    _tpl("mt-fr-honorific-surname", "fr", {"p1": "person:tn2:f", "ph": "phone:tn-fixed@p1", "c1": "city"},
         "Mme {p1.last} attend une réponse pour son dossier.",
         (A, "Je m'en occupe. Pouvez-vous me rappeler son prénom ?"),
         "{p1.first}. {p1} travaille à notre agence de {c1}, ligne directe {ph}.",
         (A, "Merci, j'appelle Mme {p1.last} au {ph}."),
         tags="honorific"),
    _tpl("mt-aeb-no-pii", "aeb-Latn",
         {"w1": "~word:amal", "w2": "~word:sana", "w3": "~word:3amel", "w4": "~word:3ali",
          "w5": "~word:nour"},
         "sbe7 el khir, 3andi {w1} kbir fel projet hedha.",
         (A, "Behi, chnowa el moshkla ?"),
         "el {w2} hedhi el as3ar tal3et barcha, w el kera {w4} zeda.",
         (A, "Fhemtek. Nchoufou el budget m3a b3adhna ?"),
         "ey, chnowa {w3} ena ghodwa ? w ma fammech {w5} fel bureau, nbadlou el lampa.",
         tags="common-word-name"),
    _tpl("mt-en-no-pii", "en", {},
         "Hello World is the first program I wrote. Can you explain it?",
         (A, "Sure. It prints a greeting, which shows the program runs."),
         "Merci! And what does Main do in Java?",
         (A, "It's the entry point: the method the runtime calls first."),
         "Great, Thanks for the help."),
    # Over-masking controls (#58): no personal data, only words that look like names.
    _tpl("mt-fr-overmask-control", "fr",
         {"w1": "~word:marie", "w2": "~word:claire", "w3": "~word:pierre", "w4": "~word:grace"},
         "Je suis {w1} depuis deux ans et l'explication du notaire était {w2}.",
         (A, "Félicitations ! Que puis-je faire pour vous ?"),
         "On veut poser une {w3} gravée dans le jardin. C'est autorisé ?",
         (A, "En général oui, mais vérifiez le règlement de votre commune."),
         "{w4} à vous, c'est plus clair."),
    _tpl("mt-aeb-overmask-control", "aeb-Latn",
         {"w1": "~word:mabrouk", "w2": "~word:sahbi", "w3": "~brand:ooredoo", "w4": "~word:saber"},
         "{w1} 3lik el bac ya {w2} !",
         (A, "Merci, rabbi y5allik."),
         "taw nemchi m3a {w3} nchri puce jdida.",
         (A, "Behi, ken t7eb n3awnek fel offre."),
         "la, lazem {w4} chwaya, el offre tetbadel ghodwa."),
    # Arabic script (#58): clitics glued to names (لـ, وـ), an honorific, listed and held-out names.
    _tpl("mt-aeb-arab-short-then-full", "aeb-Arab", {"p1": "person:ar2:m", "ph": "phone:tn@p1"},
         "كلّم {p1.first} غدوة الصباح، الحكاية مستعجلة.",
         (A, "باهي، شنوّة نومرو {p1.first}؟"),
         "{p1}، النومرو متاعو {ph}.",
         (A, "تسجّل: {p1}، {ph}."),
         "وقلّو ل{p1.first} الاجتماع تأخّر للثلاثة.",
         tags="clitic"),
    _tpl("mt-ar-honorific", "ar", {"p1": "person:ar2:f", "ph": "phone:tn-fixed@p1"},
         "السيدة {p1.last} تنتظر ردًّا على ملفها.",
         (A, "سأتابع الأمر. ما اسمها الأول؟"),
         "{p1.first}. {p1} تعمل في فرعنا، ورقمها المباشر {ph}.",
         (A, "شكرًا، سأتصل بالسيدة {p1.last} على {ph}."),
         tags="honorific"),
    _tpl("mt-aeb-arab-listed", "aeb-Arab", {"p1": "person:ar:m", "ph": "phone:tn@p1"},
         "عندي موعد مع {p1} نهار الاثنين.",
         (A, "باهي، نسجّل موعد {p1.first}."),
         "و{p1.first} قالي باش يجيب العقد، النومرو متاعو {ph}.",
         (A, "تسجّل {p1} على {ph}."),
         tags="clitic"),
    _tpl("mt-ar-no-pii", "ar",
         {"w1": "~word:karim-ar", "w2": "~word:amine-ar", "w3": "~word:amal-ar"},
         "رمضان {w1} على الجميع!",
         (A, "وعليكم، كل عام وأنتم بخير."),
         "اللهم {w2}. عندي {w3} كبير في هذا المشروع.",
         (A, "إن شاء الله يتحقق ما تتمناه."),
         tags="common-word-name"),
)  # fmt: skip

# S4 set for `dev` (#74): Arabic script, Darija and code-switching, to reach the 05 §7 balance. People come
# from the held-out `tn2` and `ar2` pools, except in `-listed` templates. Some numbers use Eastern Arabic
# digits.
DEV_S4_TEMPLATES: tuple[Template, ...] = (
    # ---- Tunisian Darija in Arabic script
    _tpl("s4-aeb-arab-intro-phone", "aeb-Arab", {"p1": "person:ar2", "ph": "phone:tn@p1"},
         "عسلامة، أنا {p1} وهذا النومرو متاعي {ph}."),
    _tpl("s4-aeb-arab-callback", "aeb-Arab", {"p1": "person:ar2:m", "ph": "phone:tn@p1"},
         "عيّط ل{p1.first} على {ph}، راهو يستنّى فيك من الصباح.", tags="clitic"),
    _tpl("s4-aeb-arab-cin", "aeb-Arab", {"cin": "cin:east"},
         "رقم بطاقة التعريف متاعي {cin}، نحتاجو باش نكمّل التسجيل.", tags="eastern-digits"),
    _tpl("s4-aeb-arab-two-people", "aeb-Arab",
         {"p1": "person:ar2:m", "p2": "person:ar2:m", "ph": "phone:tn@p2"},
         "{p1.first} و{p2.first} جايين غدوة، {p2.first} خلّى النومرو متاعو: {ph}.", tags="clitic"),
    _tpl("s4-aeb-arab-moved-city", "aeb-Arab", {"p1": "person:ar2:m", "c1": "city:ar", "rib": "rib@p1"},
         "{p1} بدّل السكنى ل{c1}، لازم نبدّلو العنوان في الملف متاعو.",
         (A, "باهي، شنوّة العنوان الجديد متاع {p1.first} في {c1}؟"),
         "باش يبعثهولك بالمايل. وهذا الـRIB الجديد متاعو: {rib}.",
         (A, "الملف متاع {p1} تبدّل، بالـRIB {rib}."),
         tags="clitic"),
    _tpl("s4-aeb-arab-listed", "aeb-Arab", {"p1": "person:ar:m", "ph": "phone:tn@p1"},
         "قلت ل{p1.first} يجيب الأوراق غدوة.",
         (A, "باهي، {p1.first} عندو النومرو متاعك؟"),
         "إيه، أما ابعثلو زادة على {ph}.",
         (A, "بعثتلو ل{p1} على {ph}."),
         tags="clitic"),
    _tpl("s4-aeb-arab-no-pii", "aeb-Arab",
         {"w1": "~word:jari-ar", "w2": "~word:sana-ar", "w3": "~word:farah-ar"},
         "صباح الخير، عندي حساب {w1} في البنك ونحب نسكّرو.",
         (A, "باهي، لازمك تمشي للفرع ببطاقة التعريف."),
         "يعيشك. و{w2} سعيدة لكل العايلة!",
         (A, "وانت من أهلو!"),
         "نشالله العام الجاي يكون فيه {w3} أكثر.",
         tags="common-word-name"),
    # ---- Modern Standard Arabic
    _tpl("s4-ar-doctor", "ar", {"p1": "person:ar2:m", "ph": "phone:tn-fixed@p1"},
         "أريد حجز موعد مع الدكتور {p1.last} الأسبوع القادم.",
         (A, "حسنًا. أي يوم يناسب الدكتور {p1.last}؟"),
         "الثلاثاء. رقم العيادة {ph}، واسمه الكامل {p1}.",
         (A, "تم الحجز مع {p1} يوم الثلاثاء، وسأؤكد على {ph}."),
         tags="honorific"),
    _tpl("s4-ar-email", "ar", {"p1": "person:ar2", "em": "email"},
         "أرسل التقرير إلى {em} وضع {p1.first} في النسخة."),
    _tpl("s4-ar-iban", "ar", {"p1": "person:ar2", "ib": "iban:tn@p1"},
         "يرجى تحويل المبلغ إلى الحساب {ib} باسم {p1}."),
    _tpl("s4-ar-short-then-full", "ar", {"p1": "person:ar2:m", "ph": "phone:tn-east@p1"},
         "اتصل ب{p1.first} صباح الغد، الأمر عاجل.",
         (A, "حسنًا، ما رقم {p1.first}؟"),
         "رقمه {ph}، واسمه الكامل {p1}.",
         (A, "تم تسجيل {p1} على الرقم {ph}."),
         "وأخبر {p1.first} أن الاجتماع تأجل إلى الثالثة.",
         tags="clitic eastern-digits"),
    _tpl("s4-ar-no-pii", "ar", {"w1": "~word:mabrouk-ar", "w2": "~word:sana-ar"},
         "{w1} عليك النجاح! كل {w2} وأنت بخير.", tags="common-word-name"),
    # ---- Code-switched
    _tpl("s4-mix-arab-fr-rib", "aeb-Arab+fr", {"rib": "rib"},
         "ابعثلي الفلوس على الـRIB هذا: {rib}، merci d'avance."),
    _tpl("s4-mix-arab-fr-appointment", "aeb-Arab+fr", {"p1": "person:ar2:m", "ph": "phone:tn@p1"},
         "عندي rendez-vous مع {p1} lundi à 10h، النومرو متاعو {ph}."),
    _tpl("s4-mix-ar-fr-card", "ar+fr", {"cd": "card", "pr": "~price:east"},
         "تم خصم {pr} دينار مرتين من ma carte {cd}، comment je peux contester ?", tags="eastern-digits"),
    _tpl("s4-mix-ar-fr-invoice", "ar+fr", {"p1": "person:ar2:m", "em": "email"},
         "Bonjour, السيد {p1} يطلب نسخة من الفاتورة، envoyez-la à {em} svp.", tags="honorific"),
    _tpl("s4-mix-aeb-en-call", "aeb-Latn+en", {"p1": "person:tn2:m", "ph": "phone:tn@p1"},
         "Can you call {p1.first}? nomrou {ph}, w 9ollou el meeting tfaskhet."),
    _tpl("s4-mix-aeb-en-invoice", "aeb-Latn+en", {"p1": "person:tn2:m", "em": "email@p1"},
         "Send the invoice to {em}, w 9oul l {p1.lower} y5alles 9bal el jem3a."),
    _tpl("s4-mix-en-ar-contract", "en+ar", {"p1": "person:ar2", "em": "email"},
         "Please email the contract to {em}. The signer is {p1}."),
    _tpl("s4-mix-aeb-fr-cin", "aeb-Latn+fr", {"p1": "person:tn2", "cin": "cin"},
         "Bonjour, el CIN mte3i {cin} w ismi {p1}, je voudrais retirer ma carte."),
    _tpl("s4-mix-aeb-fr-honorific", "aeb-Latn+fr", {"p1": "person:tn2:f", "ph": "phone:tn@p1"},
         "Mme {p1.last} 9alet bech tji lundi.",
         (A, "D'accord, je note Mme {p1.last} pour lundi. Vous avez son numéro ?"),
         "Ey, {ph}, w esmha {p1}.",
         (A, "Très bien, j'appellerai {p1} au {ph}."),
         tags="honorific"),
    _tpl("s4-mix-en-fr-account", "en+fr", {"p1": "person:tn2:m", "em": "email@p1"},
         "Hi, je vous écris pour {p1}: he can't access his account, son mail est {em}."),
    _tpl("s4-mix-arab-en-call", "aeb-Arab+en", {"p1": "person:ar2:m", "ph": "phone:tn@p1"},
         "Please call {p1.first} على {ph}، قلّو الـmeeting تفسخ."),
    _tpl("s4-mix-ar-en-meeting", "ar+en", {"p1": "person:ar2:f", "ph": "phone:tn-fixed@p1"},
         "The meeting with الدكتورة {p1.last} is moved to Thursday, please confirm on {ph}.",
         tags="honorific"),
    _tpl("s4-mix-fr-ar-address", "fr+ar", {"p1": "person:ar2", "c1": "city:ar"},
         "Merci de mettre à jour le dossier de {p1}، العنوان الجديد في {c1}."),
    _tpl("s4-mix-en-ar-fees", "en+ar", {"p1": "person:ar2", "ib": "iban:tn@p1"},
         "Transfer the fees to IBAN {ib}، صاحب الحساب {p1}."),
    _tpl("s4-mix-arab-fr-sister", "aeb-Arab+fr",
         {"p1": "person:ar2:f", "c1": "city:ar", "em": "email@p1"},
         "أختي {p1.first} عندها entretien غدوة في {c1}.",
         (A, "Bonne chance à {p1.first} ! Vous voulez que je l'aide à se préparer ?"),
         "إيه، ابعثلها les questions على {em}.",
         (A, "C'est envoyé à {em}.")),
    _tpl("s4-mix-latn-arab-no-pii", "aeb-Latn+aeb-Arab", {"w1": "~word:mabrouk", "w2": "~word:mabrouk-ar"},
         "{w1} 3lik! ألف {w2} يا صاحبي.", tags="common-word-name"),
    _tpl("s4-mix-aeb-fr-friend", "aeb-Latn+fr",
         {"p1": "person:tn2:m", "em": "email@p1", "ph": "phone:tn@p1"},
         "sa7bi {p1.first} y9ra m3aya, il cherche un stage.",
         (A, "D'accord, dans quel domaine ? Je peux aider {p1.first} à rédiger son CV."),
         "Informatique. Ab3athlou 3al {em}, w nomrou {ph}.",
         (A, "C'est fait, j'ai écrit à {p1.first} sur {em}.")),
    _tpl("s4-mix-aeb-fr-prescription", "aeb-Latn+fr", {"p1": "person:tn2", "ph": "phone:tn@p1"},
         "Docteur, l'ordonnance mta3 {p1.first} ndhanni ta7et, tnajjem t3awedha ? Nomrou {ph}."),
    _tpl("s4-mix-en-ar-eid", "en+ar", {"w1": "~word:mabrouk-ar", "w2": "~word:sana-ar"},
         "Happy Eid! عيد {w1} وكل {w2} وأنتم بخير.", tags="common-word-name"),
    _tpl("s4-mix-arab-fr-fees", "aeb-Arab+fr", {"p1": "person:ar2", "ib": "iban:tn@p1"},
         "حوّل les frais d'inscription على الـIBAN {ib}، باسم {p1}."),
    # ---- No PII: common-word names as words, in Arabizi, English and French
    _tpl("s4-aeb-no-pii-ramadan", "aeb-Latn",
         {"w1": "~word:karim", "w2": "~word:sana", "w3": "~word:farah", "w4": "~word:amine"},
         "Ramdhan {w1} lkol!",
         (A, "Allah akram, saha ftourkom."),
         "{w2} sa3ida zeda, w rabbi y5alli el {w3} fi darkom.",
         (A, "{w4} ya rabbi."),
         tags="common-word-name"),
    _tpl("s4-en-no-pii", "en", {"w1": "~word:jack-en", "w2": "~word:grace-en", "w3": "~word:bill-en"},
         "The car {w1} is in the trunk, the {w2} period ends Friday, and the {w3} is paid.",
         tags="common-word-name"),
    _tpl("s4-fr-no-pii", "fr", {"w1": "~word:olivier", "w2": "~word:rose"},
         "On a planté un {w1} et une {w2} dans le jardin, ça change tout.", tags="common-word-name"),
)  # fmt: skip


# `test` split (#74): its own templates and its own held-out names (`tn3`, `ar3`), so nothing tuned on `dev`
# carries over by construction. Frozen with the split: its output is pinned by a test.
TEST_TEMPLATES: tuple[Template, ...] = (
    # ---- French
    _tpl("t-fr-intro-email", "fr", {"p1": "person:tn3", "em": "email@p1", "ph": "phone:tn@p1"},
         "Bonjour, ici {p1}. Pour le dossier, écrivez-moi à {em} ou appelez le {ph}."),
    _tpl("t-fr-salary-iban", "fr", {"ib": "iban:tn"},
         "Merci de verser mon salaire sur l'IBAN {ib} à partir de ce mois."),
    _tpl("t-fr-cin-renewal", "fr", {"cin": "cin", "d": "~date"},
         "Ma carte d'identité n° {cin} expire le {d}, comment la renouveler ?"),
    _tpl("t-fr-lawyer", "fr", {"p1": "person:tn3", "em": "email@p1"},
         "Maître {p1.last} n'a pas encore envoyé le contrat.",
         (A, "Voulez-vous que je relance Maître {p1.last} ?"),
         "Oui, par mail : {em}. Son prénom, c'est {p1.first}.",
         (A, "J'écris à {p1} ({em}) tout de suite."),
         tags="honorific"),
    _tpl("t-fr-no-pii-garden", "fr", {"w1": "~word:olivier", "w2": "~word:rose"},
         "On a planté un {w1} près de la maison et une {w2} rouge devant la porte.", tags="common-word-name"),
    _tpl("t-fr-parcel", "fr", {"r": "~ref", "d": "~date", "pr": "~price"},
         "Le colis {r} est bloqué en douane depuis le {d}, le montant est de {pr} DT."),
    _tpl("t-fr-two-same-first-name", "fr",
         {"p1": "person:tn3:m", "p2": "person:tn3:m=p1", "ph": "phone:tn@p1"},
         "{p1} et {p2} sont convoqués à l'entretien de mardi.",
         (A, "Entendu, je convoque {p1} et {p2}."),
         "{p1.first?} a demandé à décaler l'heure.",
         (A, "S'agit-il de {p1} ou de {p2} ?"),
         "De {p1}. Son numéro : {ph}.",
         tags="ambiguous-reference"),
    _tpl("t-fr-shop-city", "fr", {"p1": "person:tn3:f", "c1": "city", "ph": "phone:tn@p1"},
         "{p1} ouvre une boutique à {c1} le mois prochain.",
         (A, "Bonne nouvelle ! {p1.first} a déjà un local à {c1} ?"),
         "Oui, et elle cherche un comptable. Son numéro : {ph}.",
         (A, "Je transmets le {ph} de {p1} au cabinet.")),
    _tpl("t-fr-card-fraud", "fr", {"cd": "card"},
         "Quelqu'un a utilisé ma carte {cd} sur un site que je ne connais pas, je fais quoi ?"),
    _tpl("t-fr-block-ip", "fr", {"ip": "ip"},
         "Bloque l'adresse {ip}, elle scanne nos ports depuis hier soir."),
    # ---- English
    _tpl("t-en-intro", "en", {"p1": "person:tn3", "ph": "phone:fr@p1"},
         "Hello, my name is {p1} and my number is {ph}."),
    _tpl("t-en-refund-iban", "en", {"p1": "person:tn3", "ib": "iban:fr@p1"},
         "Please refund the deposit to {ib}, the account is under {p1}."),
    _tpl("t-en-ticket", "en", {"p1": "person:tn3", "em": "email@p1", "ph": "phone:tn-fixed@p1"},
         "Our client {p1} reported a bug in the billing page.",
         (A, "Thanks. Do you have {p1.first}'s contact details?"),
         "Yes, {em}, and the office line is {ph}.",
         (A, "I'll email {p1.first} at {em} today."),
         "Great, cc me please."),
    _tpl("t-en-no-pii-car", "en", {"w1": "~word:jack-en", "w2": "~word:grace-en"},
         "My car {w1} broke, and the insurance {w2} period ends on Friday.", tags="common-word-name"),
    _tpl("t-en-no-pii-bill", "en", {"w1": "~word:bill-en", "pr": "~price"},
         "The electricity {w1} is {pr} TND this month, can you check if that's normal?",
         tags="common-word-name"),
    _tpl("t-en-professor", "en", {"p1": "person:tn3:f", "p2": "person:en"},
         "Professor {p1.last} will review the thesis; her assistant is {p2}.", tags="honorific"),
    _tpl("t-en-vpn", "en", {"ip": "ip"},
         "Whitelist {ip} for the VPN, it's the new office gateway."),
    # ---- Modern Standard Arabic
    _tpl("t-ar-intro", "ar", {"p1": "person:ar3", "ph": "phone:tn@p1"},
         "مرحبًا، أنا {p1}، ويمكنكم التواصل معي على الرقم {ph}."),
    _tpl("t-ar-salary-iban", "ar", {"ib": "iban:tn"},
         "الرجاء تحويل الراتب إلى الحساب {ib}."),
    _tpl("t-ar-cin", "ar", {"cin": "cin:east"},
         "رقم بطاقة التعريف الوطنية الخاصة بي هو {cin}.", tags="eastern-digits"),
    _tpl("t-ar-phone-east", "ar", {"p1": "person:ar3", "ph": "phone:tn-east@p1"},
         "اتصلوا ب{p1.first} على الرقم {ph} بعد الظهر.", tags="clitic eastern-digits"),
    _tpl("t-ar-professor", "ar", {"p1": "person:ar3:m", "ph": "phone:tn@p1"},
         "الأستاذ {p1.last} سيتأخر عن الاجتماع.",
         (A, "هل أبلغ بقية الفريق بتأخر الأستاذ {p1.last}؟"),
         "نعم، واسمه الكامل {p1} إن احتجت إليه، ورقمه {ph}.",
         (A, "حسنًا، سأتصل بالأستاذ {p1.last} على {ph} عند الحاجة."),
         tags="honorific clitic"),
    _tpl("t-ar-moved", "ar", {"p1": "person:ar3:f", "c1": "city:ar"},
         "انتقلت {p1} إلى {c1} وتحتاج إلى تحديث عنوانها."),
    _tpl("t-ar-no-pii", "ar", {"w1": "~word:mabrouk-ar", "w2": "~word:rahma-ar"},
         "{w1} على هذا الإنجاز، و{w2} الله على من رحلوا.", tags="common-word-name"),
    _tpl("t-ar-no-pii-life", "ar", {"w1": "~word:hayet-ar", "w2": "~word:chams-ar"},
         "ال{w1} جميلة رغم كل شيء.",
         (A, "صحيح، المهم الصحة."),
         "و{w2} الصباح تعطيني طاقة.",
         (A, "جميل! هل تريد نصائح للنوم المبكر؟"),
         tags="common-word-name"),
    _tpl("t-ar-card", "ar", {"cd": "card"},
         "سُحب مبلغ من بطاقتي {cd} دون علمي."),
    # ---- Tunisian Darija in Arabic script
    _tpl("t-aeb-arab-intro", "aeb-Arab", {"p1": "person:ar3", "ph": "phone:tn@p1"},
         "عسلامة، معاك {p1}، النومرو متاعي {ph}."),
    _tpl("t-aeb-arab-rib", "aeb-Arab", {"rib": "rib"},
         "حوّلّي الفلوس على الحساب هذا {rib} وكلّمني."),
    _tpl("t-aeb-arab-called", "aeb-Arab", {"p1": "person:ar3:m", "ph": "phone:tn@p1"},
         "عيّطت ل{p1.first} ما جاوبنيش، ابعثلو مساج على {ph}.", tags="clitic"),
    _tpl("t-aeb-arab-said", "aeb-Arab", {"p1": "person:ar3:m", "ph": "phone:tn@p1"},
         "{p1.first} قالّي باش يتأخّر شويّة.",
         (A, "باهي، عندك النومرو متاع {p1.first}؟"),
         "إيه، {ph}، واسمو الكامل {p1}.",
         (A, "باهي، نكلّم {p1} على {ph}.")),
    _tpl("t-aeb-arab-two-same-first-name", "aeb-Arab",
         {"p1": "person:ar3:m", "p2": "person:ar3:m=p1", "ph": "phone:tn@p1"},
         "{p1} و{p2} جايين للاجتماع نهار الخميس.",
         (A, "باهي، نستنّاو {p1} و{p2}."),
         "{p1.first?} قالّي باش يجيب الأوراق.",
         (A, "شكون فيهم، {p1} ولا {p2}؟"),
         "{p1}، والنومرو متاعو {ph}.",
         tags="ambiguous-reference clitic"),
    _tpl("t-aeb-arab-cin", "aeb-Arab", {"cin": "cin:east"},
         "رقم بطاقة التعريف متاعي {cin}، تنجم تعمّرلي الاستمارة؟", tags="eastern-digits"),
    _tpl("t-aeb-arab-no-pii", "aeb-Arab", {"w1": "~word:mabrouk-ar", "w2": "~word:sana-ar"},
         "ألف {w1} عالنجاح! وكل {w2} وانتوما طيبين.", tags="common-word-name"),
    _tpl("t-aeb-arab-no-pii-account", "aeb-Arab", {"w1": "~word:nour-ar", "w2": "~word:jari-ar"},
         "يعيشك يا {w1} عينيا، حلّيتلي حساب {w2} في البوسطة.", tags="common-word-name"),
    _tpl("t-aeb-arab-city", "aeb-Arab", {"p1": "person:ar3:m", "c1": "city:ar"},
         "{p1} سكن في {c1} من عام، وتوّا يلوّج على خدمة."),
    # ---- Tunisian Arabizi
    _tpl("t-aeb-intro", "aeb-Latn", {"p1": "person:tn3", "ph": "phone:tn@p1"},
         "Ahla, ena {p1}, kallemni 3al {ph} ki tfadda."),
    _tpl("t-aeb-cv", "aeb-Latn+fr", {"em": "email"},
         "Ab3ath el CV l {em}, houwa elli ychouf les candidatures."),
    _tpl("t-aeb-lowercase", "aeb-Latn", {"p1": "person:tn3:m", "ph": "phone:tn@p1"},
         "kallamt {p1.lower} el bera7, ma jewebnich.",
         (A, "Behi, n3awed nkallmou ena? Chnowa nomrou {p1.first}?"),
         "{ph}, ama 5ir ab3athlou msg.",
         (A, "Ok, b3athtlou msg 3al {ph}.")),
    _tpl("t-aeb-brother", "aeb-Latn", {"p1": "person:tn3:m", "ph": "phone:tn@p1"},
         "khouya {p1.first} y7eb ychri karhba, 3andek fikra?",
         (A, "Ey, 9oulli chnowa el budget mta3 {p1.first}?"),
         "Mech barcha, w ken t7eb tkallmou nomrou {ph}.",
         (A, "Behi, nkallem {p1.first} 3al {ph} ghodwa.")),
    _tpl("t-aeb-sister", "aeb-Latn", {"p1": "person:tn3:f", "c1": "city"},
         "o5ti {p1.first} 3orsha ghodwa fi {c1}, enti jeya?"),
    _tpl("t-aeb-iban", "aeb-Latn", {"ib": "iban:tn"},
         "Ha el IBAN mte3i: {ib}, 7awelli 3lih el flous ki tnajjem."),
    _tpl("t-aeb-no-pii-sky", "aeb-Latn", {"w1": "~word:chams", "w2": "~word:kmar"},
         "El {w1} 7arga barcha lyoum, ama fel lil el {w2} ya3jeb.", tags="common-word-name"),
    _tpl("t-aeb-no-pii-new-job", "aeb-Latn",
         {"w1": "~word:mabrouk", "w2": "~word:amal", "w3": "~word:karim"},
         "{w1} 3lik el 5edma jdida!",
         (A, "Allah ybarek fik."),
         "3andi {w2} kbir fik, w rabbi {w3}.",
         (A, "Rabbi y5allik."),
         tags="common-word-name"),
    _tpl("t-aeb-service", "aeb-Latn+fr", {"sv": "~service"},
         "Kallemt le service client {sv} w ma jewbounich."),
    # ---- Code-switched
    _tpl("t-mix-aeb-fr-doctor", "aeb-Latn+fr", {"p1": "person:tn3", "ph": "phone:tn-fixed@p1"},
         "3andi rendez-vous m3a Dr {p1.last} nhar el jem3a.",
         (A, "D'accord, voulez-vous que je confirme avec le Dr {p1.last} ?"),
         "Ey, w nomrou el cabinet {ph}.",
         (A, "Je confirme avec le Dr {p1.last} au {ph}."),
         tags="honorific"),
    _tpl("t-mix-aeb-fr-rib", "aeb-Latn+fr", {"p1": "person:tn3", "rib": "rib@p1"},
         "Le RIB mta3 {p1} houwa {rib}, fais le virement aujourd'hui."),
    _tpl("t-mix-aeb-en-file", "aeb-Latn+en", {"p1": "person:tn3:m", "em": "email@p1"},
         "Bro, {p1.first} sent me the file, ama ma t7allech. Email him at {em}."),
    _tpl("t-mix-aeb-en-no-pii", "aeb-Latn+en", {"w1": "~word:sana", "w2": "~word:amal"},
         "Kol {w1} w enti bkhir! Happy new year, w nchallah fama {w2} fel 3am jey.",
         tags="common-word-name"),
    _tpl("t-mix-fr-ar-request", "fr+ar", {"p1": "person:ar3", "ph": "phone:tn@p1"},
         "Bonjour, je vous transmets la demande de {p1}، رقم الهاتف {ph}."),
    _tpl("t-mix-arab-fr-meeting", "aeb-Arab+fr", {"p1": "person:ar3:m", "em": "email"},
         "عندي réunion مع {p1} الصباح.",
         (A, "D'accord. Vous voulez que je prévienne {p1.first} ?"),
         "إيه، ابعثلو mail على {em}.",
         (A, "Message envoyé à {em}.")),
    _tpl("t-mix-arab-en-cancel", "aeb-Arab+en", {"p1": "person:ar3"},
         "The meeting with {p1} تفسخ، قلّو please."),
    _tpl("t-mix-en-ar-address", "en+ar", {"p1": "person:ar3:f", "c1": "city:ar"},
         "Please update the address for {p1}: انتقلت إلى {c1}."),
    _tpl("t-mix-fr-en-call", "fr+en", {"p1": "person:tn3", "ph": "phone:fr@p1"},
         "Le client {p1} demande un call ASAP, his number is {ph}."),
    _tpl("t-mix-aeb-arab-ramadan", "aeb-Latn+aeb-Arab", {"w1": "~word:karim"},
         "Ramdhan {w1} lkol! وصحّة شريبتكم.", tags="common-word-name"),
    _tpl("t-mix-aeb-fr-overmask", "aeb-Latn+fr",
         {"w1": "~word:mabrouk", "w2": "~word:sahbi", "w3": "~brand:ooredoo"},
         "{w1} 3lik el diplôme ya {w2} !",
         (A, "Merci, c'est gentil !"),
         "Taw nemchi m3a {w3} nbadel l'abonnement.",
         (A, "D'accord, tu veux que je compare les offres ?"),
         tags="common-word-name"),
    _tpl("t-mix-aeb-fr-cin", "aeb-Latn+fr", {"cin": "cin"},
         "Svp, el CIN mta3 baba {cin}, il faut le mettre fel formulaire."),
    _tpl("t-mix-aeb-en-atm", "aeb-Latn+en", {"cd": "card"},
         "The ATM kla my card {cd}, chnowa na3mel?"),
    _tpl("t-mix-ar-fr-iban", "ar+fr", {"p1": "person:ar3", "ib": "iban:tn@p1"},
         "يرجى تحويل المبلغ sur l'IBAN {ib} باسم {p1}."),
    _tpl("t-mix-aeb-fr-landlord", "aeb-Latn+fr", {"p1": "person:tn3", "rib": "rib@p1"},
         "Le propriétaire {p1} y7eb el kera 9bal le 5 du mois, ab3athlou 3al RIB {rib}."),
    _tpl("t-mix-ar-en-eid", "ar+en", {"w1": "~word:mabrouk-ar"},
         "{w1} عليكم العيد, and have a great holiday!", tags="common-word-name"),
    _tpl("t-mix-arab-fr-moved", "aeb-Arab+fr",
         {"p1": "person:ar3:m", "c1": "city:ar", "ph": "phone:tn@p1"},
         "{p1} بدّل الدار، توّا ساكن في {c1}.",
         (A, "D'accord, je mets à jour l'adresse de {p1.first} à {c1}."),
         "Merci، وهذا النومرو الجديد متاعو {ph}.",
         (A, "C'est noté, {p1} : {ph}.")),
)  # fmt: skip

TEMPLATE_SETS: Mapping[str, tuple[Template, ...]] = {
    "v0": TEMPLATES,
    "multi-turn": MULTI_TURN_TEMPLATES,
    "dev-s4": DEV_S4_TEMPLATES,
    "test": TEST_TEMPLATES,
}


# --------------------------------------------------------------------------- generation


def _person(rng: random.Random, kind: str, taken: set[str], first_name: str | None = None) -> _Value:
    """A synthetic person. `first_name` forces a shared first name (two people called Ahmed)."""
    match = _PERSON.fullmatch(kind)
    if match is None:
        raise ValueError(f"not a person kind: {kind!r}")
    pool, gender = match.group(1), match.group(2) or rng.choice("mf")
    while True:
        first = first_name or rng.choice(FIRST_NAMES[pool][gender])
        last = rng.choice(SURNAMES[pool])
        full = f"{first} {last}"
        if full not in PUBLIC_FIGURES and (first_name or first not in taken) and last not in taken:
            taken.update((first, last))
            return _Value(EntityType.PERSON, {"": full, "first": first, "last": last, "lower": first.lower()})


def _email(rng: random.Random, person: _Value | None) -> str:
    if person is None:
        pool = rng.choice(("tn", "fr", "en"))  # the v0 pools: adding a pool must not change v0 output
        person = _person(rng, f"person:{pool}", set())
    f, last = _fold(person.forms["first"]), _fold(person.forms["last"])
    if not f or not last:  # an Arabic-script name has no ASCII letters: the address uses a Latin one
        return _email(rng, None)
    number = rng.randint(10, 99)
    local = rng.choice((f"{f}.{last}", f"{f[0]}.{last}", f"{f}{last}", f"{f}.{last[:2]}", f"{f}{number}"))
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def _fill(t: Template, rng: random.Random) -> dict[str, _Value]:
    values: dict[str, _Value] = {}
    taken: set[str] = set()
    # People first (those sharing a first name after the one they share it with), so owned values (emails)
    # can be derived from them; then the rest in declaration order.
    order = sorted(
        t.slots, key=lambda s: (not t.slots[s].startswith("person:"), _shares(t.slots[s]) is not None)
    )
    for slot in order:
        base, owner = _split_kind(t.slots[slot])
        if base.startswith("person:"):
            shared = _shares(t.slots[slot])
            values[slot] = _person(rng, base, taken, values[shared].forms["first"] if shared else None)
        elif base == "city":
            values[slot] = _Value(EntityType.LOCATION, {"": rng.choice(CITIES)})
        elif base == "city:ar":
            values[slot] = _Value(EntityType.LOCATION, {"": rng.choice(CITIES_AR)})
        elif base == "email":
            values[slot] = _Value(EntityType.EMAIL, {"": _email(rng, values.get(owner or ""))}, owner)
        elif base in _STRUCTURED:
            etype, make = _STRUCTURED[base]
            values[slot] = _Value(etype, {"": make(rng)}, owner)
        else:
            note, make = _DISTRACTORS[base]
            values[slot] = _Value(None, {"": make(rng)}, note=note)
    return values


def render(t: Template, rng: random.Random, conv_id: str) -> Conversation:
    """Fill one template and return it as a validated conversation."""
    check_template(t)
    values = _fill(t, rng)
    entity_ids: dict[str, str] = {}  # slot → e1, e2… by first appearance
    longest: dict[str, str] = {}
    turns: list[dict[str, Any]] = []
    for role, template_text in t.turns:
        parts: list[str] = []
        mentions: list[dict[str, Any]] = []
        distractors: list[dict[str, Any]] = []
        pos = 0
        length = 0
        for m in _SLOT.finditer(template_text):
            literal = template_text[pos : m.start()]
            parts.append(literal)
            length += len(literal)
            slot, form = m.group(1), m.group(2) or ""
            value = values[slot]
            surface = value.forms[form]
            span: dict[str, Any] = {"text": surface, "start": length, "end": length + len(surface)}
            if value.type is None:
                distractors.append({**span, "note": value.note})
            else:
                entity_ids.setdefault(slot, f"e{len(entity_ids) + 1}")
                mention = {**span, "entity": entity_ids[slot]}
                if m.group(3):
                    mention["ambiguous"] = True
                mentions.append(mention)
                if len(surface) > len(longest.get(slot, "")):
                    longest[slot] = surface
            parts.append(surface)
            length += len(surface)
            pos = m.end()
        parts.append(template_text[pos:])
        turn: dict[str, Any] = {"role": role, "text": "".join(parts), "mentions": mentions}
        if distractors:
            turn["distractors"] = distractors
        turns.append(turn)

    entities: list[dict[str, Any]] = []
    for slot, eid in entity_ids.items():
        value = values[slot]
        etype = value.type
        if etype is None:  # pragma: no cover - distractors never get an entity id
            raise AssertionError(slot)
        entity: dict[str, Any] = {"id": eid, "type": etype.value, "value": longest[slot]}
        if value.owner is not None and value.owner in entity_ids:
            entity["owner"] = entity_ids[value.owner]
        entities.append(entity)

    raw = {
        "id": conv_id,
        "languages": list(t.languages),
        "source": "template",
        "tags": list(t.derived_tags()),
        "entities": entities,
        "turns": turns,
    }
    return parse_conversation(raw, where=f"template {t.name}")


def generate(
    count: int, seed: int, prefix: str, templates: Sequence[Template] = TEMPLATES, start: int = 1
) -> list[Conversation]:
    """`count` conversations, numbered from `start`. Every template is used once per round, shuffled."""
    if count < 0:
        raise ValueError("count must be >= 0")
    if start < 1:
        raise ValueError("start must be >= 1")
    if not templates:
        raise ValueError("no templates")
    for t in templates:
        check_template(t)
    rng = random.Random(seed)  # noqa: S311 - synthetic test data, not a security use
    order = list(templates)
    out: list[Conversation] = []
    for i in range(count):
        if i % len(order) == 0:
            rng.shuffle(order)
        out.append(render(order[i % len(order)], rng, f"{prefix}-{start + i:04d}"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nona.bench.generate", description="Generate benchmark conversations."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--prefix", required=True, help="conversation id prefix, e.g. 'dev' → dev-0001")
    parser.add_argument("--start", type=int, default=1, help="number of the first conversation")
    parser.add_argument("--set", choices=sorted(TEMPLATE_SETS), default="v0", help="which templates to use")
    args = parser.parse_args(argv)
    templates = TEMPLATE_SETS[args.set]
    sys.stdout.write(dumps_jsonl(generate(args.count, args.seed, args.prefix, templates, args.start)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
