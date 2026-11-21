"""Template generator for the benchmark dataset (docs/design/05-dataset.md, sections 3 and 5).

Templates are written by hand. The generator fills their slots with synthetic values from a seeded RNG, so
the gold labels are exact by construction: the generator knows where it put every value. Output goes through
`parse_conversation`, so a generated conversation is held to the same rules as a hand-written one.

Usage:
    python -m nona.bench.generate --seed 1 --count 80 --prefix dev > bench/data/dev.jsonl

Slot syntax inside a template turn: `{name}` or `{name.form}`. Each slot name is declared once per template
with a kind (see `KINDS`). Person slots have the forms `first` and `last` besides the full name. A kind may
name an owner, `phone:tn@p1`, which links the value to person `p1` (deep dive B, step 6). Kinds starting with
`~` are distractors: they look like personal data but aren't, and they're recorded for over-masking analysis.

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

# --------------------------------------------------------------------------- checksums


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _alnum_to_int(s: str) -> int:
    """IBAN digit expansion: letters become 10..35 (ISO 13616)."""
    return int("".join(str(int(c, 36)) for c in s))


def iban_check_digits(country: str, bban: str) -> str:
    return f"{98 - _alnum_to_int(bban + country + '00') % 97:02d}"


def iban_valid(iban: str) -> bool:
    compact = iban.replace(" ", "").upper()
    if len(compact) < 15 or not compact.isalnum():
        return False
    return _alnum_to_int(compact[4:] + compact[:4]) % 97 == 1


def tn_rib_valid(rib: str) -> bool:
    """A Tunisian RIB is 20 digits whose last two are a key that makes the whole number divisible by 97."""
    digits = rib.replace(" ", "")
    return len(digits) == 20 and digits.isdigit() and int(digits) % 97 == 0


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
}  # fmt: skip
SURNAMES: Mapping[str, tuple[str, ...]] = {
    "tn": ("Ben Salem", "Trabelsi", "Gharbi", "Mejri", "Ayari", "Sassi", "Dridi", "Khelifi", "Chaabane",
           "Ben Amor", "Mansouri", "Jlassi", "Ferchichi", "Ouertani", "Belhadj", "Hamdi", "Zouari", "Ksouri",
           "Masmoudi", "Ben Romdhane"),
    "fr": ("Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Lefèvre", "Garnier", "Rousseau", "Fontaine",
           "Chevalier", "Girard", "Mercier"),
    "en": ("Carter", "Hughes", "Bennett", "Foster", "Walsh", "Porter", "Reed", "Hayes", "Turner", "Collins"),
}  # fmt: skip
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


def _phone_tn(rng: random.Random) -> str:
    while True:
        if rng.random() < 0.8:
            national = rng.choice("2459") + _digits(rng, 7)  # mobile
        else:
            national = "7" + rng.choice("12345") + _digits(rng, 6)  # landline
        if _valid_phone(national, "TN"):
            break
    spaced = _group(national, (2, 3, 3), " ")
    dotted = _group(national, (2, 3, 3), ".")
    return rng.choice((spaced, dotted, national, f"+216 {spaced}", f"00216 {spaced}", f"+216{national}"))


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
    "phone:fr": (EntityType.PHONE, _phone_fr),
    "iban:tn": (EntityType.IBAN, _iban_tn),
    "iban:fr": (EntityType.IBAN, _iban_fr),
    "rib": (EntityType.RIB, _rib),
    "card": (EntityType.CARD, _card),
    "cin": (EntityType.NATIONAL_ID, _cin),
    "ip": (EntityType.IP_ADDRESS, _ip),
}
_DISTRACTORS: Mapping[str, tuple[str, Callable[[random.Random], str]]] = {
    "~order": ("order number, not an ID", _order),
    "~price": ("price", _price),
    "~ref": ("reference number", _ref),
    "~date": ("date", _date),
}
_PERSON = re.compile(r"person:(tn|fr|en)(?::([mf]))?")
KINDS = (
    frozenset(_STRUCTURED)
    | frozenset(_DISTRACTORS)
    | {"email"}
    | {f"person:{pool}{g}" for pool in FIRST_NAMES for g in ("", ":m", ":f")}
)


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


_SLOT = re.compile(r"\{(\w+)(?:\.(\w+))?\}")


def _split_kind(kind: str) -> tuple[str, str | None]:
    base, _, owner = kind.partition("@")
    return base, owner or None


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
            slot, form = m.group(1), m.group(2)
            if slot not in t.slots:
                raise ValueError(f"{t.name}: slot {{{slot}}} is not declared")
            base, _ = _split_kind(t.slots[slot])
            if form is not None and (not base.startswith("person:") or form not in {"first", "last"}):
                raise ValueError(f"{t.name}: slot {{{slot}.{form}}}: unknown form for {base}")
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


# --------------------------------------------------------------------------- generation


def _person(rng: random.Random, kind: str, taken: set[str]) -> _Value:
    match = _PERSON.fullmatch(kind)
    if match is None:
        raise ValueError(f"not a person kind: {kind!r}")
    pool, gender = match.group(1), match.group(2) or rng.choice("mf")
    while True:
        first = rng.choice(FIRST_NAMES[pool][gender])
        last = rng.choice(SURNAMES[pool])
        full = f"{first} {last}"
        if full not in PUBLIC_FIGURES and first not in taken and last not in taken:
            taken.update((first, last))
            return _Value(EntityType.PERSON, {"": full, "first": first, "last": last})


def _email(rng: random.Random, person: _Value | None) -> str:
    if person is None:
        pool = rng.choice(tuple(FIRST_NAMES))
        person = _person(rng, f"person:{pool}", set())
    f, last = _fold(person.forms["first"]), _fold(person.forms["last"])
    number = rng.randint(10, 99)
    local = rng.choice((f"{f}.{last}", f"{f[0]}.{last}", f"{f}{last}", f"{f}.{last[:2]}", f"{f}{number}"))
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def _fill(t: Template, rng: random.Random) -> dict[str, _Value]:
    values: dict[str, _Value] = {}
    taken: set[str] = set()
    # People first, so owned values (emails) can be derived from them; then the rest in declaration order.
    order = sorted(t.slots, key=lambda s: not t.slots[s].startswith("person:"))
    for slot in order:
        base, owner = _split_kind(t.slots[slot])
        if base.startswith("person:"):
            values[slot] = _person(rng, base, taken)
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
            span = {"text": surface, "start": length, "end": length + len(surface)}
            if value.type is None:
                distractors.append({**span, "note": value.note})
            else:
                entity_ids.setdefault(slot, f"e{len(entity_ids) + 1}")
                mentions.append({**span, "entity": entity_ids[slot]})
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
    count: int, seed: int, prefix: str, templates: Sequence[Template] = TEMPLATES
) -> list[Conversation]:
    """`count` conversations. Every template is used once per round, in a shuffled order."""
    if count < 0:
        raise ValueError("count must be >= 0")
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
        out.append(render(order[i % len(order)], rng, f"{prefix}-{i + 1:04d}"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nona.bench.generate", description="Generate benchmark conversations."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--prefix", required=True, help="conversation id prefix, e.g. 'dev' → dev-0001")
    args = parser.parse_args(argv)
    sys.stdout.write(dumps_jsonl(generate(args.count, args.seed, args.prefix)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
