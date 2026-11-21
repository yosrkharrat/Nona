from __future__ import annotations

import hashlib
import ipaddress
import random
import re
from collections import Counter
from pathlib import Path

import phonenumbers
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona.bench.dataset import dumps_jsonl, load_jsonl, parse_conversation
from nona.bench.generate import (
    CITIES,
    EMAIL_DOMAINS,
    FIRST_NAMES,
    MULTI_TURN_TEMPLATES,
    PUBLIC_FIGURES,
    SURNAMES,
    TEMPLATE_SETS,
    TEMPLATES,
    TEST_CARDS,
    Template,
    check_template,
    generate,
    main,
    render,
)
from nona.core import EntityType
from nona.core.canonical import canonical_tokens
from nona.core.checksums import iban_valid, tn_rib_valid
from nona.core.recognizers import gazetteer

seeds = st.integers(min_value=0, max_value=2**32 - 1)


# --------------------------------------------------------------------------- templates


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_template_is_valid(template: Template) -> None:
    check_template(template)


def test_template_names_are_unique() -> None:
    names = [t.name for t in TEMPLATES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("slots", "turn", "message"),
    [
        ({}, "Hi {p1}", "not declared"),
        ({"p1": "person:tn", "ph": "phone:tn"}, "Hi {p1}", "unused"),
        ({"p1": "person:xx"}, "Hi {p1}", "unknown kind"),
        ({"ph": "phone:tn"}, "Call {ph.first}", "unknown form"),
        ({"ph": "phone:tn@p9"}, "Call {ph}", "not a person slot"),
        ({"p1": "person:tn@p1"}, "Hi {p1}", "can't have an owner"),
        ({}, "Hi {p1", "malformed"),
    ],
)
def test_check_template_rejects_broken_templates(slots: dict[str, str], turn: str, message: str) -> None:
    broken = Template("broken", ("en",), (("user", turn),), slots)
    with pytest.raises(ValueError, match=message):
        check_template(broken)


def test_tags_are_derived_from_the_template() -> None:
    t = Template("t", ("aeb-Latn", "fr"), (("user", "a"), ("assistant", "b")), {}, ("honorific",))
    assert t.derived_tags() == ("honorific", "arabizi", "code-switching", "multi-turn", "no-pii")


# --------------------------------------------------------------------------- generation


def test_generation_is_deterministic() -> None:
    assert dumps_jsonl(generate(40, seed=7, prefix="x")) == dumps_jsonl(generate(40, seed=7, prefix="x"))


def test_different_seeds_give_different_values() -> None:
    assert dumps_jsonl(generate(40, seed=1, prefix="x")) != dumps_jsonl(generate(40, seed=2, prefix="x"))


def test_every_template_is_used_once_per_round() -> None:
    convs = generate(len(TEMPLATES), seed=3, prefix="x")
    shapes = Counter(tuple(t.role for t in c.turns) + c.languages for c in convs)
    expected = Counter(tuple(r for r, _ in t.turns) + t.languages for t in TEMPLATES)
    assert shapes == expected


def test_ids_use_the_prefix() -> None:
    assert [c.id for c in generate(3, seed=1, prefix="dev")] == ["dev-0001", "dev-0002", "dev-0003"]


def test_negative_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="count"):
        generate(-1, seed=1, prefix="x")


@settings(max_examples=40, deadline=None)
@given(seed=seeds)
def test_generated_conversations_round_trip_through_the_loader(seed: int) -> None:
    convs = generate(len(TEMPLATES), seed=seed, prefix="g")
    for conv in convs:
        assert parse_conversation(conv.to_json()) == conv


@settings(max_examples=40, deadline=None)
@given(seed=seeds)
def test_synthetic_values_follow_the_rules(seed: int) -> None:
    """05 §5: reserved domains, valid checksums, test cards, documentation IPs, no public figures."""
    for conv in generate(len(TEMPLATES), seed=seed, prefix="g"):
        for e in conv.entities:
            v = e.value
            if e.type is EntityType.EMAIL:
                assert v.rsplit("@", 1)[1] in EMAIL_DOMAINS
            elif e.type is EntityType.PHONE:
                number = v.replace(
                    "(0)", ""
                )  # the "(0)" habit is written on purpose; the number must be valid
                assert any(
                    phonenumbers.is_valid_number(phonenumbers.parse(number, r)) for r in ("TN", "FR")
                ), v
            elif e.type is EntityType.IBAN:
                assert iban_valid(v)
            elif e.type is EntityType.RIB:
                assert tn_rib_valid(v)
            elif e.type is EntityType.CARD:
                assert re.sub(r"\D", "", v) in TEST_CARDS
            elif e.type is EntityType.IP_ADDRESS:
                assert ipaddress.ip_address(v).is_private  # RFC 5737 ranges are reserved, not routable
            elif e.type is EntityType.NATIONAL_ID:
                assert re.fullmatch(r"[01]\d{7}", v)
            elif e.type is EntityType.PERSON:
                assert v not in PUBLIC_FIGURES
                assert not e.public


@settings(max_examples=40, deadline=None)
@given(seed=seeds)
def test_a_cin_only_appears_next_to_a_context_word(seed: int) -> None:
    """Deep dive A: 8 digits are only an ID when the context says so; otherwise they're a distractor."""
    for conv in generate(len(TEMPLATES), seed=seed, prefix="g"):
        for turn in conv.turns:
            for m in turn.mentions:
                if conv.entity(m.entity).type is EntityType.NATIONAL_ID:
                    assert re.search(r"\bCIN\b|carte d'identité", turn.text)


@settings(max_examples=40, deadline=None)
@given(seed=seeds)
def test_people_in_one_conversation_have_distinct_names(seed: int) -> None:
    for conv in generate(len(TEMPLATES), seed=seed, prefix="g"):
        people = [e.value for e in conv.entities if e.type is EntityType.PERSON]
        tokens = [tok for p in people for tok in p.split()]
        assert len(tokens) == len(set(tokens))


def test_owned_values_point_to_a_person_in_the_conversation() -> None:
    for conv in generate(len(TEMPLATES) * 3, seed=11, prefix="g"):
        for e in conv.entities:
            if e.owner is not None:
                assert conv.entity(e.owner).type is EntityType.PERSON


def test_entity_value_is_the_longest_form_used() -> None:
    t = Template("t", ("fr",), (("user", "{p1.first} puis {p1}"),), {"p1": "person:fr"})
    conv = render(t, random.Random(0), "c")  # noqa: S311
    first, full = (m.text for m in conv.turns[0].mentions)
    assert conv.entities[0].value == full
    assert full.startswith(first + " ")


def test_cli_writes_canonical_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--seed", "5", "--count", "12", "--prefix", "t"]) == 0
    out = capsys.readouterr().out
    path = tmp_path / "t.jsonl"
    path.write_text(out, encoding="utf-8")
    assert dumps_jsonl(load_jsonl(path)) == out


# --------------------------------------------------------------------------- Tunisian numbering plan (#33)


def _parse_tn(value: str) -> phonenumbers.PhoneNumber:
    return phonenumbers.parse(value.replace("(0)", ""), "TN")


def _tn_type(value: str) -> int:
    return phonenumbers.number_type(_parse_tn(value))


def test_tunisian_numbers_cover_mobile_and_fixed_lines() -> None:
    convs = generate(len(TEMPLATES) * 6, seed=4, prefix="g")
    numbers = [_parse_tn(e.value) for c in convs for e in c.entities if e.type is EntityType.PHONE]
    tn = [n for n in numbers if n.country_code == 216]
    types = {phonenumbers.number_type(n) for n in tn}
    assert {phonenumbers.PhoneNumberType.MOBILE, phonenumbers.PhoneNumberType.FIXED_LINE} <= types
    assert {str(n.national_number)[0] for n in tn} >= {"2", "5", "7", "9"}


def test_service_number_distractors_are_service_numbers() -> None:
    service = {
        phonenumbers.PhoneNumberType.TOLL_FREE,
        phonenumbers.PhoneNumberType.SHARED_COST,
        phonenumbers.PhoneNumberType.PREMIUM_RATE,
    }
    distractors = [
        d.text
        for c in generate(len(TEMPLATES) * 6, seed=5, prefix="g")
        for t in c.turns
        for d in t.distractors
        if d.note.startswith("business service number")
    ]
    assert distractors
    assert all(_tn_type(d) in service for d in distractors)


def test_fax_numbers_are_labelled_as_phones() -> None:
    template = next(t for t in TEMPLATES if t.name == "fr-signature")
    conv = render(template, random.Random(0), "c")  # noqa: S311
    assert "Fax : " in conv.turns[0].text
    assert [conv.entity(m.entity).type for m in conv.turns[0].mentions].count(EntityType.PHONE) == 2


# --------------------------------------------------------------------------- multi-turn set (S3)


@pytest.mark.parametrize("template", MULTI_TURN_TEMPLATES, ids=lambda t: t.name)
def test_multi_turn_templates_are_valid_and_long_enough(template: Template) -> None:
    check_template(template)
    assert len(template.turns) >= 4  # 05 §7: conversations of 4+ turns


def test_template_names_are_unique_across_sets() -> None:
    names = [t.name for templates in TEMPLATE_SETS.values() for t in templates]
    assert len(names) == len(set(names))


def test_v0_output_is_pinned() -> None:
    """Adding kinds, pools or template sets must not change what v0 generates from its seed."""
    text = dumps_jsonl(generate(80, seed=1, prefix="dev"))
    assert hashlib.sha256(text.encode()).hexdigest() == V0_SHA256


V0_SHA256 = "2d759e1c5d88711e0b35349807e200961d4027c9c5215ba6980303d9f13cc89a"


def test_held_out_names_are_not_in_the_gazetteer() -> None:
    """`tn2` measures detection, not list overlap: none of its names may be in the gazetteer."""
    pool = [*FIRST_NAMES["tn2"]["m"], *FIRST_NAMES["tn2"]["f"], *SURNAMES["tn2"]]
    leaked = [name for name in pool if canonical_tokens(name) & gazetteer.NAMES]
    assert leaked == []


@settings(max_examples=30, deadline=None)
@given(seed=seeds)
def test_shared_first_names_and_ambiguous_mentions(seed: int) -> None:
    t = Template(
        "t", ("fr",), (("user", "{p1} et {p2}, puis {p1.first?}"),),
        {"p1": "person:tn2:m", "p2": "person:tn2:m=p1"},
    )  # fmt: skip
    conv = render(t, random.Random(seed), "c")  # noqa: S311
    full1, full2, short = conv.turns[0].mentions
    assert full1.text.split()[0] == full2.text.split()[0] == short.text  # one first name
    assert full1.text != full2.text  # two people
    assert short.ambiguous
    assert not full1.ambiguous
    assert short.entity == full1.entity


def test_lower_form_and_city_kind() -> None:
    t = Template(
        "t", ("aeb-Latn",), (("user", "kallem {p1.lower} f {c}"),), {"p1": "person:tn2", "c": "city"}
    )
    conv = render(t, random.Random(1), "c")  # noqa: S311
    name, city = conv.turns[0].mentions
    assert name.text == name.text.lower()
    assert conv.entity(city.entity).type is EntityType.LOCATION
    assert city.text in CITIES


@pytest.mark.parametrize(
    ("slots", "turn", "message"),
    [
        ({"ph": "phone:tn"}, "{ph?}", "only a person mention can be ambiguous"),
        ({"p1": "person:tn=p9"}, "{p1}", "not a person slot"),
        ({"p1": "person:tn=p1"}, "{p1}", "not a person slot"),
    ],
)
def test_check_template_rejects_bad_sharing_and_ambiguity(
    slots: dict[str, str], turn: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_template(Template("bad", ("fr",), (("user", turn),), slots))


def test_cli_generates_the_multi_turn_set_from_a_start_number(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["--set", "multi-turn", "--seed", "2", "--count", "3", "--prefix", "dev", "--start", "81"]) == 0
    )
    ids = [line.split('"')[3] for line in capsys.readouterr().out.splitlines()]
    assert ids == ["dev-0081", "dev-0082", "dev-0083"]


def test_dev_has_forty_conversations_of_four_turns_or_more() -> None:
    dev = load_jsonl(Path(__file__).resolve().parents[2] / "bench" / "data" / "dev.jsonl")
    assert sum(len(c.turns) >= 4 for c in dev) >= 40
