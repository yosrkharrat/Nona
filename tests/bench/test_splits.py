"""The committed splits load, and meet the 05 §7 size and balance targets (#74)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nona.bench.dataset import LANGUAGES, Stats, load_dataset, load_jsonl

DATA = Path(__file__).resolve().parents[2] / "bench" / "data"
SPLITS = ("dev", "test")


def _stats(split: str) -> Stats:
    return Stats.of(load_jsonl(DATA / f"{split}.jsonl"))


def test_splits_load_together_with_unique_ids() -> None:
    convs = load_dataset([DATA / f"{split}.jsonl" for split in SPLITS])
    assert all(c.id.startswith(("dev-", "test-")) for c in convs)


def test_size() -> None:
    both = Stats.of(load_dataset([DATA / f"{split}.jsonl" for split in SPLITS]))
    assert both.turns >= 300
    assert both.long >= 40
    test_share = _stats("test").conversations / both.conversations
    assert 0.5 <= test_share <= 0.65  # about 40% dev, 60% test


@pytest.mark.parametrize("split", SPLITS)
def test_balance(split: str) -> None:
    stats = _stats(split)
    for language in sorted(LANGUAGES):
        assert stats.share(stats.languages[language]) >= 0.15, language
    assert stats.share(stats.code_switched) >= 0.30
    assert stats.share(stats.no_pii) >= 0.10


def test_hand_written_hard_cases_and_llm_drafts() -> None:
    """About 50 hand-written hard cases, plus reviewed LLM drafts, split about 40/60 like the rest."""
    dev, test = _stats("dev"), _stats("test")
    assert 40 <= dev.sources["handwritten"] + test.sources["handwritten"] <= 60
    for source in ("handwritten", "llm", "template"):
        share = dev.sources[source] / (dev.sources[source] + test.sources[source])
        assert 0.3 <= share <= 0.5, source


@pytest.mark.parametrize("split", SPLITS)
def test_hand_written_cases_are_tagged(split: str) -> None:
    for conv in load_jsonl(DATA / f"{split}.jsonl"):
        if conv.source == "handwritten":
            assert conv.tags, conv.id
