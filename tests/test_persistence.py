"""Tests for best-score persistence."""
from __future__ import annotations

from pathlib import Path

from reflex.persistence import load_best_score, save_best_score


def test_returns_zero_when_file_missing(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"
    assert load_best_score(target) == 0


def test_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "best_score.txt"
    assert save_best_score(42, target)
    assert load_best_score(target) == 42


def test_returns_zero_on_garbage_contents(tmp_path: Path) -> None:
    target = tmp_path / "garbage.txt"
    target.write_text("not a number", encoding="utf-8")
    assert load_best_score(target) == 0


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dirs" / "best_score.txt"
    assert save_best_score(7, target)
    assert target.exists()
    assert load_best_score(target) == 7


def test_save_uses_atomic_replace(tmp_path: Path) -> None:
    # An aborted write leaves a .tmp behind, never a torn primary file.
    target = tmp_path / "best_score.txt"
    save_best_score(1, target)
    save_best_score(2, target)
    assert load_best_score(target) == 2
    # No leftover temp file from a successful write.
    assert not (tmp_path / "best_score.txt.tmp").exists()
