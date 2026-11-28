"""Tests for the chapter 2 teaching corpus."""

from pathlib import Path

_SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _lines(name: str) -> list[str]:
    return (_SAMPLES / name).read_text(encoding="utf-8").splitlines()


def test_training_files_have_expected_shape() -> None:
    english = _lines("english.txt")
    chinese = _lines("chinese.txt")

    assert len(english) == 200
    assert len(chinese) == 200
    assert sum(len(line) for line in english) == 9858
    assert sum(len(line) for line in chinese) == 2851
    assert all(line and line.strip() == line for line in english + chinese)
    assert not any("<" in line or ">" in line or "\0" in line for line in english + chinese)


def test_fixed_examples_appear_three_times() -> None:
    english = _lines("english.txt")
    chinese = _lines("chinese.txt")

    assert english.count("the cat sat on the mat") == 3
    assert chinese.count("小猫坐在垫子上") == 3


def test_heldout_lines_are_separate_and_include_oov() -> None:
    english = _lines("english.txt")
    chinese = _lines("chinese.txt")
    heldout_english = _lines("heldout_english.txt")
    heldout_chinese = _lines("heldout_chinese.txt")

    assert len(heldout_english) == 2
    assert len(heldout_chinese) == 2
    assert not set(heldout_english) & set(english)
    assert not set(heldout_chinese) & set(chinese)
    assert any("鑫" in line for line in heldout_chinese)
    assert all("鑫" not in line for line in english + chinese)
