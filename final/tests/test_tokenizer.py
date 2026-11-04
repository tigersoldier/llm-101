import pytest

from llm101.tokenizer import BPETokenizer, _merge_pair


def test_bpe_round_trip_for_english_and_chinese() -> None:
    texts = ["hello world", "hello there", "你好，世界。", "你好，朋友。"]
    tokenizer = BPETokenizer.train(texts, vocab_size=80)

    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_is_deterministic() -> None:
    texts = ["abab", "abab", "abac", "中文文本"]
    first = BPETokenizer.train(texts, vocab_size=24)
    second = BPETokenizer.train(texts, vocab_size=24)

    assert first.vocabulary == second.vocabulary
    assert first.merges == second.merges


def test_ranked_encoder_matches_sequential_merge_reference() -> None:
    texts = ["abababab", "abab", "abac", "bbab", "aaaaaaa"]
    tokenizer = BPETokenizer.train(texts, vocab_size=16, min_frequency=1)
    token_to_id = tokenizer.token_to_id

    for text in texts:
        pieces = list(text)
        for pair in tokenizer.merges:
            pieces = _merge_pair(pieces, pair)
        expected = [token_to_id.get(piece, tokenizer.unk_id) for piece in pieces]
        assert tokenizer.encode(text) == expected


def test_tokenizer_can_use_alphabet_collected_from_full_corpus() -> None:
    tokenizer = BPETokenizer.train(
        ["aa", "aa"], vocab_size=6, initial_alphabet=("a", "中")
    )

    assert tokenizer.encode("中") == [tokenizer.token_to_id["中"]]
    assert tokenizer.unk_id not in tokenizer.encode("中")


def test_unknown_code_point_uses_unknown_token() -> None:
    tokenizer = BPETokenizer.train(["abc", "abc", "abd"], vocab_size=12)

    ids = tokenizer.encode("ab☃c")

    assert tokenizer.unk_id in ids
    assert tokenizer.decode(ids) == "ab�c"


def test_optional_sequence_markers_are_skipped_when_decoding() -> None:
    tokenizer = BPETokenizer.train(["once upon a time"], vocab_size=40)

    ids = tokenizer.encode("once upon a time", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "once upon a time"


def test_tokenizer_round_trips_through_json(tmp_path) -> None:
    texts = ["hello ", "你好"]
    tokenizer = BPETokenizer.train(texts, vocab_size=24)
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.vocabulary == tokenizer.vocabulary
    assert loaded.merges == tokenizer.merges
    assert loaded.decode(loaded.encode("hello 你好")) == "hello 你好"


def test_vocab_must_fit_all_observed_code_points() -> None:
    with pytest.raises(ValueError, match="observed alphabet"):
        BPETokenizer.train(["abc"], vocab_size=6)


def test_reserved_special_token_spellings_are_rejected() -> None:
    tokenizer = BPETokenizer.train(["hello"], vocab_size=16)

    with pytest.raises(ValueError, match="reserved special-token"):
        tokenizer.encode("hello <eos>")
