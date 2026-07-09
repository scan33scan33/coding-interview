from bpe import decode, encode, train


def test_roundtrip_ascii():
    merges, vocab = train(["the cat sat on the mat", "the cat ate"], 20)
    text = "the cat sat"
    assert decode(encode(text, merges), vocab) == text


def test_roundtrip_unseen_text_and_unicode():
    # Byte-level base vocab: anything round-trips, even bytes never seen in
    # training (no <unk>). Multi-byte UTF-8 must survive token splits.
    merges, vocab = train(["hello world"], 10)
    for text in ["zebra quux", "naïve café ✨", "日本語のテキスト", "  spaces\t\ttabs\n"]:
        assert decode(encode(text, merges), vocab) == text


def test_no_merges_yields_raw_bytes():
    ids = encode("hi ☃", [])
    assert ids == list("hi ☃".encode("utf-8"))


def test_most_frequent_pair_merged_first():
    # "aa" appears (weighted) most often, so the first merge must be (a, a).
    merges, vocab = train(["aaaa aaaa ab"], 1)
    (pair, new_id), = merges
    a = ord("a")
    assert pair == (a, a)
    assert new_id == 256
    assert vocab[new_id] == b"aa"


def test_merges_never_cross_word_boundaries():
    # "b a" has 'b' and 'a' adjacent only across a space — never mergeable.
    merges, _ = train(["b a b a b a b a"], 5)
    learned = [pair for pair, _ in merges]
    assert (ord("b"), ord("a")) not in learned


def test_encode_replays_training_order():
    # Corpus where greedy-by-rank matters: train learns (a,b) before (b,c);
    # encoding "abc" must apply the EARLIER merge even though both match.
    merges, vocab = train(["ab ab ab bc bc"], 2)
    assert merges[0][0] == (ord("a"), ord("b"))
    ids = encode("abc", merges)
    assert ids[0] == merges[0][1]  # first token is the merged "ab"
    assert decode(ids, vocab) == "abc"


def test_compression_on_repetitive_corpus():
    corpus = ["the quick brown fox jumps over the lazy dog"] * 20
    merges, _ = train(corpus, 50)
    text = corpus[0]
    ids = encode(text, merges)
    assert len(ids) < len(text.encode("utf-8")) / 2  # >2x compression


def test_training_is_deterministic():
    corpus = ["deterministic ties tie ties det"]
    m1, _ = train(corpus, 15)
    m2, _ = train(corpus, 15)
    assert m1 == m2
