from grmc.core.embedder import HashingEmbedder, create_embedder


def test_hashing_embedder_normalized_and_deterministic():
    emb = HashingEmbedder(dim=64)
    a = emb.encode("long term memory reflection")
    b = emb.encode("long term memory reflection")
    c = emb.encode("totally different topic about cats")
    assert a == b
    assert len(a) == 64
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    # Different text should not be identical vectors
    assert a != c


def test_create_embedder_hash():
    emb = create_embedder("hashing")
    assert emb.name.startswith("hashing")
    assert len(emb.encode("hello")) == 384
