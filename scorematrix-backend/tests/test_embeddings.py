import sys
import types

import numpy as np
import pytest

from app.errors import EmbeddingsUnavailableError
from app.services.embeddings import HashingEmbedder, SentenceTransformerEmbedder, build_embedding_service
from tests.conftest import make_settings


def test_hashing_embedder_is_deterministic_normalised_and_lexical():
    e = HashingEmbedder()
    v = e.embed(["the cell membrane", "the cell membrane", "french revolution", ""])
    assert v.dtype == np.float32 and np.allclose(v[0], v[1])
    assert np.isclose(np.linalg.norm(v[0]), 1.0) and np.linalg.norm(v[3]) == 0
    assert float(v[0] @ v[1]) > float(v[0] @ v[2]) and e.semantic is False


def test_backend_selection_from_settings():
    assert isinstance(build_embedding_service(make_settings(embedding_backend="hashing")), HashingEmbedder)
    assert isinstance(build_embedding_service(make_settings(embedding_backend="sentence-transformers")), SentenceTransformerEmbedder)


def test_failed_model_load_is_typed_and_not_retried_every_request(monkeypatch):
    calls = []

    class Boom:
        def __init__(self, name):
            calls.append(name)
            raise OSError("no network")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Boom))
    emb = SentenceTransformerEmbedder(make_settings(embedding_model="org/some-model"))
    for _ in range(3):
        with pytest.raises(EmbeddingsUnavailableError) as exc:
            emb.embed(["hello"])
    assert len(calls) == 1, "cooldown must stop a retry storm"
    assert "org/some-model" in exc.value.message and "no network" not in exc.value.message


def test_sentence_transformer_wrapper_normalises_and_types_output(monkeypatch):
    class Fake:
        def __init__(self, name):
            self.kwargs = None

        def encode(self, texts, **kwargs):
            Fake.kwargs = kwargs
            return [[3.0, 4.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Fake))
    emb = SentenceTransformerEmbedder(make_settings())
    out = emb.embed(["a", "b"])
    assert out.shape == (2, 2) and out.dtype == np.float32
    assert Fake.kwargs["normalize_embeddings"] is True and Fake.kwargs["convert_to_numpy"] is True
    assert emb.embed([]).shape[0] == 0
