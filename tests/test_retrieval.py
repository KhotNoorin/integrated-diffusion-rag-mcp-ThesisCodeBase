"""
tests/test_retrieval.py
Tests for retrieval components (embedder, retriever, index builder).
"""

import pytest
from models.retrieval.embedder import Embedder
from models.retrieval.retriever import Retriever

@pytest.fixture
def sample_texts():
    return ["Quantum computing basics", "Deep learning in medical imaging"]

def test_embedding_dimension(sample_texts):
    embedder = Embedder(model_name="clip-ViT-B/32")
    embeddings = embedder.embed_texts(sample_texts)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] > 100, "Embedding dimension too low"

def test_retriever_initialization(tmp_path):
    retriever = Retriever(index_path=tmp_path)
    assert retriever is not None
    assert hasattr(retriever, "search")

def test_retrieval_query(tmp_path, sample_texts):
    retriever = Retriever(index_path=tmp_path)
    retriever.build_index(sample_texts)
    results = retriever.search("What is quantum computing?", top_k=1)
    assert isinstance(results, list)