from __future__ import annotations

from collections.abc import Sequence

from local_graphrag.ports.interfaces import Embedder


class E5Embedder(Embedder):
    """Lazy sentence-transformers adapter for multilingual-e5-small."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
        return int(getter())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = self._load().encode([f"passage: {text}" for text in texts], normalize_embeddings=True)
        return values.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._load().encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
