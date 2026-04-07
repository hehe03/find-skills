from typing import List, Optional, Union

import numpy as np

from ..logger import logger


class EmbeddingModel:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 512):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self._model = None
        self._dimension = None
        self._load_model()

    def _load_model(self):
        """Load model at startup and keep it in memory"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_path, 
                device=self.device,
                trust_remote_code=True
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"[EmbeddingModel] Model loaded: {self.model_path}, dimension: {self._dimension}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. Install it with: pip install sentence-transformers"
            )

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if self._model is None:
            self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        if self._model is None:
            self._load_model()
        
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0]

    def get_dimension(self) -> int:
        if self._dimension is None:
            if self._model is not None:
                self._dimension = self._model.get_sentence_embedding_dimension()
            else:
                self._load_model()
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._dimension

    def reload_model(self):
        """Reload model (for hot reload)"""
        self._model = None
        self._dimension = None
        self._load_model()
