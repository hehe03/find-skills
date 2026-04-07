import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(
        self,
        dimension: int = 1024,
        index_path: Optional[str] = None,
        metric: str = "ip",
    ):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata: List[Dict[str, Any]] = []
        
        if metric == "ip":
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.metric = metric

    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if embeddings.shape[0] == 0:
            return
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"expected dimension {self.dimension}"
            )
        
        if self.metric == "ip":
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} does not match "
                f"expected dimension {self.dimension}"
            )
        
        if self.metric == "ip":
            faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if score_threshold is not None and dist < score_threshold:
                continue
            
            result = {
                "score": float(dist),
                "index": int(idx),
                "metadata": self.metadata[idx],
            }
            results.append(result)
        
        return results

    def save(self, path: Optional[str] = None) -> str:
        save_path = path or self.index_path
        if save_path is None:
            raise ValueError("No index path specified")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, f"{save_path}.index")
        
        with open(f"{save_path}.meta", "wb") as f:
            pickle.dump(self.metadata, f)
        
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        load_path = path or self.index_path
        if load_path is None:
            raise ValueError("No index path specified")
        
        if not Path(f"{load_path}.index").exists():
            return
        
        self.index = faiss.read_index(f"{load_path}.index")
        
        with open(f"{load_path}.meta", "rb") as f:
            self.metadata = pickle.load(f)

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def get_total(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        self.index.reset()
        self.metadata = []
