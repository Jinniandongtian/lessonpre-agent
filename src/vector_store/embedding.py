"""Embedding模型封装"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: int = 384,
    ):
        configured_model = os.getenv("EMBEDDING_MODEL")
        sf_api_key = os.getenv("SILICONFLOW_API_KEY")
        sf_base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        st_model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model_name = configured_model or st_model_name
        self._st_model_name = st_model_name
        self._dimension = int(os.getenv("EMBEDDING_DIM", str(dimension)))
        self._backend = None
        self._model = None
        self._tokenizer = None
        self._api_client = None
        self._api_dimension = None
        self._sf_base_url = sf_base_url

        # 规则：优先遵循 .env 中的 EMBEDDING_MODEL
        # - 若设置为 hash/simple：强制使用本地 hash embedding（不依赖 sentence-transformers）
        # - 其它值：视为 sentence-transformers 的模型名
        if configured_model and configured_model.strip().lower() in {"hash", "simple"}:
            self._backend = "hash"
            return

        if configured_model and sf_api_key:
            try:
                from openai import OpenAI

                self._api_client = OpenAI(api_key=sf_api_key, base_url=sf_base_url)
                self._backend = "siliconflow"
                return
            except Exception:
                self._api_client = None

        self._backend = "sentence_transformers"

    @property # 把方法伪装成属性
    def dimension(self) -> int:
        if self._api_dimension is not None:
            return int(self._api_dimension)
        if self._backend == "sentence_transformers" and self._model is not None:
            dim = getattr(self._model, "get_sentence_embedding_dimension", None)
            if callable(dim):
                return int(dim())
        return self._dimension

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._backend == "siliconflow" and self._api_client is not None:
            try:
                model = self.model_name
                resp = self._api_client.embeddings.create(model=model, input=texts)
                embeddings = [d.embedding for d in resp.data]
                if embeddings and self._api_dimension is None:
                    self._api_dimension = len(embeddings[0])
                return embeddings
            except Exception:
                pass

        if self._backend == "sentence_transformers" and self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self._st_model_name)
            except Exception:
                return [self._hash_embed(t) for t in texts]

        if self._backend == "sentence_transformers" and self._model is not None:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype(np.float32).tolist()

        return [self._hash_embed(t) for t in texts]

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def _hash_embed(self, text: str) -> List[float]:
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not text:
            return vec.tolist()

        for token in text.split():
            idx = (hash(token) % self._dimension + self._dimension) % self._dimension
            vec[idx] += 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()
