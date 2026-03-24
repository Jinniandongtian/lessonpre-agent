"""Embedding模型封装"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        configured_remote_model = (os.getenv("EMBEDDING_MODEL") or "").strip()
        sf_api_key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
        sf_base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        local_model_name = (model_name or "").strip() or DEFAULT_LOCAL_MODEL
        self.model_name = ""
        self._local_model_name = local_model_name
        self._remote_model_name = configured_remote_model
        self._dimension: Optional[int] = int(dimension) if dimension is not None else None
        self._backend: Optional[str] = None
        self._model = None
        self._api_client = None
        self._sf_base_url = sf_base_url

        if self._dimension is not None and self._dimension <= 0:
            raise ValueError("dimension 必须为正整数")

        # 规则：
        # 1. 远端配置完整（EMBEDDING_MODEL + SILICONFLOW_API_KEY）时优先使用远端
        # 2. 否则使用本地 sentence-transformers，模型名由构造参数 model_name 指定
        if configured_remote_model and sf_api_key:
            self._init_siliconflow(api_key=sf_api_key)
            return

        self._backend = "sentence_transformers"
        self.model_name = self._local_model_name

    def _init_siliconflow(self, api_key: Optional[str]) -> None:
        if not self._remote_model_name:
            raise ValueError("使用 SiliconFlow embedding 时必须设置 EMBEDDING_MODEL")
        if not api_key:
            raise ValueError(
                "检测到 EMBEDDING_MODEL，但未设置 SILICONFLOW_API_KEY。"
                "如需本地模型，请在 EmbeddingModel(model_name=...) 中传入本地模型名。"
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("未安装 OpenAI 客户端，无法调用 SiliconFlow embedding") from exc

        self._api_client = OpenAI(api_key=api_key, base_url=self._sf_base_url)
        self._backend = "siliconflow"
        self.model_name = self._remote_model_name

    # 校验本地embedding模型的维度是否与期望的维度一致
    def _set_detected_dimension(self, detected_dim: int, source: str) -> None:
        if detected_dim <= 0:
            raise RuntimeError(f"{source} 返回了非法 embedding 维度: {detected_dim}")
        if self._dimension is not None and self._dimension != detected_dim:
            raise RuntimeError(
                f"{source} embedding 维度与期望不一致：期望 {self._dimension}，实际 {detected_dim}"
            )
        self._dimension = detected_dim

    # 懒加载本地embedding模型
    def _ensure_local_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError("未安装 sentence-transformers，无法加载本地 embedding 模型") from exc

        try:
            self._model = SentenceTransformer(self._local_model_name)
        except Exception as exc:
            raise RuntimeError(f"加载本地 embedding 模型失败：{self._local_model_name}") from exc

        dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(dim):
            self._set_detected_dimension(int(dim()), "sentence-transformers")

    @property # 把方法伪装成属性
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        if self._backend == "sentence_transformers":
            self._ensure_local_model()
            if self._dimension is not None:
                return self._dimension
        raise RuntimeError("当前 embedding 维度尚未确定；远端模型会在第一次成功 encode 后自动探测")

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._backend == "siliconflow" and self._api_client is not None:
            try:
                model = self.model_name
                resp = self._api_client.embeddings.create(model=model, input=texts)
                embeddings = [d.embedding for d in resp.data]
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"SiliconFlow 返回向量数量异常：期望 {len(texts)}，实际 {len(embeddings)}"
                    )
                if embeddings:
                    detected_dim = len(embeddings[0])
                    self._set_detected_dimension(detected_dim, "SiliconFlow")
                    if any(len(emb) != detected_dim for emb in embeddings):
                        raise RuntimeError("SiliconFlow 返回了不一致的 embedding 维度")
                return embeddings
            except Exception as exc:
                raise RuntimeError(f"siliconflow embedding 失败：{exc}") from exc

        if self._backend == "sentence_transformers":
            self._ensure_local_model()

        if self._backend == "sentence_transformers" and self._model is not None:
            try:
                vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as exc:
                raise RuntimeError(f"sentence_transformers embedding 失败：{exc}") from exc
            if vectors.ndim != 2:
                raise RuntimeError(f"sentence_transformers 返回了非法向量形状: {vectors.shape}")
            if vectors.shape[0] != len(texts):
                raise RuntimeError(
                    f"sentence_transformers 返回向量数量异常：期望 {len(texts)}，实际 {vectors.shape[0]}"
                )
            if vectors.shape[0] > 0:
                self._set_detected_dimension(int(vectors.shape[1]), "sentence-transformers")
            return vectors.astype(np.float32).tolist()

        raise RuntimeError("EmbeddingModel 未正确初始化")

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def encode_question(self, question: Dict[str, Any]) -> List[float]:
        """
        用 embedding_text 生成向量。
        embedding_text 是唯一用于向量化的字段；其他字段（source_meta、
        question_meta、content 等）只用于过滤和渲染，不参与向量计算。
        兜底：旧数据没有 embedding_text 时，依次尝试 stem_plain、stem_latex。
        """
        text = question.get('embedding_text', '')
        if not text:
            content = question.get('content', {})
            if isinstance(content, dict):
                text = content.get('stem_plain', '') or content.get('stem_latex', '')
            elif isinstance(content, str):
                # 旧格式：content 是纯字符串
                text = content
        return self.encode_single(text or '')
