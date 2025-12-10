"""Embedding模型封装"""
from typing import List, Optional
import os


class EmbeddingModel:
    """Embedding模型封装"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: 模型名称，可选 "openai", "sentence-bert"
                        默认会根据环境变量选择：
                        - 如果存在 SILICONFLOW_API_KEY：使用 openai 协议 + deepseek-embedding
                        - 否则使用 text-embedding-3-small (OpenAI)
        """
        # 自动选择模式
        self.model_name = model_name or "openai"
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        if self.model_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
            if not api_key:
                raise ValueError("需要设置 OPENAI_API_KEY 或 SILICONFLOW_API_KEY")
            
            # 如果使用 SiliconFlow，默认用官方提供的通用 embedding 模型 text-embedding-v1
            # （可在 .env 设置 EMBEDDING_MODEL 覆盖，常用：deepseek-embedding）
            if os.getenv("SILICONFLOW_API_KEY"):
                self._model = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
            else:
                self._model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        elif self.model_name == "sentence-bert":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except ImportError:
                raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量列表
        """
        if self.model_name == "openai":
            return self._encode_openai(texts)
        elif self.model_name == "sentence-bert":
            return self._encode_sentence_bert(texts)
    
    def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI 兼容 API 生成 embedding（支持 SiliconFlow）"""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
            base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.openai.com/v1")
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model=self._model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding失败: {e}")
    
    def _encode_sentence_bert(self, texts: List[str]) -> List[List[float]]:
        """使用Sentence-BERT生成embedding"""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        return self.encode([text])[0]

