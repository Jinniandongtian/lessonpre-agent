"""向量数据库封装"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告：FAISS未安装，将使用简单的内存存储。建议安装: pip install faiss-cpu")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception as e:
    # 容错：当chromadb依赖（如onnxruntime）缺失或不兼容时，禁用chroma
    print(f"警告：Chroma不可用，原因: {e}")
    CHROMA_AVAILABLE = False


class VectorDatabase:
    """向量数据库：支持FAISS和Chroma"""
    
    def __init__(
        self,
        storage_path: str = "data/vector_db",
        backend: str = "faiss"
    ):
        """
        Args:
            storage_path: 存储路径
            backend: 后端类型，"faiss" 或 "chroma"
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        
        if backend == "faiss":
            if not FAISS_AVAILABLE:
                print("FAISS不可用，使用简单内存存储")
                backend = "simple"
            else:
                self._init_faiss()
        elif backend == "chroma":
            if not CHROMA_AVAILABLE:
                print("Chroma不可用，使用简单内存存储")
                backend = "simple"
            else:
                self._init_chroma()
        
        if backend == "simple":
            self._init_simple()
    
    def _init_faiss(self):
        """初始化FAISS"""
        self.backend = "faiss"
        self.index = None
        self.dimension = None
        self.metadata = []  # 存储题目的元数据
        self.metadata_path = self.storage_path / "metadata.json"
        self.index_path = self.storage_path / "index.faiss"
        self._load_faiss()
    
    def _init_chroma(self):
        """初始化Chroma"""
        self.backend = "chroma"
        self.client = chromadb.PersistentClient(path=str(self.storage_path))
        self.collection = self.client.get_or_create_collection(
            name="questions",
            metadata={"description": "数学题目向量库"}
        )
    
    def _init_simple(self):
        """初始化简单内存存储（备用方案）"""
        self.backend = "simple"
        self.vectors = []
        self.metadata = []
        self.metadata_path = self.storage_path / "metadata.json"
        self._load_simple()
    
    def _load_faiss(self):
        """加载FAISS索引"""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.dimension = self.index.d
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"加载FAISS索引：{self.index.ntotal} 条记录")
            except Exception as e:
                print(f"加载FAISS索引失败: {e}")
                self.index = None
                self.metadata = []
    
    def _load_simple(self):
        """加载简单存储"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", [])
                    # 简单存储不持久化向量，只存储元数据
            except Exception as e:
                print(f"加载元数据失败: {e}")
                self.metadata = []
    
    def add_questions(
        self,
        questions: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """
        添加题目和对应的向量
        
        Args:
            questions: 题目列表（包含content等字段）
            embeddings: 对应的向量列表
        """
        if len(questions) != len(embeddings):
            raise ValueError("题目数量和向量数量不匹配")
        
        if self.backend == "faiss":
            self._add_faiss(questions, embeddings)
        elif self.backend == "chroma":
            self._add_chroma(questions, embeddings)
        else:
            self._add_simple(questions, embeddings)
    
    def _add_faiss(self, questions: List[Dict[str, Any]], embeddings: List[List[float]]):
        """添加到FAISS"""
        embeddings_array = np.array(embeddings, dtype='float32')
        if embeddings_array.ndim != 2 or embeddings_array.shape[0] == 0:
            raise ValueError(f"无效的向量矩阵形状: {embeddings_array.shape}")
        if embeddings_array.shape[0] != len(questions):
            raise ValueError(f"题目数量({len(questions)})与向量数量({embeddings_array.shape[0]})不匹配")
        
        if self.index is None:
            self.dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            expected_dim = int(getattr(self.index, 'd', self.dimension or 0))
            if expected_dim and embeddings_array.shape[1] != expected_dim:
                raise ValueError(
                    f"向量维度不匹配：当前索引维度={expected_dim}，本次导入维度={embeddings_array.shape[1]}。"
                    f"请删除 data/vector_db/index.faiss 和 data/vector_db/metadata.json 后重试。"
                )
        
        # 添加到索引
        self.index.add(embeddings_array)
        
        # 保存元数据
        for q in questions:
            self.metadata.append({
                "id": q.get("id", f"q{len(self.metadata)}"),
                "content": q.get("content", ""),
                "question_type": q.get("question_type", ""),
                "knowledge_points": q.get("knowledge_points", []),
                "difficulty": q.get("difficulty", 3),
                "source_meta": q.get("source_meta", {}),
            })
        
        self._save_faiss()
    
    def _add_chroma(self, questions: List[Dict[str, Any]], embeddings: List[List[float]]):
        """添加到Chroma"""
        ids = [q.get("id", f"q{i}") for i, q in enumerate(questions)]
        documents = [q.get("content", "") for q in questions]
        metadatas = [
            {
                "question_type": q.get("question_type", ""),
                "knowledge_points": ",".join(q.get("knowledge_points", [])),
                "difficulty": str(q.get("difficulty", 3)),
            }
            for q in questions
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def _add_simple(self, questions: List[Dict[str, Any]], embeddings: List[List[float]]):
        """添加到简单存储"""
        for q, emb in zip(questions, embeddings):
            self.metadata.append({
                "id": q.get("id", f"q{len(self.metadata)}"),
                "content": q.get("content", ""),
                "question_type": q.get("question_type", ""),
                "knowledge_points": q.get("knowledge_points", []),
                "difficulty": q.get("difficulty", 3),
                "source_meta": q.get("source_meta", {}),
                "embedding": emb,  # 简单存储也保存向量
            })
        self._save_simple()
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索相似题目
        
        Args:
            query_embedding: 查询向量
            top_k: 返回前k个结果
            filters: 过滤条件（如knowledge_points, difficulty等）
        
        Returns:
            [(题目字典, 相似度分数), ...]
        """
        if self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k, filters)
        elif self.backend == "chroma":
            return self._search_chroma(query_embedding, top_k, filters)
        else:
            return self._search_simple(query_embedding, top_k, filters)
    
    def _search_faiss(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """FAISS搜索"""
        if self.index is None or len(self.metadata) == 0:
            return []
        
        query_array = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_array, min(top_k * 2, len(self.metadata)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # 应用过滤
                if filters:
                    if not self._match_filters(metadata, filters):
                        continue
                
                # 转换为相似度分数（距离越小，相似度越高）
                similarity = 1 / (1 + dist)
                results.append((metadata, similarity))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _search_chroma(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Chroma搜索"""
        where = {}
        if filters:
            if "knowledge_points" in filters:
                # Chroma的过滤语法
                where["knowledge_points"] = {"$contains": filters["knowledge_points"][0]}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None
        )
        
        output = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            output.append((
                {
                    "content": doc,
                    **metadata
                },
                1 / (1 + distance)  # 转换为相似度
            ))
        
        return output
    
    def _search_simple(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """简单存储搜索（使用余弦相似度）"""
        if not self.metadata:
            return []
        
        query_norm = np.array(query_embedding) / (np.linalg.norm(query_embedding) + 1e-8)
        
        similarities = []
        for meta in self.metadata:
            if "embedding" not in meta:
                continue
            
            # 应用过滤
            if filters:
                if not self._match_filters(meta, filters):
                    continue
            
            emb = np.array(meta["embedding"])
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append((meta, similarity))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        if "knowledge_points" in filters:
            meta_kp = set(metadata.get("knowledge_points", []))
            filter_kp = set(filters["knowledge_points"])
            if not meta_kp.intersection(filter_kp):
                return False
        
        if "difficulty" in filters:
            diff_range = filters["difficulty"]
            meta_diff = metadata.get("difficulty", 3)
            if not (diff_range[0] <= meta_diff <= diff_range[1]):
                return False
        
        return True
    
    def _save_faiss(self):
        """保存FAISS索引"""
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _save_simple(self):
        """保存简单存储"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({"metadata": self.metadata}, f, ensure_ascii=False, indent=2)
    
    def count(self) -> int:
        """获取题目数量"""
        if self.backend == "faiss":
            return self.index.ntotal if self.index else 0
        elif self.backend == "chroma":
            return self.collection.count()
        else:
            return len(self.metadata)

