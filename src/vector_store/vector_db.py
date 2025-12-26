"""向量数据库封装"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import os
from collections import defaultdict
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

    def get_existing_ids(self, ids: List[str]) -> set:
        """获取向量库中已存在的 ids（用于写入前去重/避免重复生成向量）"""
        if not ids:
            return set()
        if self.backend in {"faiss", "simple"}:
            existing = set()
            for m in (self.metadata or []):
                if isinstance(m, dict):
                    mid = m.get("id")
                    if mid:
                        existing.add(mid)
            return existing.intersection(set(ids))
        if self.backend == "chroma":
            try:
                existing = self.collection.get(ids=ids)
                return set(existing.get("ids") or [])
            except Exception:
                return set()
        return set()

    # 返回meta中的source_meta字段，确保返回值一定是字典
    def _get_source_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        # 提取出source_meta字段
        sm = meta.get("source_meta", {})
        # 处理“source_meta是JSON字符串”的场景（比如数据库存储时序列化过）
        if isinstance(sm, str):
            try:
                # 尝试将JSON字符串解析为字典（还原原始结构）
                sm = json.loads(sm)
            except Exception:
                sm = {}
        # 最终兜底：确保返回值一定是字典（排除列表/数字/None等类型）
        if not isinstance(sm, dict):
            sm = {}
        return sm

    # 判断meta中是否匹配给定的filters过滤条件
    def _match_source_meta_filters(self, meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        sm = self._get_source_meta(meta)

        # 内部核心函数：实现文本类字段的“模糊包含匹配”（needle是否在hay中）
        def _contains(hay: Any, needle: Any) -> bool:
            if needle is None:
                return True
            hs = str(hay or "").strip().lower()
            ns = str(needle or "").strip().lower()
            if not ns:
                return True
            return ns in hs
        # 模糊匹配
        if "exam_name" in filters and not _contains(sm.get("exam_name"), filters.get("exam_name")):
            return False
        if "region" in filters and not _contains(sm.get("region"), filters.get("region")):
            return False
        if "grade" in filters and not _contains(sm.get("grade"), filters.get("grade")):
            return False
        if "source_type" in filters and not _contains(sm.get("source_type"), filters.get("source_type")):
            return False
        # 精确匹配
        if "year" in filters and filters.get("year") is not None:
            try:
                fy = int(filters.get("year"))
                my = int(sm.get("year"))
                if my != fy:
                    return False
            except Exception:
                return False
        if "id" in filters and filters.get("id") is not None:
            if str(meta.get("id") or "") != str(filters.get("id")):
                return False
        return True

    # 返回标准化的分页结果（包含符合条件的题目总数 + 分页数据）
    def query_by_source_meta(
        self,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 20,
        include_content: bool = False,
    ) -> Dict[str, Any]:
        filters = filters or {}

        if self.backend == "chroma":
            try:
                res = self.collection.get(include=["documents", "metadatas"])
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
                ids = res.get("ids") or []

                matched = []
                for _id, doc, md in zip(ids, docs, metas):
                    m = {"id": _id, "content": doc, **(md or {})}
                    if self._match_source_meta_filters(m, filters):
                        matched.append(m)

                total = len(matched)
                sliced = matched[offset: offset + limit]
                items: List[Dict[str, Any]] = []
                for m in sliced:
                    item = {
                        "id": m.get("id"),
                        "question_type": m.get("question_type", ""),
                        "knowledge_points": m.get("knowledge_points", []),
                        "difficulty": m.get("difficulty", 3),
                        "source_meta": self._get_source_meta(m),
                    }
                    content = m.get("content", "")
                    if include_content:
                        item["content"] = content
                    else:
                        item["content_preview"] = (content or "")[:200]
                    items.append(item)
                return {"total": total, "items": items}
            except Exception:
                return {"total": 0, "items": []}

        if self.backend in {"faiss", "simple"}:
            # 待筛选的原始数据源，（预处理后的有效元数据列表）
            metas = [m for m in (self.metadata or []) if isinstance(m, dict)]
            # 符合过滤条件的全量题目列表
            matched = [m for m in metas if self._match_source_meta_filters(m, filters)]
            # 符合过滤条件的题目总数
            total = len(matched)
            # 分页后的当前页题目列表
            sliced = matched[offset: offset + limit]
            items: List[Dict[str, Any]] = []
            for m in sliced:
                item = {
                    "id": m.get("id"),
                    "question_type": m.get("question_type", ""),
                    "knowledge_points": m.get("knowledge_points", []),
                    "difficulty": m.get("difficulty", 3),
                    "source_meta": self._get_source_meta(m),
                }
                content = m.get("content", "")
                # 如果需要返回完整内容
                if include_content:
                    item["content"] = content
                else:
                    item["content_preview"] = (content or "")[:200]
                items.append(item)
            return {"total": total, "items": items}

        return {"total": 0, "items": []}

    # 按指定元数字段（如地区 / 年份 / 年级）分组，
    # 统计每个分组下符合过滤条件的题目数量，最终返回按数量降序排列的前 top_k 个分组结果
    # 示例返回结果：[{"key":"北京","count":2}, {"key":"上海","count":1}, {"key":"(empty)","count":1}]
    def group_counts_by_source_meta(
        self,
        group_by: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        counts: Dict[str, int] = defaultdict(int)

        if self.backend == "chroma":
            try:
                res = self.collection.get(include=["documents", "metadatas"])
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
                ids = res.get("ids") or []
                for _id, doc, md in zip(ids, docs, metas):
                    m = {"id": _id, "content": doc, **(md or {})}
                    if not self._match_source_meta_filters(m, filters):
                        continue
                    sm = self._get_source_meta(m)
                    key = sm.get(group_by) if group_by in sm else None
                    key = str(key or "").strip() or "(empty)"
                    counts[key] += 1
            except Exception:
                return []
        else:
            for m in (self.metadata or []):
                if not isinstance(m, dict):
                    continue
                if not self._match_source_meta_filters(m, filters):
                    continue
                sm = self._get_source_meta(m)
                key = sm.get(group_by) if group_by in sm else None
                key = str(key or "").strip() or "(empty)"
                counts[key] += 1

        out = [{"key": k, "count": v} for k, v in counts.items()]
        out.sort(key=lambda x: x["count"], reverse=True)
        return out[:top_k]

    def delete_by_source_meta(
        self,
        filters: Dict[str, Any],
        dry_run: bool = True,
        sample: int = 20,
    ) -> Dict[str, Any]:
        if not filters:
            raise ValueError("filters 不能为空")

        if self.backend == "chroma":
            res = self.query_by_source_meta(filters=filters, offset=0, limit=100_000, include_content=False)
            ids = [x.get("id") for x in (res.get("items") or []) if x.get("id")]
            matched = len(ids)
            if dry_run or matched == 0:
                return {"matched": matched, "deleted": 0, "sample_ids": ids[:sample], "remaining": self.count()}
            self.collection.delete(ids=ids)
            return {"matched": matched, "deleted": matched, "sample_ids": ids[:sample], "remaining": self.count()}

        metas = [m for m in (self.metadata or []) if isinstance(m, dict)]
        to_delete = [i for i, m in enumerate(metas) if self._match_source_meta_filters(m, filters)]
        matched = len(to_delete)
        sample_ids = [metas[i].get("id") for i in to_delete[:sample]]
        if dry_run or matched == 0:
            return {"matched": matched, "deleted": 0, "sample_ids": sample_ids, "remaining": self.count()}

        del_set = set(to_delete)
        keep_idx = [i for i in range(len(metas)) if i not in del_set]
        new_metadata = [metas[i] for i in keep_idx]

        if self.backend == "faiss":
            if self.index is None:
                self.metadata = new_metadata
                self._save_faiss()
                return {"matched": matched, "deleted": matched, "sample_ids": sample_ids, "remaining": self.count()}
            dim = int(getattr(self.index, 'd', 0) or 0)
            try:
                vecs = self.index.reconstruct_n(0, int(self.index.ntotal))
            except Exception as e:
                raise RuntimeError(f"FAISS 索引不支持 reconstruct_n，无法删除：{e}")
            vecs = np.array(vecs, dtype='float32')
            kept = vecs[keep_idx] if keep_idx else np.zeros((0, dim), dtype='float32')
            self.index = faiss.IndexFlatL2(dim) if dim else None
            if self.index is not None and kept.shape[0] > 0:
                self.index.add(kept)
            self.metadata = new_metadata
            self._save_faiss()
        else:
            self.metadata = new_metadata
            self._save_simple()

        return {"matched": matched, "deleted": matched, "sample_ids": sample_ids, "remaining": self.count()}

    def reset(self) -> None:
        if self.backend == "chroma":
            try:
                self.client.delete_collection("questions")
            except Exception:
                pass
            self._init_chroma()
            return

        if self.backend == "faiss":
            self.index = None
            self.dimension = None
            self.metadata = []
            try:
                if getattr(self, "index_path", None) and self.index_path.exists():
                    os.remove(self.index_path)
            except Exception:
                pass
            try:
                if getattr(self, "metadata_path", None) and self.metadata_path.exists():
                    os.remove(self.metadata_path)
            except Exception:
                pass
            return

        self.metadata = []
        self._save_simple()
    
    def add_questions(
        self,
        questions: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Dict[str, int]:
        """
        添加题目和对应的向量
        
        Args:
            questions: 题目列表（包含content等字段）
            embeddings: 对应的向量列表
        """
        if len(questions) != len(embeddings):
            raise ValueError("题目数量和向量数量不匹配")

        total_input = len(questions)

        if self.backend == "faiss":
            return self._add_faiss(questions, embeddings, total_input=total_input)
        elif self.backend == "chroma":
            return self._add_chroma(questions, embeddings, total_input=total_input)
        else:
            return self._add_simple(questions, embeddings, total_input=total_input)
    
    def _add_faiss(
        self,
        questions: List[Dict[str, Any]],
        embeddings: List[List[float]],
        total_input: int,
    ) -> Dict[str, int]:
        """添加到FAISS"""
        # 基于 id 做“库内去重”：FAISS(IndexFlatL2) 不支持删除/原地更新，最稳妥是跳过重复 id
        existing_ids = {m.get("id") for m in (self.metadata or []) if isinstance(m, dict) and m.get("id")}
        filtered_questions: List[Dict[str, Any]] = []
        filtered_embeddings: List[List[float]] = []
        skipped_existing = 0
        for q, emb in zip(questions, embeddings):
            qid = q.get("id")
            if qid and qid in existing_ids:
                skipped_existing += 1
                continue
            filtered_questions.append(q)
            filtered_embeddings.append(emb)
            if qid:
                existing_ids.add(qid)

        if not filtered_questions:
            return {"total_input": total_input, "added": 0, "skipped_existing": skipped_existing}

        embeddings_array = np.array(filtered_embeddings, dtype='float32')
        if embeddings_array.ndim != 2 or embeddings_array.shape[0] == 0:
            raise ValueError(f"无效的向量矩阵形状: {embeddings_array.shape}")
        if embeddings_array.shape[0] != len(filtered_questions):
            raise ValueError(f"题目数量({len(filtered_questions)})与向量数量({embeddings_array.shape[0]})不匹配")
        
        if self.index is None:
            self.dimension = len(filtered_embeddings[0])
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
        for q in filtered_questions:
            self.metadata.append({
                "id": q.get("id", f"q{len(self.metadata)}"),
                "content": q.get("content", ""),
                "question_type": q.get("question_type", ""),
                "knowledge_points": q.get("knowledge_points", []),
                "difficulty": q.get("difficulty", 3),
                "source_meta": q.get("source_meta", {}),
            })
        
        self._save_faiss()

        return {"total_input": total_input, "added": len(filtered_questions), "skipped_existing": skipped_existing}
    
    def _add_chroma(
        self,
        questions: List[Dict[str, Any]],
        embeddings: List[List[float]],
        total_input: int,
    ) -> Dict[str, int]:
        """添加到Chroma（按id去重）"""
        ids = [q.get("id", f"q{i}") for i, q in enumerate(questions)]

        existing_ids: set = set()
        try:
            existing = self.collection.get(ids=ids)
            for _id in (existing.get("ids") or []):
                existing_ids.add(_id)
        except Exception:
            # 某些版本/实现中 get(ids=[]) 或 ids 过长可能报错；这里容错为不预先过滤
            existing_ids = set()

        filtered_ids: List[str] = []
        filtered_embeddings: List[List[float]] = []
        filtered_documents: List[str] = []
        filtered_metadatas: List[Dict[str, Any]] = []

        skipped_existing = 0
        for q, qid, emb in zip(questions, ids, embeddings):
            if qid in existing_ids:
                skipped_existing += 1
                continue
            filtered_ids.append(qid)
            filtered_embeddings.append(emb)
            filtered_documents.append(q.get("content", ""))
            filtered_metadatas.append(
                {
                    "question_type": q.get("question_type", ""),
                    "knowledge_points": ",".join(q.get("knowledge_points", [])),
                    "difficulty": str(q.get("difficulty", 3)),
                    "source_meta": json.dumps(q.get("source_meta", {}), ensure_ascii=False),
                }
            )

        if not filtered_ids:
            return {"total_input": total_input, "added": 0, "skipped_existing": skipped_existing}

        # 优先使用 upsert（如果可用），否则 add
        if hasattr(self.collection, "upsert"):
            self.collection.upsert(
                ids=filtered_ids,
                embeddings=filtered_embeddings,
                documents=filtered_documents,
                metadatas=filtered_metadatas,
            )
        else:
            self.collection.add(
                ids=filtered_ids,
                embeddings=filtered_embeddings,
                documents=filtered_documents,
                metadatas=filtered_metadatas,
            )

        return {"total_input": total_input, "added": len(filtered_ids), "skipped_existing": skipped_existing}
    
    def _add_simple(
        self,
        questions: List[Dict[str, Any]],
        embeddings: List[List[float]],
        total_input: int,
    ) -> Dict[str, int]:
        """添加到简单存储"""
        existing_ids = {m.get("id") for m in (self.metadata or []) if isinstance(m, dict) and m.get("id")}
        skipped_existing = 0
        for q, emb in zip(questions, embeddings):
            qid = q.get("id")
            if qid and qid in existing_ids:
                skipped_existing += 1
                continue
            self.metadata.append({
                "id": q.get("id", f"q{len(self.metadata)}"),
                "content": q.get("content", ""),
                "question_type": q.get("question_type", ""),
                "knowledge_points": q.get("knowledge_points", []),
                "difficulty": q.get("difficulty", 3),
                "source_meta": q.get("source_meta", {}),
                "embedding": emb,  # 简单存储也保存向量
            })
            if qid:
                existing_ids.add(qid)
        self._save_simple()

        added = total_input - skipped_existing
        return {"total_input": total_input, "added": added, "skipped_existing": skipped_existing}
    
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

