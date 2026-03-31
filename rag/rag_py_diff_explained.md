# rag.py 改动解读

## 一、改动背景

`rag/rag.py` 的核心变化，是把原来的“混合检索手动分权”改成了“向量召回 + reranker 排序”。

原来的主流程是：

1. 先做向量召回。
2. 再计算语义分。
3. 再计算关键词分。
4. 按手工权重合成最终分数。
5. 按最终分数排序。

现在的主流程是：

1. 先做向量召回，拿到候选文档。
2. 再调用 reranker 服务做重排。
3. 如果 reranker 失败，则回退到向量召回顺序。

也就是说，这次改动的重点不是“换召回方式”，而是“换最终排序方式”。

## 二、代码改动总览

从代码层面看，主要有 6 类改动：

1. 新增 reranker 所需依赖。
2. 废弃旧的手工混合权重参数。
3. 新增 reranker 请求、解析和回退逻辑。
4. 重写 `hybrid_retrieve()` 的最终排序路径。
5. 更新调试信息输出。
6. 微调回答 prompt。

---

## 三、逐段 diff 解释

### 1. 新增 `httpx` 依赖

文件位置：`rag/rag.py`

```diff
+ import httpx
```

作用：

- 用来请求 reranker 的 HTTP 接口。
- 说明 reranker 不是本地纯函数，而是一个远程模型服务。

意义：

- 检索排序能力从“本地规则”转到“模型服务”。
- 后续可以通过替换 reranker 服务来升级排序能力，而不需要重写检索主逻辑。

---

### 2. 废弃旧的混合分权参数

旧逻辑里有三类核心参数：

```diff
- self._hybrid_sem_weight = float(os.getenv("HYBRID_SEM_WEIGHT", "0.75"))
- self._hybrid_keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.25"))
- self._min_final_score = float(os.getenv("RETRIEVAL_MIN_FINAL_SCORE", "0.0"))
```

并且旧代码会做权重归一化：

```diff
- total_weight = self._hybrid_sem_weight + self._hybrid_keyword_weight
- if total_weight <= 0:
-     self._hybrid_sem_weight = 0.75
-     self._hybrid_keyword_weight = 0.25
- else:
-     self._hybrid_sem_weight /= total_weight
-     self._hybrid_keyword_weight /= total_weight
```

新代码改成：

```diff
+ self._deprecated_hybrid_sem_weight = float(os.getenv("HYBRID_SEM_WEIGHT", "0.75"))
+ self._deprecated_hybrid_keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.25"))
+ self._deprecated_min_final_score = float(os.getenv("RETRIEVAL_MIN_FINAL_SCORE", "0.0"))
```

并增加了警告：

```diff
+ print(
+   "[WARN] HYBRID_SEM_WEIGHT/HYBRID_KEYWORD_WEIGHT/"
+   "RETRIEVAL_MIN_FINAL_SCORE are deprecated and ignored. "
+   "Retrieval ranking now uses reranker scores."
+ )
```

作用：

- 旧参数还可以从环境变量里读出来，但已经不再参与排序。
- 如果外部还在配这些变量，系统会明确提示它们已失效。

意义：

- 降低切换到 reranker 的破坏性。
- 避免旧环境继续以为自己在调权重，结果实际不生效。

---

### 3. 新增 reranker 配置

新增的初始化参数包括：

```diff
+ self._reranker_model = os.getenv("RERANKERMODEL", "BAAI/bge-reranker-v2-m3")
+ self._reranker_base_url = (
+     os.getenv("RERANKER_BASE_URL") or embeddings_url or ""
+ ).rstrip("/")
+ self._reranker_api_key = os.getenv("RERANKER_API_KEY") or embeddings_key
+ self._reranker_timeout = float(os.getenv("RERANKER_TIMEOUT", "20"))
+ self._reranker_path = os.getenv("RERANKER_PATH", "/rerank").strip()
```

以及 path 修正：

```diff
+ if not self._reranker_path.startswith("/"):
+     self._reranker_path = f"/{self._reranker_path}"
```

还有空地址告警：

```diff
+ if not self._reranker_base_url:
+     print(
+         "[WARN] RERANKER_BASE_URL is empty. Rerank requests may fail and "
+         "fallback to vector recall order."
+     )
```

作用：

- 支持单独配置 reranker 服务。
- 默认可复用 embedding 服务地址和 key。
- 如果地址没配，系统会提醒你可能会回退到向量顺序。

意义：

- 把 reranker 接入做成可配置能力，而不是写死在代码里。
- 运维和部署上更灵活。

---

### 4. 新增 reranker 核心方法

新增的方法包括：

- `_reranker_url()`
- `_parse_rerank_results()`
- `_clone_doc()`
- `_rerank_documents()`

#### 4.1 `_reranker_url()`

```diff
+ def _reranker_url(self) -> str:
+     return f"{self._reranker_base_url}{self._reranker_path}"
```

作用：

- 统一拼接 reranker 请求地址。

#### 4.2 `_parse_rerank_results()`

```diff
+ if isinstance(payload, dict):
+     maybe_rows = (
+         payload.get("results") or payload.get("data") or payload.get("items")
+     )
```

作用：

- 兼容多种服务返回格式。
- 最终统一成 `(index, score)` 列表。

意义：

- 降低对某一个服务返回格式的强耦合。

#### 4.3 `_clone_doc()`

```diff
+ def _clone_doc(doc: Document, metadata: dict) -> Document:
+     return Document(
+         id=doc.id,
+         page_content=doc.page_content,
+         metadata=metadata,
+     )
```

作用：

- 在保留原文内容的同时，写入新的 metadata。
- 避免原始 `Document` 对象被直接污染。

#### 4.4 `_rerank_documents()`

这是本次最核心的新方法。

请求 payload：

```diff
+ payload = {
+     "model": self._reranker_model,
+     "query": question,
+     "documents": [doc.page_content for doc in candidates],
+     "top_n": len(candidates),
+     "return_documents": False,
+ }
```

请求执行：

```diff
+ with httpx.Client(timeout=self._reranker_timeout) as client:
+     response = client.post(
+         self._reranker_url(),
+         headers=headers,
+         json=payload,
+     )
```

结果处理：

```diff
+ rerank_rows = self._parse_rerank_results(result)
+ if not rerank_rows:
+     raise ValueError("reranker returned no parsable ranking rows")
```

写入 rerank 元数据：

```diff
+ metadata["_rerank_score"] = score
+ metadata["_rerank_rank"] = rank
+ metadata["_rerank_fallback"] = False
```

作用：

- 把候选文档交给 reranker 模型重新排序。
- 把 rerank 分数和 rank 写回文档 metadata，便于调试。

意义：

- 最终排序权从“手工规则”变成“语义重排模型”。
- 更适合处理复杂问句、多跳问题、关键词不完全重合的问题。

---

### 5. 增加部分返回和失败回退机制

#### 5.1 部分返回补齐

```diff
+ if len(reranked_docs) < len(candidates):
+     for idx, base_doc in enumerate(candidates):
+         if idx in used_indices:
+             continue
+         metadata["_rerank_score"] = None
+         metadata["_rerank_rank"] = len(reranked_docs) + 1
+         metadata["_rerank_fallback"] = False
+         reranked_docs.append(...)
```

作用：

- 如果 reranker 服务只返回部分候选，不会直接丢掉剩余文档。
- 未返回的文档会按原候选顺序补回去。

意义：

- 保证结果稳定。
- 防止因为服务返回不完整导致排序结果异常缩水。

#### 5.2 异常回退

```diff
+ except Exception as exc:
+     print(
+         f"[WARN] rerank failed ({type(exc).__name__}: {exc}); "
+         "fallback to vector recall order."
+     )
+     ...
+     metadata["_rerank_fallback"] = True
+     return fallback_docs
```

作用：

- reranker 超时、网络失败、返回异常时，不会让检索链路直接报错。
- 会自动退回到向量召回顺序。

意义：

- 提高线上稳定性。
- 这是工程可用性非常关键的一步。

---

### 6. 旧关键词相关方法被标记为废弃

新增注释：

```diff
+ # Deprecated: kept for compatibility docs only; no longer used in retrieval.
```

对应方法：

- `_tokenize_query_for_keyword()`
- `_keyword_score()`

作用：

- 明确告诉后续维护者：这些函数已经不属于主检索路径。
- 现在保留只是为了兼容文档说明或历史参考。

意义：

- 降低误解风险。
- 避免别人继续以为系统还在走关键词加权排序。

---

### 7. `hybrid_retrieve()` 被重写为“向量召回 + rerank”

这部分是主路径改动。

旧逻辑的核心步骤：

```diff
- tokens = self._tokenize_query_for_keyword(question)
- semantic_score = ...
- keyword_score = self._keyword_score(doc.page_content, tokens, question)
- final_score = (
-     self._hybrid_sem_weight * semantic_score
-     + self._hybrid_keyword_weight * keyword_score
- )
- metadata["_semantic_score"] = semantic_score
- metadata["_keyword_score"] = keyword_score
- metadata["_final_score"] = final_score
- ranked_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
- filtered_docs = [
-     d for d in ranked_docs
-     if d.metadata.get("_final_score", 0.0) >= self._min_final_score
- ]
- return filtered_docs[:top_k]
```

新逻辑变成：

```diff
+ candidates: List[Document] = []
+ for doc, distance in docs_and_scores:
+     metadata = dict(doc.metadata or {})
+     metadata.setdefault("source_file", "unknown")
+     metadata.setdefault("page", "unknown")
+     metadata["_vector_distance"] = (
+         float(distance) if distance is not None else None
+     )
+     candidates.append(self._clone_doc(doc, metadata))
+
+ return self._rerank_documents(question=question, candidates=candidates, top_k=top_k)
```

作用：

- 向量检索只负责提供候选集合。
- 最终排序完全交给 reranker。
- 向量距离现在只作为 debug 信息保留，不再参与最终打分。

意义：

- 召回与排序分层。
- 这是更标准、更容易演进的 RAG 结构。

---

### 8. 调试输出改成 reranker 视角

旧输出关注：

```diff
- final_score
- semantic
- keyword
- distance
```

新输出关注：

```diff
+ rerank_rank
+ rerank_score
+ distance
+ fallback
```

作用：

- 更直接观察 reranker 是否在工作。
- 可以看到这次排序是 reranker 排的，还是 fallback 的。

意义：

- 更适合当前架构。
- 也更利于定位线上问题。

---

### 9. Prompt 小幅调整

```diff
- 使用中文回答
+ 使用中文回答，并在关键术语后保留英文原词（如有）。
```

作用：

- 中文回答里尽量保留英文术语。
- 对论文问答、模型名、方法名、benchmark 名更友好。

意义：

- 这不是检索核心改动，但会提升结果可读性和术语准确性。

## 四、这次改动带来的实际效果

从评估结果看，这套代码改动带来了很明确的收益：

- `Recall@K` 不变：说明召回能力没有变差。
- `Top1 Source Hit Rate` 明显提升：说明正确文档更容易排在第一。
- `MRR` 提升：说明正确文档整体排序前移。
- `Precision-like` 提升：说明上下文更干净。
- `Answer Accuracy` 提升：说明排序改进已经传导到最终回答质量。

也就是说，这次 `rag.py` 的改动方向和评估结果是匹配的。

## 五、整体判断

这次 `rag/rag.py` 的改动是一次正确的架构升级，核心价值在于：

1. 把排序从“手工规则”升级到“模型判断”。
2. 把召回和排序拆开，结构更清晰。
3. 加入了回退机制，工程上更稳。
4. 从实际评估结果看，收益已经被验证。

## 六、后续建议

如果后面继续收敛这部分代码，我建议优先做两件事：

1. 在确认不会回滚后，删除真正已经废弃的关键词打分逻辑，减少维护成本。
2. 把 reranker 的调用状态、耗时、fallback 次数纳入评估输出，方便进一步定位问题。
