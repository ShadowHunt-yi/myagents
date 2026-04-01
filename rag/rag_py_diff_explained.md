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
2. 再调用 `reranker` 服务做重排。
3. 如果 `reranker` 失败，则回退到向量召回顺序。

也就是说，这次改动的重点不是“换召回方式”，而是“换最终排序方式”。

## 二、代码改动总览

从代码层面看，主要有 6 类改动：

1. 新增 `reranker` 所需依赖。
2. 废弃旧的手工混合权重参数。
3. 新增 `reranker` 请求、解析和回退逻辑。
4. 重写 `hybrid_retrieve()` 的最终排序路径。
5. 更新调试信息输出。
6. 微调回答 prompt。

## 三、逐段 diff 解释

### 1. 新增 `httpx` 依赖

```diff
+ import httpx
```

作用：

- 用来请求 `reranker` 的 HTTP 接口。
- 说明 `reranker` 不是本地纯函数，而是一个远程模型服务。

### 2. 废弃旧的混合分权参数

旧逻辑里有三类核心参数：

```diff
- self._hybrid_sem_weight = float(os.getenv("HYBRID_SEM_WEIGHT", "0.75"))
- self._hybrid_keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.25"))
- self._min_final_score = float(os.getenv("RETRIEVAL_MIN_FINAL_SCORE", "0.0"))
```

新代码改成兼容保留但不参与排序：

```diff
+ self._deprecated_hybrid_sem_weight = float(os.getenv("HYBRID_SEM_WEIGHT", "0.75"))
+ self._deprecated_hybrid_keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.25"))
+ self._deprecated_min_final_score = float(os.getenv("RETRIEVAL_MIN_FINAL_SCORE", "0.0"))
```

并增加废弃提示。

作用：

- 保留旧环境变量兼容性。
- 明确告诉使用者，当前排序已经不再依赖手工权重。

### 3. 新增 `reranker` 配置

新增初始化参数包括：

- `self._reranker_model`
- `self._reranker_base_url`
- `self._reranker_api_key`
- `self._reranker_timeout`
- `self._reranker_path`

作用：

- 支持独立配置 `reranker` 服务。
- 允许复用现有 embedding 服务地址和 key。

### 4. 新增 `reranker` 核心方法

新增方法包括：

- `_reranker_url()`
- `_parse_rerank_results()`
- `_clone_doc()`
- `_rerank_documents()`

作用：

- 统一请求 `reranker` 服务。
- 解析返回结果。
- 把 rerank 分数和 rank 写进文档 metadata。
- 在失败时自动回退到向量顺序。

### 5. `hybrid_retrieve()` 改为“向量召回 + rerank”

旧逻辑会做：

- 语义分归一化
- 关键词分计算
- 按权重合成最终分数
- 按阈值过滤

新逻辑改为：

- 向量检索只负责召回候选文档
- 最终排序交给 `reranker`
- 向量距离只保留为 debug 信息

这次改动是主路径上的核心变化。

### 6. 调试输出改成 `reranker` 视角

旧输出关注：

- `final_score`
- `semantic`
- `keyword`
- `distance`

新输出关注：

- `rerank_rank`
- `rerank_score`
- `distance`
- `fallback`

作用：

- 更方便判断 `reranker` 是否生效。
- 更方便定位是否走了回退路径。

### 7. Prompt 小幅调整

```diff
- 使用中文回答
+ 使用中文回答，并在关键术语后保留英文原词（如有）。
```

作用：

- 回答里尽量保留英文术语。
- 对论文、模型名、方法名和 benchmark 名更友好。

## 四、这次改动带来的实际效果

从评估结果看，这套代码改动带来了明确收益：

- `Recall@K` 不变，说明召回能力没有变差。
- `Top1 Source Hit Rate` 提升，说明正确文档更容易排到第一。
- `MRR` 提升，说明正确文档整体排序前移。
- `Precision-like` 提升，说明上下文更干净。
- `Answer Accuracy` 提升，说明排序改进已经传导到最终回答质量。

## 五、整体判断

这次 `rag/rag.py` 的改动是一次正确的架构升级，核心价值在于：

1. 把排序从“手工规则”升级到“模型判断”。
2. 把召回和排序拆开，结构更清晰。
3. 加入了回退机制，工程上更稳。
4. 从实际评估结果看，收益已经被验证。

## 六、后续建议

1. 在确认不会回滚后，删除真正已经废弃的关键词打分逻辑，减少维护成本。
2. 把 `reranker` 的调用状态、耗时、fallback 次数纳入评估输出，方便进一步定位问题。
