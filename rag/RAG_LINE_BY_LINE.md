# `rag.py` 逐行注释版（学习手册）

说明：
- 本文按当前文件 `rag/rag.py` 的**行号**解释（以你本地当前版本为准）。
- 采用“逐行讲解（空行略）”方式，重点解释每个函数里的关键行与参数。
- 如果后续你改了 `rag.py`，行号会变化；建议把本文当“结构化导读”使用。

---

## 1) 顶层导入与全局配置（L1-L69）

- `L1-L7`：导入标准库与类型工具，`Any/Callable/List` 供后续类型标注。
- `L9-L11`：设置 BLAS/OMP 线程数为 1，降低本地多线程资源争抢。
- `L14`：把项目根目录加入 `sys.path`，让脚本模式下也能导入上级模块。
- `L16-L25`：导入 RAG 核心依赖：
  - `ChatOpenAI` 负责最终生成
  - `OpenAIEmbeddings` 负责向量化
  - `PGVector` 负责向量存储/检索
  - `Prompt` 和 `document chain` 负责组织上下文
  - `SQLAlchemy` 异常类用于断线重试判断
- `L27`：`load_dotenv()`，让 `.env` 变量进入运行环境。
- `L30`：索引记录文件路径，存放“已索引文件 hash”。
- `L32-L56`：英文停用词集合，用于关键词重排阶段过滤低信息词。
- `L57-L68`：中文停用词集合，避免“的/是/在”等词污染关键词分数。

---

## 2) 辅助函数（L71-L94）

### `_env_bool`（L71-L75）
- `L72`：读取环境变量。
- `L73-L74`：没配置就回退默认值。
- `L75`：把字符串规范化并判断真值集合（`1/true/yes/y/on`）。

### `_file_hash`（L78-L84）
- `L80`：创建 MD5 对象。
- `L81-L83`：二进制分块读文件并累计 hash（避免大文件一次性入内存）。
- `L84`：返回十六进制 hash 字符串。

### `_load_index_record`（L87-L90）
- `L88-L89`：索引文件存在时，按 UTF-8 读取并反序列化为字典。
- `L90`：不存在则返回空字典。

### `_save_index_record`（L93-L94）
- `L94`：把字典写入 JSON，`ensure_ascii=False` 保证中文可读，`indent=2` 便于人工查看。

---

## 3) `RAGApplication.__init__`（L100-L135）

- `L107`：`self.chain` 保留兼容位（当前实际使用 `document_chain`）。
- `L108`：`self.document_chain` 是真正调用的链对象。
- `L109`：存 LLM 客户端实例。
- `L110-L113`：初始化检索相关状态（retriever、collection 名、db url）。
- `L114`：`RETRIEVAL_TOP_K`，最终喂给 LLM 的文档数，默认 `8`。
- `L115`：`RETRIEVAL_FETCH_K`，候选池大小，默认 `40`。
- `L116-L117`：混合分权重：语义分 `0.75`，关键词分 `0.25`。
- `L118`：是否打印检索明细分数（debug）。
- `L119`：最终分最小阈值。
- `L121-L127`：权重归一化与兜底，防止配置非法导致总和为 0。
- `L129-L134`：构造 embedding 模型：
  - `model="BAAI/bge-m3"`：向量模型
  - `chunk_size=64`：批量 embedding 大小，减少 413 风险
- `L135`：构建 `PGVector` 连接（进入 `_build_vector_store`）。

---

## 4) 向量库与重连机制（L137-L193）

### `_build_vector_store`（L137-L154）
- `L140`：`pool_pre_ping=True`，取连接前先探活，避免死连接直接执行 SQL。
- `L141`：`pool_recycle=1800`，连接超过 30 分钟回收重建。
- `L142-L147`：TCP keepalive 参数，降低远程数据库空闲断连概率。
- `L149-L154`：构造 PGVector（embedding、db 连接、collection、engine 参数）。

### `_is_db_connection_error`（L157-L169）
- `L158-L159`：先按异常类型判断（`OperationalError/InterfaceError`）。
- `L161-L168`：再按错误文本兜底匹配（reset/abort/ssl close）。
- `L169`：任一命中则判定“数据库连接问题”。

### `_reconnect_vector_store`（L171-L176）
- `L172`：重建 PGVector 实例。
- `L173-L174`：若 retriever 已存在，按之前 `k` 重新 setup。
- `L175-L176`：若链存在且指定 `rebuild_chain=True`，重建问答链。

### `_run_with_db_retry`（L178-L193）
- `L184`：执行传入函数 `fn`。
- `L186-L188`：非数据库错误直接抛出，不吞异常。
- `L189-L191`：打印警告，说明将重连+重试。
- `L192`：重连向量库/可选重建链。
- `L193`：重试一次 `fn`（只重试一次，避免无限循环）。

---

## 5) 关键词处理与混合检索（L196-L307）

### `_tokenize_query_for_keyword`（L196-L227）
- `L197`：正则抽 token：英文/数字串 + 中文连续串。
- `L198-L199`：初始化输出列表和去重集合。
- `L201-L210`：处理英文 token：
  - 空串跳过
  - 停用词和单字符跳过
  - 其余去重后加入
- `L212-L217`：处理中文 token：
  - 停用词和单字跳过
  - 合法中文词去重后加入
- `L218-L225`：中文 2-gram 扩展（例如“智能体”生成“智能”“能体”）。
- `L227`：返回 token 列表。

### `_keyword_score`（L230-L245）
- `L231-L232`：无 token 直接返回 0。
- `L234`：文本小写化，便于大小写无关匹配。
- `L235`：统计命中 token 个数。
- `L236`：基础覆盖率分 `hits / len(tokens)`。
- `L238`：总词频 `tf`（命中 token 出现次数总和）。
- `L239`：词频加分，上限 `0.2`。
- `L241-L243`：若问题整句出现在段落中，再加 `0.1`。
- `L245`：clip 到 `[0,1]`。

### `hybrid_retrieve`（L247-L307）
- `L254`：确定最终返回数量 `top_k`。
- `L255`：候选数 `candidate_k = max(top_k, fetch_k)`。
- `L257-L262`：先向量召回候选（含原始距离分），且包在 DB 重试器里。
- `L263-L264`：无候选直接返回空。
- `L266`：提取 query 关键词 token。
- `L267-L270`：计算距离归一化所需的 `min/max/denom`。
- `L272`：准备承载重排后的文档列表。
- `L273-L281`：逐条候选计算：
  - `distance_value`（原始距离）
  - `semantic_score`（距离归一化后反转，越近越高）
  - `keyword_score`
  - `final_score = sem_weight * semantic + keyword_weight * keyword`
- `L283-L289`：写入调试元数据（source/page/各分项得分）。
- `L291-L297`：重建 `Document` 对象，保留 id/content/metadata。
- `L299`：按 `final_score` 降序排序。
- `L300-L304`：根据 `RETRIEVAL_MIN_FINAL_SCORE` 做阈值过滤。
- `L305-L307`：
  - 过滤后够 `top_k`：返回过滤结果前 `top_k`
  - 否则：回退到未过滤排序前 `top_k`

---

## 6) 按文档分组（L310-L324）

### `_group_docs_by_source`
- `L311-L312`：`groups` 存每个文档的 chunk 列表，`order` 保存首次出现顺序。
- `L314-L319`：按 `source_file` 聚合。
- `L321-L323`：按出现顺序再展开，返回分组后序列。

为什么要这一步：
- 让 LLM 在输入中先看到同一文档的连续片段，更容易按文档输出。

---

## 7) 文档加载/切分/入库（L326-L372）

### `load_documents`（L326-L339）
- `L328-L331`：PDF 走 `PyPDFLoader`，其他文本走 `TextLoader(utf-8)`。
- `L333`：执行 `loader.load()`。
- `L335-L336`：给每个文档块写 `source_file` 元数据。
- `L338-L339`：打印并返回文档列表。

### `split_documents`（L341-L351）
- `L343-L348`：切分参数：
  - `chunk_size=1000`
  - `chunk_overlap=200`
  - 分隔符优先级：段落 > 换行 > 中文句号 > 空格 > 字符级兜底
- `L349`：执行切分。
- `L350-L351`：打印并返回 chunks。

### `create_vectorstore`（L353-L372）
- `L355-L356`：准备清洗后的 chunk 列表和计数器。
- `L357-L365`：逐 chunk 清洗：
  - 去除 `\x00`
  - 空白内容剔除
  - 合法 chunk 进入 `cleaned`
- `L366-L369`：入库动作放在 `_run_with_db_retry` 里。
- `L370-L372`：打印本次写入统计。

---

## 8) 清库与 retriever 兼容接口（L374-L395）

### `clear_collection`（L374-L382）
- `L376`：删除 PGVector collection。
- `L378`：重建空 collection。
- `L380-L381`：删除 `.indexed_files.json`，让下次可全量重建索引。
- `L382`：打印提示。

### `setup_retriever`（L384-L395）
- `L386`：记录当前 `k`，用于断连后恢复。
- `L387-L394`：构造 MMR retriever（当前主流程已转向 `hybrid_retrieve`，这里主要保留兼容和备用）。

---

## 9) Debug 检索与问答链（L397-L471）

### `debug_retrieval`（L397-L423）
- `L398-L402`：调用 `hybrid_retrieve` 拿最终 top-k。
- `L403`：打印召回数量。
- `L405-L407`：读取 `source_file/page`。
- `L408-L417`：若开启 debug 分数，打印 `final/semantic/keyword/distance`。
- `L419`：否则只打印基础信息。
- `L420`：打印内容预览（前 500 字符）。
- `L423`：返回 docs，供下游直接复用。

### `create_chain`（L424-L457）
- `L426-L444`：构建系统提示词 + 用户消息模板。
- `L445-L447`：定义每条文档渲染格式（带文档名和页码）。
- `L449-L454`：创建 `stuff documents chain`。
- `L455`：兼容赋值给 `self.chain`。
- `L456`：打印创建成功。

### `query`（L458-L471）
- `L460-L462`：链未初始化时抛异常。
- `L463-L467`：如果调用方未传 docs，则内部触发混合检索。
- `L468`：按文档分组排序。
- `L470`：调用 LLM 链生成答案。
- `L471`：返回 `answer + context`。

---

## 10) CLI 主循环（L473-L514）

### `chat_loop`
- `L477-L479`：打印启动 banner。
- `L482`：读取用户问题。
- `L484-L486`：输入 `exit/quit/q` 退出。
- `L488-L489`：空输入跳过。
- `L491-L495`：
  - 先 `debug_retrieval`
  - 再 `query`
  - 打印链路成功
- `L497-L507`：输出答案与来源片段（含文件名和页码）。
- `L509-L513`：异常时打印类型、消息、完整 traceback。

---

## 11) 程序入口 `main`（L516-L570）

- `L518-L520`：读取 LLM 配置，提供默认值。
- `L522-L523`：没 API key 时交互式输入。
- `L524`：创建 `ChatOpenAI` 客户端。
- `L525-L527`：读取数据库与 embedding 配置。
- `L529`：实例化 `RAGApplication`。
- `L532-L533`：加载索引记录，准备待索引列表。
- `L535-L541`：收集候选文件（`papers/*.pdf` + 可选 `sample.txt`）。
- `L543-L550`：按 hash 判断是否变更：
  - 未变更：跳过
  - 新增/变更：加入索引队列并更新记录
- `L551-L560`：有变更就逐文件执行：
  - 加载
  - 切分
  - 入向量库
  - 保存索引记录
- `L563-L564`：设置 retriever 并创建问答链。
- `L567`：启动交互问答循环。
- `L569-L570`：脚本入口保护。

---

## 12) 参数总览（你调优最常用）

- `RETRIEVAL_TOP_K`：最终给 LLM 的文档数，默认 `8`。
- `RETRIEVAL_FETCH_K`：候选池大小，默认 `40`。
- `HYBRID_SEM_WEIGHT`：语义分权重，默认 `0.75`。
- `HYBRID_KEYWORD_WEIGHT`：关键词分权重，默认 `0.25`。
- `RETRIEVAL_MIN_FINAL_SCORE`：最终分阈值，默认 `0`。
- `RETRIEVAL_DEBUG_SCORES`：是否输出分数明细，默认 `true`。

---

## 13) 学习顺序建议

1. 先读 `main -> chat_loop -> query`，理解主流程。
2. 再读 `hybrid_retrieve`，理解为什么比纯向量检索更稳。
3. 最后读 `_run_with_db_retry`，理解工程稳定性处理。

如果你愿意，我下一步可以继续生成一份 `rag/rag_annotated.py`（在代码里直接插注释版，和这份 MD 配套）。

