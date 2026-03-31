"""
RAG Annotated Copy (Detailed)
=============================

这个文件是 `rag.py` 的“注释增强副本（教学版）”。

设计原则：
1) 完全保持业务逻辑一致，不引入行为差异；
2) 用“代码贴脸”的方式解释关键实现；
3) 重点解释“为什么要这样做”，而不仅是“做了什么”。

推荐阅读顺序：
1) 先看 `main()` 把握系统入口；
2) 再看 `RAGApplication.hybrid_retrieve()` 理解检索质量控制；
3) 最后看 `create_chain()/query()` 理解生成阶段。

配套文档：
- `rag/RAG_LINE_BY_LINE.md`：逐行文字版导读
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, List

# 限制底层数值库线程，降低本地调试时 CPU 争用与延迟抖动。
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"

# 允许脚本从 rag/ 目录运行时导入项目根模块。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_postgres import PGVector
from dotenv import load_dotenv
from sqlalchemy.exc import InterfaceError as SAInterfaceError
from sqlalchemy.exc import OperationalError as SAOperationalError

load_dotenv()
RERANKERMODEL = os.getenv("RERANKERMODEL", "BAAI/bge-reranker-v2-m3")
EMBEDDINGMODEL = os.getenv("EMBEDDINGMODEL", "BAAI/bge-m3")

# 增量索引记录文件：记录每个源文件的哈希，避免重复 embedding 写库。
INDEX_RECORD_PATH = Path(__file__).parent / ".indexed_files.json"

# 英文停用词：用于关键词重排阶段，过滤低信息量词。
EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "be",
    "as",
    "that",
    "this",
    "it",
    "from",
    "how",
    "what",
    "which",
}

# 中文停用词：同样用于关键词重排。
CN_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "中",
    "请",
    "关于",
    "进行",
    "分析",
}


def _env_bool(name: str, default: bool) -> bool:
    """
    读取布尔环境变量。

    Args:
        name: 环境变量名，例如 `RETRIEVAL_DEBUG_SCORES`。
        default: 当变量缺失时使用的默认值。

    Returns:
        bool: 解析后的布尔值。

    Why:
        统一处理环境变量布尔解析，避免每个配置位都重复写解析逻辑。
    """
    # 1) 从环境变量读取原始字符串，例如 "true"/"0"/"yes"。
    value = os.getenv(name)
    if value is None:
        # 2) 未配置时直接返回调用方提供的默认值。
        return default
    # 3) 统一做空白清理和小写化，再映射到“真值集合”。
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _file_hash(file_path: str) -> str:
    """
    计算文件 MD5，用于增量索引判定。

    Args:
        file_path: 待计算哈希的文件路径。

    Returns:
        str: 文件的 MD5 十六进制字符串。

    Why:
        RAG 的 embedding 写库通常耗时/耗费调用额度，只有文件变化时才应重建索引。
    """
    # 1) 创建 MD5 哈希器（这里只用于“文件变更检测”，不用于安全场景）。
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        # 2) 分块读取，避免一次性把大文件全读入内存。
        for chunk in iter(lambda: f.read(8192), b""):
            # 3) 持续增量更新哈希状态。
            h.update(chunk)
    # 4) 返回十六进制字符串，便于写入 JSON 与比较。
    return h.hexdigest()


def _load_index_record() -> dict:
    """
    读取索引记录（路径 -> hash）。

    Returns:
        dict: 已索引记录；若文件不存在则返回空字典。
    """
    # 1) 若记录文件存在，则读取并反序列化为 dict。
    if INDEX_RECORD_PATH.exists():
        return json.loads(INDEX_RECORD_PATH.read_text(encoding="utf-8"))
    # 2) 首次运行或记录被清理时，返回空状态。
    return {}


def _save_index_record(record: dict):
    """
    持久化索引记录。

    Args:
        record: 索引记录字典。

    Why:
        让“增量索引状态”跨进程持久化，下一次启动仍能跳过未变更文件。
    """
    # 1) 把“路径->hash”映射写成易读 JSON（缩进+保留中文）。
    INDEX_RECORD_PATH.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class RAGApplication:
    """
    一个可交互的 RAG 应用：
    - 负责文档切分与向量写入；
    - 负责混合检索（语义 + 关键词）；
    - 负责将检索结果组织后交给 LLM 生成答案。
    """

    def __init__(
        self,
        llm_client: ChatOpenAI,
        embeddings_key: str,
        embeddings_url: str,
        db_url: str,
    ):
        """
        初始化 RAG 运行时。

        Args:
            llm_client: 生成答案用的聊天模型客户端。
            embeddings_key: embedding 服务 API key。
            embeddings_url: embedding 服务 base url。
            db_url: PostgreSQL 连接串。

        配置来源（.env）：
            RETRIEVAL_TOP_K
            RETRIEVAL_FETCH_K
            HYBRID_SEM_WEIGHT
            HYBRID_KEYWORD_WEIGHT
            RETRIEVAL_MIN_FINAL_SCORE
            RETRIEVAL_DEBUG_SCORES

        Why:
            把所有检索行为参数集中在初始化阶段读取，便于调参与排查。
        """
        # 兼容旧字段：当前实际主要使用 document_chain。
        # chain 与 document_chain 并存是为了兼容旧调用路径；当前主用 document_chain。
        self.chain = None
        self.document_chain = None
        self.llm = llm_client
        self.retriever = None
        # retriever 默认返回条数（若后续 setup_retriever 未覆写）。
        self._retriever_k = 6
        # PGVector 的 collection 名称，等价于“逻辑索引名”。
        self._collection_name = "documents"
        # 数据库连接串缓存，供重连时复用。
        self._db_url = db_url

        # 检索参数（可通过 .env 调整）。
        # 最终交给 LLM 的上下文条数上限。
        self._retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
        # 向量召回候选池大小，通常应 >= top_k。
        self._retrieval_fetch_k = int(os.getenv("RETRIEVAL_FETCH_K", "40"))
        # 混合重排中“语义分”权重。
        self._hybrid_sem_weight = float(os.getenv("HYBRID_SEM_WEIGHT", "0.75"))
        # 混合重排中“关键词分”权重。
        self._hybrid_keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.25"))
        # 是否打印每条候选的打分细节（便于调参与排障）。
        self._retrieval_debug_scores = _env_bool("RETRIEVAL_DEBUG_SCORES", True)
        # 最终阈值：低于该分数的候选可被过滤。
        self._min_final_score = float(os.getenv("RETRIEVAL_MIN_FINAL_SCORE", "0.0"))

        # 归一化权重，避免配置异常（如总和为 0）。
        total_weight = self._hybrid_sem_weight + self._hybrid_keyword_weight
        if total_weight <= 0:
            self._hybrid_sem_weight = 0.75
            self._hybrid_keyword_weight = 0.25
        else:
            self._hybrid_sem_weight /= total_weight
            self._hybrid_keyword_weight /= total_weight

        # embedding 客户端；chunk_size 避免批量过大导致 413。
        # embedding 客户端：负责把文本转成向量写入/查询 PGVector。
        self.embeddings = OpenAIEmbeddings(
            api_key=embeddings_key,
            model=EMBEDDINGMODEL,
            base_url=embeddings_url,
            chunk_size=64,
        )
        # 初始化向量存储实例（含连接池与保活参数）。
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self) -> PGVector:
        """
        构建 PGVector。
        关键点：打开连接保活和连接池探活，降低远程数据库断连概率。

        Returns:
            PGVector: 配置好的向量库实例。

        参数解释：
            pool_pre_ping=True:
                每次借出连接前先 ping，坏连接会被自动回收。
            pool_recycle=1800:
                连接超过 30 分钟自动回收，防止服务端空闲断开导致“僵尸连接”。
            keepalives*:
                TCP 保活参数，降低跨网络场景中的静默断连。
        """
        # 1) SQLAlchemy 引擎参数：连接池探活 + 周期回收 + TCP keepalive。
        engine_args = {
            "pool_pre_ping": True,
            "pool_recycle": 1800,
            "connect_args": {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        }
        # 2) 构建 PGVector 实例，并绑定 embedding 模型与 collection 名。
        return PGVector(
            embeddings=self.embeddings,
            connection=self._db_url,
            collection_name=self._collection_name,
            engine_args=engine_args,
        )

    @staticmethod
    def _is_db_connection_error(exc: Exception) -> bool:
        """
        识别数据库连接类错误（类型 + 文本双重判断）。

        Args:
            exc: 捕获到的异常对象。

        Returns:
            bool: 是否可判定为“连接问题”。

        Why:
            某些驱动/部署环境下，异常类型不稳定，仅靠类型判断不够稳；
            增加文本特征可提升断连识别召回率。
        """
        # 1) 先按异常类型判断（最快、最准确）。
        if isinstance(exc, (SAOperationalError, SAInterfaceError)):
            return True

        # 2) 再按错误文本兜底，兼容不同驱动/网关返回差异。
        text = str(exc).lower()
        markers = (
            "software caused connection abort",
            "could not receive data from server",
            "server closed the connection unexpectedly",
            "connection reset by peer",
            "ssl connection has been closed unexpectedly",
        )
        # 3) 任一 marker 命中都视为“可重试的连接错误”。
        return any(marker in text for marker in markers)

    def _reconnect_vector_store(self, rebuild_chain: bool = False):
        """
        重连向量库，并在需要时恢复 retriever / chain。

        Args:
            rebuild_chain: 是否在重连后重建问答链。

        Why:
            断连后不仅 vector_store 失效，部分依赖对象可能持有旧连接状态；
            因此这里把 retriever/chain 的恢复放在同一处集中处理。
        """
        # 1) 重建 vector_store（拿到全新连接与连接池状态）。
        self.vector_store = self._build_vector_store()
        # 2) 若 retriever 已初始化，则按原 k 重新挂载到新 store。
        if self.retriever is not None:
            self.setup_retriever(k=self._retriever_k)
        # 3) 某些场景需要 chain 也指向新 retriever / store 依赖。
        if rebuild_chain and self.chain is not None and self.retriever is not None:
            self.create_chain()

    def _run_with_db_retry(
        self,
        fn: Callable[[], Any],
        op_name: str,
        rebuild_chain: bool = False,
    ) -> Any:
        """
        执行数据库相关动作；若断连则自动重连后重试一次。

        Args:
            fn: 需要执行的数据库动作（闭包形式）。
            op_name: 操作名，用于日志定位。
            rebuild_chain: 重试前是否重建 chain。

        Returns:
            Any: `fn` 的返回结果。

        设计取舍：
            - 重试一次：避免无限重试掩盖真实故障。
            - 非连接错误直接抛出：避免误吞业务错误。
        """
        # 1) 先执行一次真实操作。
        try:
            return fn()
        except Exception as exc:
            # 2) 非连接错误：直接抛出，不吞业务异常。
            if not self._is_db_connection_error(exc):
                raise
            # 3) 连接错误：重连后仅重试一次，防止无限重试。
            print(
                f"[WARN] {op_name} failed due to DB disconnect, reconnecting and retrying once..."
            )
            self._reconnect_vector_store(rebuild_chain=rebuild_chain)
            return fn()

    @staticmethod
    def _tokenize_query_for_keyword(question: str) -> List[str]:
        """
        提取关键词 token（中英混合）：
        - 英文按词提取并停用词过滤；
        - 中文按连续字串提取，并扩展 2-gram 以增强匹配能力。

        Args:
            question: 用户原始问题文本。

        Returns:
            List[str]: 去重后的关键词 token 列表。

        Why:
            关键词重排阶段需要“高区分度词”而非全部词；
            中文 2-gram 是轻量且有效的增强手段。
        """
        # 1) 统一小写后提取 token：
        # - 英文/数字/下划线连续串
        # - 中文连续串
        raw_tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", question.lower())
        tokens: List[str] = []
        # 用于去重，保持 tokens 顺序稳定。
        seen = set()

        for tok in raw_tokens:
            # 跳过空 token。
            if not tok.strip():
                continue
            # 英文 token 分支。
            if re.fullmatch(r"[A-Za-z0-9_]+", tok):
                # 停用词或单字符直接过滤。
                if tok in EN_STOPWORDS or len(tok) <= 1:
                    continue
                # 仅保留首次出现，避免重复计分。
                if tok not in seen:
                    seen.add(tok)
                    tokens.append(tok)
                continue

            # 中文 token 分支：先做停用词/长度过滤。
            if tok in CN_STOPWORDS or len(tok) <= 1:
                continue
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)

            # 对长度>2的中文串追加 2-gram（如“向量检索” -> “向量”“量检”“检索”）。
            # 这样对中文术语局部匹配更友好。
            if len(tok) > 2:
                for i in range(len(tok) - 1):
                    bigram = tok[i : i + 2]
                    if bigram in CN_STOPWORDS:
                        continue
                    if bigram not in seen:
                        seen.add(bigram)
                        tokens.append(bigram)

        # 返回去重后的关键词序列。
        return tokens

    @staticmethod
    def _keyword_score(content: str, tokens: List[str], question: str) -> float:
        """
        关键词分：
        - 命中覆盖率；
        - 词频增益；
        - 问句整串命中加分；
        最终限制在 [0, 1]。

        Args:
            content: 候选文档片段文本。
            tokens: 从 query 提取出的关键词集合。
            question: 原始问题文本。

        Returns:
            float: 关键词相关度分，范围 [0, 1]。

        Why:
            语义向量能抓“语义近似”，关键词分用于补充“术语精确命中”能力。
        """
        # 1) 无 token 时关键词分直接为 0。
        if not tokens:
            return 0.0

        # 2) 文本统一小写，做大小写无关匹配。
        text = content.lower()
        # 3) 覆盖率：命中的 token 数 / 总 token 数。
        hits = sum(1 for token in tokens if token in text)
        score = hits / len(tokens)

        # 4) 词频增益：命中总次数越多，额外加分越高（上限 0.2）。
        tf = sum(text.count(token) for token in tokens)
        score += min(tf / 20.0, 0.2)

        # 5) 问句整串命中加分：对“几乎原文复现”的高相关片段提高排序。
        normalized_question = question.strip().lower()
        if (
            normalized_question
            and len(normalized_question) >= 4
            and normalized_question in text
        ):
            score += 0.1

        # 6) 最终夹到 [0, 1] 区间，便于和 semantic_score 融合。
        return max(0.0, min(1.0, score))

    def hybrid_retrieve(
        self,
        question: str,
        k: int | None = None,
        fetch_k: int | None = None,
        filter: dict | None = None,
    ) -> List[Document]:
        """
        混合检索主流程：
        1) 先用向量检索拿候选；
        2) 计算语义分 + 关键词分；
        3) 线性融合后排序返回 top-k。

        Args:
            question: 用户问题。
            k: 最终返回条数；为空则用 `RETRIEVAL_TOP_K`。
            fetch_k: 候选池条数；为空则用 `RETRIEVAL_FETCH_K`。
            filter: 可选元数据过滤条件（透传给 PGVector）。

        Returns:
            List[Document]: 已重排后的文档列表。

        实现细节：
            - 先拿 `candidate_k` 候选，再重排，避免“直接 top-k”丢掉潜在高质量候选；
            - 语义分来自距离归一化反转，关键词分来自 token 匹配；
            - 最终分写回 metadata，便于 debug 和离线评测。
        """
        # A. 决定“输出条数”和“候选池大小”：
        # top_k 是最终返回给 LLM 的条数；
        # candidate_k 是先召回再重排的池子，越大越不容易错过潜在好结果。
        top_k = k or self._retrieval_top_k
        candidate_k = max(top_k, fetch_k or self._retrieval_fetch_k)

        # B. 从向量库拿回 (Document, distance) 列表。
        # 这里 distance 越小越相关，后面会转成“越大越好”的 semantic_score。
        docs_and_scores = self._run_with_db_retry(
            lambda: self.vector_store.similarity_search_with_score(
                question, k=candidate_k, filter=filter
            ),
            op_name="hybrid retrieval candidates",
        )
        if not docs_and_scores:
            # 召回为空直接返回，避免后续 min/max 计算报错。
            return []

        # 1) 关键词提取：为后续 keyword_score 准备输入。
        tokens = self._tokenize_query_for_keyword(question)

        # 2) 距离统计：用于把不同查询下的原始距离归一化到可比较区间。
        distances = [float(score) for _, score in docs_and_scores if score is not None]
        min_distance = min(distances) if distances else 0.0
        max_distance = max(distances) if distances else 1.0
        # 保护项：当 max==min 时避免除零。
        denom = max(max_distance - min_distance, 1e-12)

        ranked_docs: List[Document] = []
        for doc, distance in docs_and_scores:
            distance_value = float(distance) if distance is not None else max_distance

            # 距离越小越相关；归一化后反转得到 semantic_score（越大越好）。
            semantic_score = 1.0 - ((distance_value - min_distance) / denom)
            semantic_score = max(0.0, min(1.0, semantic_score))

            # 关键词分与语义分进行融合，兼顾“语义近似”和“术语精确”。
            keyword_score = self._keyword_score(doc.page_content, tokens, question)
            # 线性融合：这一步是“混合检索”核心。
            final_score = (
                self._hybrid_sem_weight * semantic_score
                + self._hybrid_keyword_weight * keyword_score
            )

            metadata = dict(doc.metadata or {})
            metadata.setdefault("source_file", "unknown")
            metadata.setdefault("page", "unknown")
            metadata["_vector_distance"] = distance_value
            metadata["_semantic_score"] = semantic_score
            metadata["_keyword_score"] = keyword_score
            metadata["_final_score"] = final_score

            # 复制成新 Document 返回，而不是原地改写，减少副作用。
            ranked_docs.append(
                Document(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        # C. 按最终融合分降序排序，得到重排结果。
        ranked_docs.sort(
            key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True
        )

        # 可选阈值过滤：在噪声较高语料场景中常见且有效。
        filtered_docs = [
            doc
            for doc in ranked_docs
            if doc.metadata.get("_final_score", 0.0) >= self._min_final_score
        ]
        # D. 若阈值过滤后仍足够 top_k，则优先返回过滤结果；
        # 否则退回 ranked_docs，避免“阈值过严导致结果太少”。
        if len(filtered_docs) >= top_k:
            return filtered_docs[:top_k]
        return ranked_docs[:top_k]

    @staticmethod
    def _group_docs_by_source(docs: List[Document]) -> List[Document]:
        """
        按 `source_file` 聚合并展开，便于模型按文档块阅读。

        Args:
            docs: 检索到的文档片段列表。

        Returns:
            List[Document]: 按文档聚类后的文档顺序。

        Why:
            对“综合分析”任务来说，先按文档归拢上下文，模型更容易输出分文档结构。
        """
        # 1) source_file -> 文档片段列表
        groups: dict[str, List[Document]] = {}
        # 2) 记录 source 首次出现顺序，保证输出稳定。
        order: List[str] = []

        for doc in docs:
            source = doc.metadata.get("source_file", "unknown")
            if source not in groups:
                groups[source] = []
                order.append(source)
            groups[source].append(doc)

        # 3) 按首次出现顺序拍平，形成“按文档分组但仍是线性列表”的结果。
        grouped_docs: List[Document] = []
        for source in order:
            grouped_docs.extend(groups[source])
        return grouped_docs

    def load_documents(self, file_path: str) -> List:
        """
        加载 PDF/TXT 文档并补充 `source_file` 元数据。

        Args:
            file_path: 文档路径。

        Returns:
            List: LangChain 文档对象列表。
        """
        # 1) 按后缀选择加载器：PDF 或纯文本。
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        # 2) 执行加载，得到 LangChain Document 列表。
        documents = loader.load()
        # 3) 补充 source_file 元数据，供后续溯源与分组使用。
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(file_path)

        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents

    def split_documents(self, documents: List) -> List:
        """
        文档切分：
        - chunk_size 控制每段长度；
        - chunk_overlap 保留上下文衔接；
        - separators 按“段落 -> 行 -> 句 -> 词 -> 字符”优先切。

        Args:
            documents: 原始文档列表。

        Returns:
            List: 分块后的文档片段列表。

        参数解释：
            chunk_size=1000:
                平衡“上下文完整性”和“检索粒度”。
            chunk_overlap=200:
                避免跨块语义断裂导致召回丢信息。
        """
        # 1) 配置递归切分器：
        # - 优先按段落、换行、句号切，再退化到空格/字符。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", " ", ""],
        )
        # 2) 执行切块。
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List):
        """
        清洗 chunk 后写入向量库。

        Args:
            chunks: 文档分块结果。

        Why:
            在写库前做基础清洗（去空、去异常字符）可显著减少下游检索噪声。
        """
        # 1) 入库前清洗：去掉空块、异常空字符块。
        cleaned = []
        removed = 0
        for doc in chunks:
            text = getattr(doc, "page_content", "")
            if "\x00" in text:
                text = text.replace("\x00", "")
            if not text.strip():
                removed += 1
                continue
            doc.page_content = text
            cleaned.append(doc)

        # 2) 执行写库（带断连重试）。
        self._run_with_db_retry(
            lambda: self.vector_store.add_documents(cleaned),
            op_name="vectorstore add_documents",
        )
        print(
            f"Vectorstore: {len(cleaned)} chunks added (removed {removed} empty/invalid)"
        )

    def clear_collection(self):
        """
        清空向量库 collection，并重置本地索引记录。

        使用场景：
            - 语料大规模更新；
            - 检索策略升级后希望全量重建。
        """
        # 1) 清空当前 collection。
        self.vector_store.delete_collection()
        # 2) 立即重建 vector_store，避免后续对象持有失效状态。
        self.vector_store = self._build_vector_store()
        # 3) 删除本地增量记录，确保下次全量重建。
        if INDEX_RECORD_PATH.exists():
            INDEX_RECORD_PATH.unlink()
        print("向量数据库已清空，索引记录已重置。下次启动会重新索引所有文件。")

    def setup_retriever(self, k: int = 6):
        """
        配置 MMR retriever（兼容路径）。

        Args:
            k: 返回条数。

        Note:
            当前主流程主要走 `hybrid_retrieve`，这里保留用于兼容与实验切换。
        """
        # 1) 记录当前 k，供重连恢复使用。
        self._retriever_k = k
        # 2) 初始化 MMR 检索器：
        # - fetch_k 先拿更大候选池
        # - lambda_mult 控制“相关性 vs 多样性”
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(20, self._retrieval_fetch_k),
                "lambda_mult": 0.5,
            },
        )
        print(f"Retriever setup with MMR, k={k}")

    def debug_retrieval(self, question: str, k: int | None = None) -> List[Document]:
        """
        打印检索结果与分数明细，便于调参。

        Args:
            question: 用户问题。
            k: 期望返回条数，默认沿用配置。

        Returns:
            List[Document]: 检索文档列表（可直接复用到 query，避免二次检索）。
        """
        # 1) 执行混合检索，拿到最终重排结果。
        docs = self.hybrid_retrieve(
            question=question,
            k=k or self._retrieval_top_k,
            fetch_k=self._retrieval_fetch_k,
        )
        print(f"\n[DEBUG] Retrieved {len(docs)} docs\n")

        # 2) 打印每个文档片段与分数字段，支持可解释调参。
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "unknown")
            if self._retrieval_debug_scores:
                final_score = doc.metadata.get("_final_score", 0.0)
                semantic = doc.metadata.get("_semantic_score", 0.0)
                keyword = doc.metadata.get("_keyword_score", 0.0)
                distance = doc.metadata.get("_vector_distance", 0.0)
                print(
                    f"--- DOC {i} | file={source} | page={page} | "
                    f"final={final_score:.4f} semantic={semantic:.4f} "
                    f"keyword={keyword:.4f} distance={distance:.4f} ---"
                )
            else:
                print(f"--- DOC {i} | file={source} | page={page} ---")
            # 只展示前 500 字，防止控制台刷屏。
            print(doc.page_content[:500])
            print()
        return docs

    def create_chain(self):
        """
        构建文档问答链：
        - 把检索结果按模板串成 context；
        - 用系统提示词约束输出为“分文档总结 + 综合结论”。

        Why:
            仅靠检索质量不够，提示词结构也会显著影响输出可读性与稳定性。
        """
        # 1) 系统提示词：约束“仅依据参考资料”并要求“分文档总结 + 综合结论”。
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的论文分析助手。请根据提供的参考资料回答用户问题。

规则：
- 只根据参考资料中的内容回答，不要编造信息。
- 回答时先按文档分模块总结，格式为“在 <文档名> 中，...”
- 至少覆盖检索到的主要文档观点，再给出“综合结论”小节。
- 如果某文档证据不足，请明确写“该文档证据不足”。
- 使用中文回答

参考资料：
{context}""",
                ),
                ("human", "{input}"),
            ]
        )

        # 2) 每个 Document 的渲染模板：显式附带 source_file/page 便于引用。
        document_prompt = PromptTemplate.from_template(
            "[文档: {source_file} | 页码: {page}]\n{page_content}"
        )

        # 3) 构建 stuff chain：把检索文档拼接成 context 后一次性喂给模型。
        self.document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=document_prompt,
            document_separator="\n\n---\n\n",
        )
        # 兼容旧字段。
        self.chain = self.document_chain
        print("Document QA chain created successfully")

    def query(self, question: str, docs: List[Document] | None = None) -> dict:
        """
        执行一次问答；可复用已检索 docs，避免重复检索。

        Args:
            question: 用户问题。
            docs: 可选预检索文档，通常来自 `debug_retrieval`。

        Returns:
            dict: `{"answer": str, "context": List[Document]}`

        Why:
            将“检索”与“生成”解耦后，既方便调试，也便于后续挂接评测。
        """
        # chain 必须先构建，否则无法把 context 送入 LLM 生成答案。
        if not self.document_chain:
            raise ValueError("Chain not initialized. Call create_chain() first.")

        # 支持“外部已检索 docs 复用”：
        # 调试模式下通常先 debug_retrieval，再 query，这里可避免重复检索。
        retrieved_docs = docs or self.hybrid_retrieve(
            question=question,
            k=self._retrieval_top_k,
            fetch_k=self._retrieval_fetch_k,
        )
        # 将片段按 source_file 聚拢，提升模型输出结构化总结的稳定性。
        grouped_docs = self._group_docs_by_source(retrieved_docs)

        # 调用文档链：input 是用户问题，context 是重排后的参考材料。
        answer = self.document_chain.invoke(
            {"input": question, "context": grouped_docs}
        )
        # answer + context 一起返回，便于前端展示溯源。
        return {"answer": answer, "context": retrieved_docs}

    def chat_loop(self):
        """
        CLI 交互循环。

        交互路径：
            读取问题 -> debug 检索 -> query 生成 -> 输出答案与来源。
        """
        # 局部导入 traceback，仅在 CLI 调试场景使用。
        import traceback

        print("\n" + "=" * 50)
        print("RAG Application Ready! Type 'exit' to quit.")
        print("=" * 50 + "\n")

        while True:
            # 1) 读取用户输入。
            question = input("Question: ").strip()

            # 2) 退出指令。
            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # 3) 空输入直接继续下一轮。
            if not question:
                continue

            try:
                # 4) 先 debug 检索（可见召回质量），再执行 query 生成答案。
                print("[DEBUG] invoking chain...")
                retrieved_docs = self.debug_retrieval(question, k=self._retrieval_top_k)
                response = self.query(question, docs=retrieved_docs)
                print("[DEBUG] chain invoke success")

                answer = response.get("answer", "No answer returned")
                context = response.get("context", [])

                # 5) 输出答案与溯源片段。
                print(f"\nAnswer: {answer}")
                print(f"\nSources ({len(context)} documents retrieved):")
                for i, doc in enumerate(context, 1):
                    content = getattr(doc, "page_content", "")
                    source = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "unknown")
                    print(f"  {i}. [file={source}, page={page}] {content[:300]}...")
                print()

            except Exception as exc:
                # 6) 任何异常都打印类型、消息和堆栈，便于排查。
                print(f"\n[ERROR TYPE] {type(exc).__name__}")
                print(f"[ERROR MSG] {exc}\n")
                traceback.print_exc()
                print()


def main():
    """
    启动入口：加载配置、增量索引、创建链、进入交互。

    整体流程：
        1) 初始化 LLM + Embedding + PGVector
        2) 基于 hash 做增量索引
        3) 配置检索器并创建生成链
        4) 进入交互式问答
    """
    # 1) 读取 LLM 相关配置。
    api_key = os.getenv("LONGCAT_API_KEY")
    model = os.getenv("LONGCAT_MODEL", "gpt-5.2-codex")
    base_url = os.getenv("LONGCAT_BASE_URL", "http://localhost:11434")

    if not api_key:
        # 未配置时交互式输入，方便本地快速试跑。
        api_key = input("Enter your OpenAI API key: ")

    # 2) 初始化生成模型客户端。
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
    # 3) 读取向量库与 embedding 服务配置。
    db_url = os.getenv("DATABASE_URL")
    embeddings_key = os.getenv("SILICONFLOW_API_KEY")
    embeddings_url = os.getenv("SILICONFLOW_BASE_URL")

    # 4) 构建 RAG 应用对象（内部会初始化 vector_store）。
    rag = RAGApplication(llm, embeddings_key, embeddings_url, db_url)

    # 增量索引：只处理新文件或内容发生变化的文件。
    index_record = _load_index_record()
    files_to_index: list[str] = []

    papers_dir = Path(__file__).parent / "papers"
    pdf_files = sorted(papers_dir.glob("*.pdf")) if papers_dir.exists() else []
    sample_path = Path(__file__).parent / "sample.txt"

    # 候选语料 = papers/*.pdf + 可选 sample.txt
    candidates = [str(path) for path in pdf_files]
    if sample_path.exists():
        candidates.append(str(sample_path))

    for file_path in candidates:
        # 为每个文件计算当前 hash，与上次索引记录比较。
        current_hash = _file_hash(file_path)
        if index_record.get(file_path) == current_hash:
            print(f"[跳过] 已索引且未变化: {os.path.basename(file_path)}")
            continue
        # 进入待索引列表，并提前更新内存中的 hash 记录。
        files_to_index.append(file_path)
        index_record[file_path] = current_hash

    if files_to_index:
        for file_path in files_to_index:
            print(f"[索引] {os.path.basename(file_path)} ...")
            # 文档加载 -> 切块 -> 写入向量库（标准 RAG ingestion 三步）。
            docs = rag.load_documents(file_path)
            chunks = rag.split_documents(docs)
            rag.create_vectorstore(chunks)
        # 所有新文件索引成功后，落盘最新 hash 记录。
        _save_index_record(index_record)
        print(f"共索引 {len(files_to_index)} 个文件")
    else:
        print("所有文件已索引，无需重复写入。")

    # 5) 初始化检索器与问答链，进入 CLI 交互。
    rag.setup_retriever(k=rag._retrieval_top_k)
    rag.create_chain()
    rag.chat_loop()


if __name__ == "__main__":
    # 仅在“脚本直接运行”时进入 main；被 import 时不自动启动。
    main()
