import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, List

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"

# 将项目根目录加入 Python 路径
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

# 已索引文件的记录，防止重复写入
INDEX_RECORD_PATH = Path(__file__).parent / ".indexed_files.json"

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
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希，用于判断文件是否变更"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_index_record() -> dict:
    if INDEX_RECORD_PATH.exists():
        return json.loads(INDEX_RECORD_PATH.read_text(encoding="utf-8"))
    return {}


def _save_index_record(record: dict):
    INDEX_RECORD_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


class RAGApplication:
    """一个简单的 RAG 应用,展示如何结合文档加载、文本分割、向量数据库和 LLM 来实现基于知识库的问答"""

    def __init__(
        self,
        llm_client: ChatOpenAI,
        embeddings_key: str,
        embeddings_url: str,
        db_url: str,
    ):
        self.chain = None  # 兼容旧调用，实际存放 document_chain
        self.document_chain = None
        self.llm = llm_client
        self.retriever = None  # 稍后初始化
        self._retriever_k = 6
        self._collection_name = "documents"
        self._db_url = db_url
        self._retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
        self._retrieval_fetch_k = int(os.getenv("RETRIEVAL_FETCH_K", "40"))
        self._hybrid_sem_weight = float(os.getenv("HYBRID_SEM_WEIGHT", "0.75"))
        self._hybrid_keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.25"))
        self._retrieval_debug_scores = _env_bool("RETRIEVAL_DEBUG_SCORES", True)
        self._min_final_score = float(os.getenv("RETRIEVAL_MIN_FINAL_SCORE", "0.0"))

        total_weight = self._hybrid_sem_weight + self._hybrid_keyword_weight
        if total_weight <= 0:
            self._hybrid_sem_weight = 0.75
            self._hybrid_keyword_weight = 0.25
        else:
            self._hybrid_sem_weight /= total_weight
            self._hybrid_keyword_weight /= total_weight
        # Limit batch size to avoid API 413 when a chunk list exceeds provider max
        self.embeddings = OpenAIEmbeddings(
            api_key=embeddings_key,
            model="BAAI/bge-m3",
            base_url=embeddings_url,
            chunk_size=64,
        )
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self) -> PGVector:
        # Keep pooled DB connections healthy in long-running CLI sessions.
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
        return PGVector(
            embeddings=self.embeddings,
            connection=self._db_url,
            collection_name=self._collection_name,
            engine_args=engine_args,
        )

    @staticmethod
    def _is_db_connection_error(exc: Exception) -> bool:
        if isinstance(exc, (SAOperationalError, SAInterfaceError)):
            return True

        text = str(exc).lower()
        markers = (
            "software caused connection abort",
            "could not receive data from server",
            "server closed the connection unexpectedly",
            "connection reset by peer",
            "ssl connection has been closed unexpectedly",
        )
        return any(marker in text for marker in markers)

    def _reconnect_vector_store(self, rebuild_chain: bool = False):
        self.vector_store = self._build_vector_store()
        if self.retriever is not None:
            self.setup_retriever(k=self._retriever_k)
        if rebuild_chain and self.chain is not None and self.retriever is not None:
            self.create_chain()

    def _run_with_db_retry(
        self,
        fn: Callable[[], Any],
        op_name: str,
        rebuild_chain: bool = False,
    ) -> Any:
        try:
            return fn()
        except Exception as exc:
            if not self._is_db_connection_error(exc):
                raise
            print(
                f"[WARN] {op_name} failed due to DB disconnect, reconnecting and retrying once..."
            )
            self._reconnect_vector_store(rebuild_chain=rebuild_chain)
            return fn()

    @staticmethod
    def _tokenize_query_for_keyword(question: str) -> List[str]:
        raw_tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", question.lower())
        tokens: List[str] = []
        seen = set()

        for tok in raw_tokens:
            if not tok.strip():
                continue
            if re.fullmatch(r"[A-Za-z0-9_]+", tok):
                if tok in EN_STOPWORDS or len(tok) <= 1:
                    continue
                if tok not in seen:
                    seen.add(tok)
                    tokens.append(tok)
                continue

            if tok in CN_STOPWORDS or len(tok) <= 1:
                continue
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)

            if len(tok) > 2:
                for i in range(len(tok) - 1):
                    bg = tok[i : i + 2]
                    if bg in CN_STOPWORDS:
                        continue
                    if bg not in seen:
                        seen.add(bg)
                        tokens.append(bg)

        return tokens

    @staticmethod
    def _keyword_score(content: str, tokens: List[str], question: str) -> float:
        if not tokens:
            return 0.0

        text = content.lower()
        hits = sum(1 for token in tokens if token in text)
        score = hits / len(tokens)

        tf = sum(text.count(token) for token in tokens)
        score += min(tf / 20.0, 0.2)

        q = question.strip().lower()
        if q and len(q) >= 4 and q in text:
            score += 0.1

        return max(0.0, min(1.0, score))

    def hybrid_retrieve(
        self,
        question: str,
        k: int | None = None,
        fetch_k: int | None = None,
        filter: dict | None = None,
    ) -> List[Document]:
        top_k = k or self._retrieval_top_k
        candidate_k = max(top_k, fetch_k or self._retrieval_fetch_k)

        docs_and_scores = self._run_with_db_retry(
            lambda: self.vector_store.similarity_search_with_score(
                question, k=candidate_k, filter=filter
            ),
            op_name="hybrid retrieval candidates",
        )
        if not docs_and_scores:
            return []

        tokens = self._tokenize_query_for_keyword(question)
        distances = [float(score) for _, score in docs_and_scores if score is not None]
        min_distance = min(distances) if distances else 0.0
        max_distance = max(distances) if distances else 1.0
        denom = max(max_distance - min_distance, 1e-12)

        ranked_docs: List[Document] = []
        for doc, distance in docs_and_scores:
            distance_value = float(distance) if distance is not None else max_distance
            semantic_score = 1.0 - ((distance_value - min_distance) / denom)
            semantic_score = max(0.0, min(1.0, semantic_score))
            keyword_score = self._keyword_score(doc.page_content, tokens, question)
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

            ranked_docs.append(
                Document(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        ranked_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
        filtered_docs = [
            d
            for d in ranked_docs
            if d.metadata.get("_final_score", 0.0) >= self._min_final_score
        ]
        if len(filtered_docs) >= top_k:
            return filtered_docs[:top_k]
        return ranked_docs[:top_k]

    @staticmethod
    def _group_docs_by_source(docs: List[Document]) -> List[Document]:
        groups: dict[str, List[Document]] = {}
        order: List[str] = []

        for doc in docs:
            source = doc.metadata.get("source_file", "unknown")
            if source not in groups:
                groups[source] = []
                order.append(source)
            groups[source].append(doc)

        grouped_docs: List[Document] = []
        for source in order:
            grouped_docs.extend(groups[source])
        return grouped_docs

    def load_documents(self, file_path: str) -> List:
        """加载文档"""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(file_path)

        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents

    def split_documents(self, documents: List) -> List:
        """分割文档"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List):
        """创建向量数据库并写入文档"""
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
        self._run_with_db_retry(
            lambda: self.vector_store.add_documents(cleaned),
            op_name="vectorstore add_documents",
        )
        print(
            f"Vectorstore: {len(cleaned)} chunks added (removed {removed} empty/invalid)"
        )

    def clear_collection(self):
        """清空向量数据库中的所有文档（用于清理重复数据后重新索引）"""
        self.vector_store.delete_collection()
        # 重新创建空 collection
        self.vector_store = self._build_vector_store()
        # 同时清除索引记录
        if INDEX_RECORD_PATH.exists():
            INDEX_RECORD_PATH.unlink()
        print("向量数据库已清空，索引记录已重置。下次启动会重新索引所有文件。")

    def setup_retriever(self, k: int = 6):
        """设置检索器，使用 MMR 避免返回重复内容"""
        self._retriever_k = k
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
        docs = self.hybrid_retrieve(
            question=question,
            k=k or self._retrieval_top_k,
            fetch_k=self._retrieval_fetch_k,
        )
        print(f"\n[DEBUG] Retrieved {len(docs)} docs\n")

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
            print(doc.page_content[:500])
            print()
        return docs

    def create_chain(self):
        """创建文档问答链（检索在链外完成）"""
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
- 使用中文回答，并在关键术语后保留英文原词（如有）。

参考资料：
{context}""",
                ),
                ("human", "{input}"),
            ]
        )
        document_prompt = PromptTemplate.from_template(
            "[文档: {source_file} | 页码: {page}]\n{page_content}"
        )

        self.document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=document_prompt,
            document_separator="\n\n---\n\n",
        )
        self.chain = self.document_chain
        print("Document QA chain created successfully")

    def query(self, question: str, docs: List[Document] | None = None) -> dict:
        """执行查询"""
        if not self.document_chain:
            raise ValueError("Chain not initialized. Call create_chain() first.")

        retrieved_docs = docs or self.hybrid_retrieve(
            question=question,
            k=self._retrieval_top_k,
            fetch_k=self._retrieval_fetch_k,
        )
        grouped_docs = self._group_docs_by_source(retrieved_docs)

        answer = self.document_chain.invoke({"input": question, "context": grouped_docs})
        return {"answer": answer, "context": retrieved_docs}

    def chat_loop(self):
        """交互式问答循环"""
        import traceback

        print("\n" + "=" * 50)
        print("RAG Application Ready! Type 'exit' to quit.")
        print("=" * 50 + "\n")

        while True:
            question = input("Question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                print("[DEBUG] invoking chain...")
                retrieved_docs = self.debug_retrieval(question, k=self._retrieval_top_k)
                response = self.query(question, docs=retrieved_docs)
                print("[DEBUG] chain invoke success")

                answer = response.get("answer", "No answer returned")
                context = response.get("context", [])

                print(f"\nAnswer: {answer}")
                print(f"\nSources ({len(context)} documents retrieved):")
                for i, doc in enumerate(context, 1):
                    content = getattr(doc, "page_content", "")
                    source = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "unknown")
                    print(f"  {i}. [file={source}, page={page}] {content[:300]}...")
                print()

            except Exception as e:
                print(f"\n[ERROR TYPE] {type(e).__name__}")
                print(f"[ERROR MSG] {e}\n")
                traceback.print_exc()
                print()


def main():
    """主函数"""
    api_key = os.getenv("LONGCAT_API_KEY")
    model = os.getenv("LONGCAT_MODEL", "gpt-5.2-codex")
    base_url = os.getenv("LONGCAT_BASE_URL", "http://localhost:11434")

    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
    db_url = os.getenv("DATABASE_URL")
    embeddings_key = os.getenv("SILICONFLOW_API_KEY")
    embeddings_url = os.getenv("SILICONFLOW_BASE_URL")

    rag = RAGApplication(llm, embeddings_key, embeddings_url, db_url)

    # ---- 去重索引：只索引新增或内容变化的文件 ----
    index_record = _load_index_record()
    files_to_index: list[str] = []

    papers_dir = Path(__file__).parent / "papers"
    pdf_files = sorted(papers_dir.glob("*.pdf")) if papers_dir.exists() else []
    sample_path = Path(__file__).parent / "sample.txt"

    candidates = [str(p) for p in pdf_files]
    if sample_path.exists():
        candidates.append(str(sample_path))

    for fpath in candidates:
        current_hash = _file_hash(fpath)
        if index_record.get(fpath) == current_hash:
            print(f"[跳过] 已索引且未变化: {os.path.basename(fpath)}")
            continue
        files_to_index.append(fpath)
        index_record[fpath] = current_hash

    if files_to_index:
        for fpath in files_to_index:
            print(f"[索引] {os.path.basename(fpath)} ...")
            docs = rag.load_documents(fpath)
            chunks = rag.split_documents(docs)
            rag.create_vectorstore(chunks)
        _save_index_record(index_record)
        print(f"共索引 {len(files_to_index)} 个文件")
    else:
        print("所有文件已索引，无需重复写入。")

    # 设置检索器和链
    rag.setup_retriever(k=rag._retrieval_top_k)
    rag.create_chain()

    # 启动交互式问答
    rag.chat_loop()

if __name__ == "__main__":
    main()
