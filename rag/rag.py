import hashlib
import json
import os
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_postgres import PGVector
from dotenv import load_dotenv
from sqlalchemy.exc import InterfaceError as SAInterfaceError
from sqlalchemy.exc import OperationalError as SAOperationalError

load_dotenv()

# 已索引文件的记录，防止重复写入
INDEX_RECORD_PATH = Path(__file__).parent / ".indexed_files.json"


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
        self.chain = None  # 稍后初始化
        self.llm = llm_client
        self.retriever = None  # 稍后初始化
        self._retriever_k = 6
        self._collection_name = "documents"
        self._db_url = db_url
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
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
        )
        print(f"Retriever setup with MMR, k={k}")

    def debug_retrieval(self, question: str, k: int = 8):
        docs = self._run_with_db_retry(
            lambda: self.vector_store.similarity_search(question, k=k),
            op_name="debug retrieval",
        )
        print(f"\n[DEBUG] Retrieved {len(docs)} docs\n")

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "unknown")
            print(f"--- DOC {i} | file={source} | page={page} ---")
            print(doc.page_content[:500])
            print()

    def create_chain(self):
        """创建检索链"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的知识库问答助手。请根据下面提供的参考资料回答用户问题。

规则：
- 只根据参考资料中的内容回答，不要编造信息
- 如果参考资料中没有相关内容，请明确说"参考资料中没有找到相关信息"
- 回答要简洁完整，尽量引用原文关键内容
- 使用中文回答

参考资料：
{context}""",
                ),
                ("human", "{input}"),
            ]
        )

        # 创建文档处理链
        document_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=prompt, document_separator="\n\n---\n\n"
        )

        # 创建检索链
        self.chain = create_retrieval_chain(
            retriever=self.retriever, combine_docs_chain=document_chain
        )
        print("RAG chain created successfully")

    def query(self, question: str) -> dict:
        """执行查询"""
        if not self.chain:
            raise ValueError("Chain not initialized. Call create_chain() first.")

        response = self._run_with_db_retry(
            lambda: self.chain.invoke({"input": question}),
            op_name="chain invoke",
            rebuild_chain=True,
        )
        return response

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
                self.debug_retrieval(question, k=8)
                response = self.query(question)
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
    rag.setup_retriever(k=4)
    rag.create_chain()

    # 启动交互式问答
    rag.chat_loop()

if __name__ == "__main__":
    main()
