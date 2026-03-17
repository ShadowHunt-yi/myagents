import os
import sys
from pathlib import Path
from typing import List

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

load_dotenv()


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
        # Limit batch size to avoid API 413 when a chunk list exceeds provider max
        self.embeddings = OpenAIEmbeddings(
            api_key=embeddings_key,
            model="BAAI/bge-m3",
            base_url=embeddings_url,
            chunk_size=64,
        )
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            connection=db_url,
            collection_name="documents",
        )

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
        self.vector_store.add_documents(cleaned)
        print(
            f"Vectorstore: {len(cleaned)} chunks added (removed {removed} empty/invalid)"
        )

    def setup_retriever(self, k: int = 8):
        """设置检索器"""
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        print(f"Retriever setup with k={k}")

    def debug_retrieval(self, question: str, k: int = 8):
        docs = self.vector_store.similarity_search(question, k=k)
        print(f"\n[DEBUG] Retrieved {len(docs)} docs\n")

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "unknown")
            print(f"--- DOC {i} | file={source} | page={page} ---")
            print(doc.page_content[:500])
            print()

    def create_chain(self):
        """创建检索链"""
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant specialized in answering questions based on provided context.
          
            Guidelines:
            - Answer based only on the provided context
            - If the answer is not in the context, say "I don't have enough information to answer this question"
            - Be concise but complete
            - Cite specific parts of the context when possible
          
            Context:
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

        response = self.chain.invoke({"input": question})
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
    # 设置API密钥
    api_key = os.getenv("LONGCAT_API_KEY")
    model = os.getenv("LONGCAT_MODEL", "gpt-5.2-codex")
    base_url = os.getenv("LONGCAT_BASE_URL", "http://localhost:11434")

    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
    db_url = os.getenv("DATABASE_URL")
    embeddings_key = os.getenv("SILICONFLOW_API_KEY")
    embeddings_url = os.getenv("SILICONFLOW_BASE_URL")
    # 创建RAG应用
    rag = RAGApplication(llm, embeddings_key, embeddings_url, db_url)

    indexed_any = False

    papers_dir = Path(__file__).parent / "papers"
    pdf_files = sorted(papers_dir.glob("*.pdf")) if papers_dir.exists() else []
    if pdf_files:
        for pdf in pdf_files:
            print(f"Loading and indexing PDF: {pdf.name} ...")
            docs = rag.load_documents(str(pdf))
            chunks = rag.split_documents(docs)
            rag.create_vectorstore(chunks)
            indexed_any = True
    else:
        print("No PDFs found in papers/, using existing vectorstore data.")

    sample_path = Path(__file__).parent / "sample.txt"
    if sample_path.exists():
        print("Loading and indexing documents from sample.txt ...")
        docs = rag.load_documents(str(sample_path))
        chunks = rag.split_documents(docs)
        rag.create_vectorstore(chunks)
        indexed_any = True
    else:
        print("No sample.txt found, using existing vectorstore data.")

    if not indexed_any:
        print("Nothing indexed this run. Vectorstore may be empty.")

    # 设置检索器和链
    rag.setup_retriever(k=4)
    rag.create_chain()

    # 启动交互式问答
    rag.chat_loop()


# def main():
#     api_key = os.getenv("LONGCAT_API_KEY")
#     model = os.getenv("LONGCAT_MODEL", "gpt-5.2-codex")
#     base_url = os.getenv("LONGCAT_BASE_URL", "http://localhost:11434")

#     print("LLM_BASE_URL =", base_url)
#     print("LLM_MODEL =", model)

#     llm = ChatOpenAI(
#         model=model,
#         api_key=api_key,
#         base_url=base_url,
#         temperature=0,
#     )

#     print(llm.invoke("你好，请回复一句测试成功"))
if __name__ == "__main__":
    main()
