MyAgents 升级方案：LangGraph + 多类型记忆 + RAG 整合
Context
当前 Agent 框架基于自定义 Agent(ABC) + 手动 function calling 循环，记忆是简单的三级文本（short/long/core）存于 PostgreSQL，无向量检索。RAG 系统独立于 rag/rag.py，使用旧版 PGVector API。

本次升级目标：

Agent 编排层迁移到 LangGraph StateGraph（替代自定义 Agent ABC）
构建多类型记忆系统（工作/情景/语义/感知），向量检索基于 PGVector（复用现有 PostgreSQL）
RAG 整合进记忆模块，共享 Embedding 和存储基础设施
实现范围：Phase 0-2（基础层 + 记忆类型 + RAG/工具集成）
Phase 0: 基础层
0.1 LLM 层迁移：LLMClient → ChatOpenAI
为什么：LangGraph 原生使用 LangChain 的 ChatModel 接口（.bind_tools(), .invoke() 返回 AIMessage）。当前自定义 LLMClient 无法直接接入。

新建 core/llm_factory.py：

def create_chat_model(provider=None, **kwargs) -> BaseChatModel:
    """工厂函数：复用当前 LLMClient 的 provider 自动检测逻辑，返回 LangChain ChatModel"""
复用 core/llm.py 的 _auto_detect_provider() 和 _resolve_credentials() 逻辑
返回 ChatOpenAI(model=..., api_key=..., base_url=..., streaming=True)
保留对 Ollama、ModelScope、智谱、SiliconFlow 等 provider 的自动检测
当前 core/llm.py 保留不删（向后兼容，ReflectionAgent 暂时仍用）
0.2 工具适配层：自定义 Tool → LangChain Tool
为什么：LangGraph 的 ToolNode 需要 LangChain BaseTool 接口，当前自定义 Tool(ABC) 不兼容。

新建 tools/adapter.py：

class LangChainToolAdapter(BaseTool):
    """将自定义 Tool(ABC) 包装为 LangChain BaseTool"""
    name: str
    description: str
    custom_tool: Any  # 我们的 Tool(ABC) 实例

    def _run(self, **kwargs) -> str:
        return self.custom_tool.run(kwargs)

def adapt_tools(registry: ToolRegistry) -> list[BaseTool]:
    """批量转换 ToolRegistry 中所有工具"""
现有 CalculatorTool、SearchTool 无需任何修改即可接入 LangGraph
新工具推荐直接用 @tool 装饰器编写
0.3 记忆基础数据结构
新建 memory/__init__.py 新建 memory/base.py：

MemoryType — 枚举：WORKING / EPISODIC / SEMANTIC / PERCEPTUAL
MemoryItem(BaseModel)：
id: str(UUID), content: str, memory_type: MemoryType
embedding: Optional[List[float]]（缓存向量）
importance: float（0.0-1.0）, tags: List[str], ttl: Optional[int]
timestamp, metadata, access_count, last_access
方法：touch(), is_expired(), relevance_score()
MemoryConfig(BaseModel)：
embedding 配置（provider / model / api_key / base_url）
pgvector 配置（database_url, 各 table name）
sqlite 配置（路径，用于轻量索引记录）
各项阈值（TTL, 晋升条件等）
类方法 from_env() 从环境变量读取
BaseMemory(ABC)：
add(content, importance, metadata, tags) -> str
search(query, k, filters) -> List[MemoryItem]
get(id), delete(id), clear(), count(), get_all()
0.4 统一 Embedding 服务
新建 memory/embedding.py：

EmbeddingService：封装 LangChain Embeddings 接口

工厂模式，根据 config.embedding_provider 选择：
"siliconflow" / "openai" → OpenAIEmbeddings(api_key, model="BAAI/bge-m3", base_url, chunk_size=64) — 复用 rag/rag.py:129-134
"dashscope" → OpenAIEmbeddings + DashScope 兼容 URL
接口：embed_text(), embed_texts(), embed_query(), dimension, langchain_embeddings
0.5 PGVector 存储后端
新建 memory/storage/__init__.py 新建 memory/storage/pgvector_store.py：

PGVectorMemoryStore：基于新版 langchain-postgres API

class PGVectorMemoryStore:
    def __init__(self, config: MemoryConfig, embedding_service: EmbeddingService, table_name: str):
        self.engine = PGEngine.from_connection_string(url=config.database_url)
        self.engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=embedding_service.dimension,
        )
        self.store = PGVectorStore.create_sync(
            engine=self.engine,
            table_name=table_name,
            embedding_service=embedding_service.langchain_embeddings,
        )
add(items: List[MemoryItem]) -> List[str] — 存入 PGVectorStore（Document 格式）
search(query, k, filters) -> List[Tuple[MemoryItem, float]] — similarity_search_with_score
delete(id), count()
as_retriever(k) -> VectorStoreRetriever — 供 RAG chain 使用
各记忆类型使用不同 table：memory_episodic, memory_semantic, rag_documents
共享同一个 PGEngine（连接池复用，避免连接数爆炸）
注意：当前 rag/rag.py 使用旧版 PGVector 类，新代码使用 PGEngine + PGVectorStore。这是 langchain-postgres 的 API 升级。

0.6 SQLite 文档索引存储
新建 memory/storage/sqlite_store.py：

SQLiteDocumentStore：

SQLAlchemy + SQLite (data/memory.db)
表 memory_documents：记忆元数据持久化（id, content, memory_type, importance, tags, timestamp...）
表 indexed_files：文件索引去重（替代 rag/.indexed_files.json）
CRUD + 按类型/标签/重要度查询
0.7 依赖更新
修改 pyproject.toml，新增：

langgraph>=0.4.0
langchain-core>=0.3.0
已有但确认版本：langchain-postgres>=0.0.17（支持 PGEngine API）、langchain-openai、langchain-community、langchain-text-splitters

0.8 环境变量
修改 .env.example，新增：

EMBEDDING_PROVIDER=siliconflow
EMBEDDING_MODEL=BAAI/bge-m3
WORKING_MEMORY_TTL=1800
MEMORY_SQLITE_PATH=data/memory.db
Phase 1: 记忆类型 + MemoryManager
1.1 工作记忆
新建 memory/types/__init__.py 新建 memory/types/working.py：

WorkingMemory(BaseMemory)：

纯内存 dict[str, MemoryItem] + threading.Timer TTL
默认 TTL = 1800s（与当前 decay_short 的 30 分钟一致）
search() — 子串匹配 + 时间倒序（量小无需向量搜索）
get_recent(n=10) — 获取最近 N 条
should_promote(item) -> bool — 综合评分超阈值时建议晋升
1.2 情景记忆
新建 memory/types/episodic.py：

EpisodicMemory(BaseMemory)：

双存储：PGVectorStore（向量检索）+ SQLite（结构化持久化）
add() — embed → PGVectorStore + SQLite
search(query, k) — PGVector 相似搜索 → 混合评分：
语义相似度 * 0.6 + 时间衰减 * 0.2 + 重要度 * 0.2
get_by_timerange(start, end) — 时间范围查询
核心改进：不再 dump 所有记忆进 context，而是按当前 query 检索相关记忆
1.3 语义记忆（Phase 1 简化版）
新建 memory/types/semantic.py：

SemanticMemory(BaseMemory)（仅 PGVector，无 Neo4j）：

独立 PGVectorStore table (memory_semantic)
每条记忆包含 subject/predicate/object 标签
add_knowledge(subject, predicate, object_) — 存储知识三元组
search() — 向量相似搜索 + metadata 过滤
1.4 感知记忆 (stub)
新建 memory/types/perceptual.py：

PerceptualMemory(BaseMemory) — 所有方法抛出 NotImplementedError
1.5 统一记忆管理器
新建 memory/manager.py：

MemoryManager：

class MemoryManager:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig.from_env()
        self.embedding_service = EmbeddingService(self.config)
        self.working = WorkingMemory(self.config, self.embedding_service)
        self.episodic = EpisodicMemory(self.config, self.embedding_service)
        self.semantic = SemanticMemory(self.config, self.embedding_service)
关键方法：

add(content, memory_type, **kwargs) -> str
search(query, memory_types=None, k=5) -> List[MemoryItem] — 跨类型搜索+合并排序
consolidate() — working→episodic 晋升，高频 episodic 标记 "core"
build_context(query=None) -> str — 构建记忆上下文文本，供 LangGraph 节点注入
向后兼容层（确保尚未迁移的代码正常工作）：

def add_short(self, content, importance=1): ...
def get_short(self): return self.working.get_all()
def get_long(self): ...
def get_core(self): ...
def memory_cycle(self): self.consolidate()
Phase 2: LangGraph Agent + RAG + 工具
2.1 LangGraph Agent 状态定义
新建 agents/state.py：

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class AgentState(TypedDict):
    """所有 Agent 共享的基础状态"""
    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史
    memory_context: str       # 注入的记忆上下文
    iteration_count: int      # 当前迭代次数

class PlanSolveState(AgentState):
    """PlanAndSolve 专用状态"""
    plan: list[str]           # 计划步骤列表
    current_step: int         # 当前执行到第几步
    step_results: list[str]   # 各步骤的执行结果

class ReflectionState(AgentState):
    """Reflection 专用状态"""
    current_code: str         # 当前代码版本
    feedback: str             # 评审反馈
    iteration: int            # 当前迭代轮次
2.2 LangGraph ReAct Agent
新建 agents/react_graph.py（替代 agents/reactAgent.py）：

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def create_react_agent(model, tools, memory_manager, system_prompt=None):
    """构建 ReAct Agent Graph"""
    model_with_tools = model.bind_tools(tools)

    def inject_memory(state: AgentState):
        """记忆检索节点：根据最新用户消息检索相关记忆"""
        last_user_msg = [m for m in state["messages"] if m.type == "human"][-1]
        memory_text = memory_manager.build_context(query=last_user_msg.content)
        return {"memory_context": memory_text}

    def llm_call(state: AgentState):
        """LLM 调用节点：注入记忆 + system prompt + 消息历史"""
        sys_msgs = []
        if system_prompt:
            sys_msgs.append(SystemMessage(content=system_prompt))
        if state.get("memory_context"):
            sys_msgs.append(SystemMessage(content=f"相关记忆:\n{state['memory_context']}"))
        return {"messages": [model_with_tools.invoke(sys_msgs + state["messages"])]}

    def update_memory(state: AgentState):
        """记忆更新节点：将重要交互存入工作记忆"""
        last_msg = state["messages"][-1]
        if last_msg.type == "ai" and last_msg.content:
            memory_manager.working.add(last_msg.content[:200], importance=0.3)
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("inject_memory", inject_memory)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("update_memory", update_memory)

    graph.add_edge(START, "inject_memory")
    graph.add_edge("inject_memory", "llm_call")
    graph.add_conditional_edges("llm_call", tools_condition, ["tools", "update_memory"])
    graph.add_edge("tools", "llm_call")
    graph.add_edge("update_memory", END)

    return graph.compile()
对比当前 ReactAgent：

当前 agents/reactAgent.py:28-71：手动循环 for i in range(max_iterations) + 手动 JSON parse tool_calls
新版：LangGraph 的 ToolNode + tools_condition 自动处理工具循环
新增 inject_memory 节点：query-aware 记忆检索（核心升级）
2.3 LangGraph PlanAndSolve Agent
新建 agents/plan_solve_graph.py（替代 agents/planSolveAgent.py）：

def create_plan_solve_agent(model, memory_manager):
    """构建 PlanAndSolve Agent Graph"""

    def inject_memory(state): ...  # 同上

    def plan_node(state: PlanSolveState):
        """规划节点：生成步骤列表"""
        # 复用当前 PLANNER_PROMPT_TEMPLATE
        ...
        return {"plan": plan_list, "current_step": 0, "step_results": []}

    def execute_step(state: PlanSolveState):
        """执行节点：执行当前步骤"""
        # 复用当前 EXECUTOR_PROMPT_TEMPLATE
        ...
        return {"step_results": [...], "current_step": state["current_step"] + 1}

    def should_continue(state: PlanSolveState):
        """条件边：判断是否还有步骤"""
        if state["current_step"] >= len(state["plan"]):
            return "done"
        return "execute"

    graph = StateGraph(PlanSolveState)
    graph.add_node("inject_memory", inject_memory)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_step)

    graph.add_edge(START, "inject_memory")
    graph.add_edge("inject_memory", "plan")
    graph.add_conditional_edges("plan", should_continue, {"execute": "execute", "done": END})
    graph.add_conditional_edges("execute", should_continue, {"execute": "execute", "done": END})

    return graph.compile()
2.4 LangGraph Reflection Agent
新建 agents/reflection_graph.py（替代 agents/reflectionAgent.py）：

def create_reflection_agent(model, memory_manager, max_iterations=3):
    """构建 Reflection Agent Graph"""

    def generate(state: ReflectionState):
        """生成代码节点"""
        ...
        return {"current_code": code, "iteration": state.get("iteration", 0)}

    def reflect(state: ReflectionState):
        """评审节点"""
        ...
        return {"feedback": feedback}

    def should_refine(state: ReflectionState):
        """条件边：是否需要继续优化"""
        if "无需改进" in state["feedback"]:
            return END
        if state["iteration"] >= max_iterations:
            return END
        return "refine"

    def refine(state: ReflectionState):
        """优化节点"""
        ...
        return {"current_code": refined_code, "iteration": state["iteration"] + 1}

    graph = StateGraph(ReflectionState)
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)
    graph.add_node("refine", refine)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges("reflect", should_refine, {"refine": "refine", END: END})
    graph.add_edge("refine", "reflect")

    return graph.compile()
2.5 RAG 文档处理器
新建 memory/rag/__init__.py 新建 memory/rag/document.py：

DocumentProcessor：从 rag/rag.py:326-372 提取

load(file_path) — PyPDFLoader / TextLoader
split(documents) — RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
file_hash(path) — MD5 去重（复用 rag/rag.py:78-84）
2.6 RAG 管道
新建 memory/rag/pipeline.py：

RAGPipeline（替代 RAGApplication）：

使用共享的 PGVectorMemoryStore(table_name="rag_documents") + EmbeddingService
index_file(path) / index_directory(dir, pattern) — SQLite 去重索引
hybrid_retrieve(question, k, fetch_k) — 完整移植混合检索逻辑：
从 rag/rag.py:196-307 移植：_tokenize_query_for_keyword, _keyword_score, 停用词、混合评分（语义 75% + 关键词 25%）
将 self.vector_store.similarity_search_with_score() 替换为新版 PGVectorStore API
create_chain() — 复用 rag/rag.py:424-456 的 prompt 模板
query(question) -> dict — 端到端问答
2.7 记忆工具
新建 tools/builtin/memory_tool.py：

from langchain_core.tools import tool

@tool
def store_memory(content: str, importance: float = 0.5, memory_type: str = "working") -> str:
    """存储一条记忆。memory_type: working/episodic/semantic"""
    ...

@tool
def recall_memory(query: str, k: int = 5) -> str:
    """搜索相关记忆"""
    ...

@tool
def list_recent_memory(n: int = 10) -> str:
    """列出最近的工作记忆"""
    ...
使用 LangChain @tool 装饰器（而非自定义 Tool ABC），直接兼容 LangGraph ToolNode。

2.8 RAG 工具
新建 tools/builtin/rag_tool.py：

@tool
def knowledge_search(query: str, k: int = 5) -> str:
    """搜索知识库文档，回答基于文档的问题"""
    ...
2.9 更新入口
修改 main.py：

from langchain_openai import ChatOpenAI
from memory import MemoryManager, MemoryConfig
from memory.rag import RAGPipeline
from tools.adapter import adapt_tools
from tools.builtin.memory_tool import store_memory, recall_memory
from tools.builtin.rag_tool import knowledge_search
from tools.builtin.calculator_tool import CalculatorTool
from tools.builtin.search import SearchTool
from agents.react_graph import create_react_agent

def main():
    # LLM (LangChain ChatModel)
    model = ChatOpenAI(model=..., api_key=..., base_url=..., streaming=True)

    # Memory
    config = MemoryConfig.from_env()
    memory_mgr = MemoryManager(config)

    # RAG
    rag = RAGPipeline(config, memory_mgr.embedding_service)
    rag.index_directory("rag/papers", "*.pdf")

    # Tools
    legacy_tools = adapt_tools(ToolRegistry([CalculatorTool(), SearchTool()]))
    all_tools = legacy_tools + [store_memory, recall_memory, knowledge_search]

    # Agent
    agent = create_react_agent(model, all_tools, memory_mgr)

    # Interactive loop
    while True:
        user_input = input("You: ")
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print(result["messages"][-1].content)
关键文件清单
操作	文件路径	说明
新建	memory/__init__.py	公共 API 导出
新建	memory/base.py	MemoryItem, MemoryConfig, BaseMemory
新建	memory/embedding.py	EmbeddingService
新建	memory/storage/__init__.py	存储层导出
新建	memory/storage/pgvector_store.py	PGVectorMemoryStore（新版 API）
新建	memory/storage/sqlite_store.py	SQLiteDocumentStore
新建	memory/types/__init__.py	记忆类型导出
新建	memory/types/working.py	WorkingMemory
新建	memory/types/episodic.py	EpisodicMemory
新建	memory/types/semantic.py	SemanticMemory
新建	memory/types/perceptual.py	PerceptualMemory (stub)
新建	memory/manager.py	MemoryManager
新建	memory/rag/__init__.py	RAG 导出
新建	memory/rag/document.py	DocumentProcessor
新建	memory/rag/pipeline.py	RAGPipeline
新建	core/llm_factory.py	create_chat_model 工厂函数
新建	tools/adapter.py	LangChainToolAdapter
新建	tools/builtin/memory_tool.py	记忆工具（@tool 装饰器）
新建	tools/builtin/rag_tool.py	RAG 工具（@tool 装饰器）
新建	agents/state.py	AgentState / PlanSolveState / ReflectionState
新建	agents/react_graph.py	LangGraph ReAct Agent
新建	agents/plan_solve_graph.py	LangGraph PlanSolve Agent
新建	agents/reflection_graph.py	LangGraph Reflection Agent
修改	main.py	整合新组件
修改	pyproject.toml	新增 langgraph 依赖
修改	.env.example	新增环境变量
保留	core/agent.py	向后兼容，迁移完成后废弃
保留	core/llm.py	向后兼容
保留	core/memory.py	向后兼容
保留	rag/rag.py	向后兼容
可复用的现有代码
来源	移植到	内容
rag/rag.py:129-134	memory/embedding.py	OpenAIEmbeddings 配置
rag/rag.py:196-227	memory/rag/pipeline.py	中英文关键词分词
rag/rag.py:229-245	memory/rag/pipeline.py	关键词评分算法
rag/rag.py:247-307	memory/rag/pipeline.py	混合检索逻辑
rag/rag.py:326-372	memory/rag/document.py	文档加载和分割
rag/rag.py:424-456	memory/rag/pipeline.py	QA chain prompt 模板
rag/rag.py:78-84	memory/rag/document.py	MD5 文件哈希
core/llm.py:64-110	core/llm_factory.py	Provider 自动检测
agents/planSolveAgent.py:8-39	agents/plan_solve_graph.py	Planner/Executor prompt 模板
agents/reflectionAgent.py:5-47	agents/reflection_graph.py	Generate/Reflect/Refine prompt 模板
LangChain/LangGraph 组件映射
目标组件	包	类/函数	用于
Agent 编排	langgraph	StateGraph, START, END	所有 Agent graph
工具节点	langgraph.prebuilt	ToolNode, tools_condition	ReAct Agent
会话持久化	langgraph.checkpoint.memory	MemorySaver	对话状态跨轮次
消息类型	langchain-core	HumanMessage, AIMessage, SystemMessage, ToolMessage	状态中 messages
消息合并	langgraph.graph	add_messages	AgentState 注解
LLM 模型	langchain-openai	ChatOpenAI	LLM 调用
工具绑定	langchain-openai	ChatOpenAI.bind_tools()	ReAct 工具调用
工具定义	langchain-core	@tool 装饰器	新工具
工具适配	langchain-core	BaseTool	旧工具包装
Embedding	langchain-openai	OpenAIEmbeddings	EmbeddingService
向量存储	langchain-postgres	PGEngine, PGVectorStore	PGVectorMemoryStore
文档加载	langchain-community	PyPDFLoader, TextLoader	DocumentProcessor
文本分割	langchain-text-splitters	RecursiveCharacterTextSplitter	DocumentProcessor
QA 链	langchain	create_stuff_documents_chain	RAGPipeline
Prompt	langchain-core	ChatPromptTemplate	RAGPipeline
实现顺序
Phase 0 (基础层) — 无外部依赖变化
  memory/base.py ─────────────── 无依赖
  memory/embedding.py ─────────── 依赖 base.py, langchain-openai
  memory/storage/pgvector_store.py ── 依赖 base.py, embedding.py, langchain-postgres
  memory/storage/sqlite_store.py ── 依赖 base.py, sqlalchemy
  core/llm_factory.py ─────────── 依赖 langchain-openai (复用 core/llm.py 逻辑)
  tools/adapter.py ────────────── 依赖 tools/base.py, langchain-core
  pyproject.toml + .env.example ── 配置更新

Phase 1 (记忆类型 + 管理器)
  memory/types/working.py ─────── 依赖 base.py
  memory/types/episodic.py ────── 依赖 base.py, pgvector_store, sqlite_store, embedding
  memory/types/semantic.py ────── 依赖 base.py, pgvector_store, embedding
  memory/types/perceptual.py ──── 依赖 base.py (stub)
  memory/manager.py ───────────── 依赖 所有 types, 向后兼容层

Phase 2 (LangGraph + RAG + 工具 + 集成)
  agents/state.py ─────────────── 依赖 langgraph
  agents/react_graph.py ───────── 依赖 state.py, langgraph, memory/manager
  agents/plan_solve_graph.py ──── 依赖 state.py, langgraph, memory/manager
  agents/reflection_graph.py ──── 依赖 state.py, langgraph, memory/manager
  memory/rag/document.py ──────── 依赖 langchain loaders/splitters
  memory/rag/pipeline.py ──────── 依赖 document.py, pgvector_store, embedding
  tools/builtin/memory_tool.py ── 依赖 memory/manager (用 @tool 装饰器)
  tools/builtin/rag_tool.py ──── 依赖 memory/rag/pipeline (用 @tool 装饰器)
  main.py ─────────────────────── 依赖 所有以上
验证方式
Phase 0：

EmbeddingService 能通过 SiliconFlow 生成 1024 维向量
PGVectorMemoryStore 能 add + similarity_search
create_chat_model() 返回可用的 ChatOpenAI 实例
LangChainToolAdapter 能包装 CalculatorTool 并通过 LangGraph ToolNode 调用
Phase 1：

WorkingMemory TTL 过期正确
EpisodicMemory 向量搜索返回语义相关结果
MemoryManager 向后兼容 API（add_short, get_core, memory_cycle）全部通过
Phase 2：

create_react_agent 生成的 graph 能处理 "23*47等于多少"（走 CalculatorTool）
create_plan_solve_agent 能拆解复杂问题并逐步执行
RAGPipeline 用现有 rag/ragtest_cases.json 测试通过
main.py 启动后完整交互闭环：Agent 能用工具、存取记忆、查询知识库