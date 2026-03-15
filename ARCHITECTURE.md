# MyAgents 架构设计文档

## 一、项目结构

```
myagents/
├── main.py                    # 入口（待更新）
├── useModel.py                # Ollama 启动脚本（独立可用）
├── core/
│   ├── llm.py                 # LLM 客户端 + LLMResponse ✅
│   ├── agent.py               # Agent 基类（集成 LLM + 工具 + 记忆）✅
│   ├── message.py             # 消息类（支持 function calling 格式）✅
│   ├── memory.py              # 记忆管理 ✅
│   ├── config.py              # 配置（待补充）
│   └── exceptions.py          # 异常（待补充）
├── tools/
│   ├── base.py                # Tool 基类 + ToolParameter + FunctionTool ✅
│   ├── registry.py            # 工具注册中心 ✅
│   └── builttin/
│       ├── calculator_tool.py # 计算器工具 ✅
│       └── search.py          # 搜索工具 ✅
└── agents/
    ├── reactAgent.py          # ReAct Agent（基于 Function Calling）✅
    ├── reflectionAgent.py     # 反思 Agent（待接入基类）
    └── planSolveAgent.py      # 计划 Agent（待接入基类）
```

---

## 二、问题诊断（改造前是什么样）

> 这一节记录改造前发现的所有问题，以及"为什么这是问题"的思考过程。
> 已修复的标注 ✅，未修复的标注 ⬜。

### 2.1 问题总览

| # | 问题 | 位置 | 严重程度 | 状态 |
|---|------|------|----------|------|
| 1 | LLM 不支持 function calling | core/llm.py | 🔴 高 | ✅ |
| 2 | Message 不支持工具消息格式 | core/message.py | 🔴 高 | ✅ |
| 3 | Agent 基类没有工具集成 | core/agent.py | 🔴 高 | ✅ |
| 4 | ReactAgent 两套范式混用 | agents/reactAgent.py | 🔴 高 | ✅ |
| 5 | Agent 基类类型引用错误 | core/agent.py | 🟡 中 | ✅ |
| 6 | ReflectionAgent 未继承基类 | agents/reflectionAgent.py | 🟡 中 | ⬜ |
| 7 | PlanSolveAgent 未继承基类 | agents/planSolveAgent.py | 🟡 中 | ⬜ |
| 8 | main.py 用旧接口 | main.py | 🟡 中 | ⬜ |
| 9 | config / exceptions 空文件 | core/ | 🟢 低 | ⬜ |

### 2.2 逐个分析

#### 问题 1：LLM 不支持 function calling ✅ 已修复

**改造前的代码** (`core/llm.py`):

```python
def chat(self, messages, temperature=0, max_tokens=512) -> Optional[str]:
    response = self.client.chat.completions.create(
        model=self.model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True,
    )
    collected = []
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        collected.append(content)
    return "".join(collected)   # ← 返回纯字符串
```

**问题在哪**:
- 没有 `tools` 参数 → 无法把工具 schema 传给 OpenAI API
- 返回值是 `str` → 上层拿不到 `tool_calls` 结构
- 永远用流式 → function calling 的流式处理比非流式复杂得多（tool_calls 会被拆成多个 chunk）

**思考过程**:
OpenAI 的 function calling 需要在请求中传入 `tools` 参数，API 会返回一个包含 `tool_calls` 的结构化响应。
如果 `chat()` 只返回字符串，上层 Agent 就无法知道 LLM 到底是"给出了文字回答"还是"想调用工具"。
所以需要一个统一的返回类型来承载两种情况。

**怎么改的**:
1. 新增 `LLMResponse` 数据类，同时容纳 `content`（文本）和 `tool_calls`（工具调用）
2. `chat()` 新增 `tools` 参数：
   - 有 tools → 关闭流式，拿完整响应，提取 `message.tool_calls`
   - 无 tools → 保持流式，拼接文本
3. 返回值从 `str` 改为 `LLMResponse`

**为什么有 tools 时关闭流式**:
流式 function calling 中，`tool_calls` 会被分散到多个 chunk 里（id 在第一个 chunk，函数名和参数可能跨多个 chunk），
拼装逻辑复杂且容易出错。非流式直接拿到完整的 `tool_calls`，简单可靠。
纯文本对话保持流式，是为了用户体验（逐字输出）。

---

#### 问题 2：Message 不支持工具消息格式 ✅ 已修复

**改造前的代码** (`core/message.py`):

```python
class Message(BaseModel):
    role: MessageRole          # "user" | "assistant" | "system" | "tool"
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}
```

**问题在哪**:
OpenAI function calling 需要在消息列表里传递两种特殊消息：

```
# 1. assistant 说"我想调用计算器"
{"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "calculator", ...}}]}

# 2. 系统告诉 LLM"工具返回了什么"
{"role": "tool", "tool_call_id": "call_123", "content": "1081"}
```

旧 Message 只有 `role + content`，无法表达 `tool_calls` 和 `tool_call_id`，
也就无法把工具调用的"请求-响应"链路完整地放进消息历史。

**思考过程**:
function calling 的消息流是一个**配对关系**：
assistant 发出带 `tool_calls` 的消息（id=call_123），
紧跟一条 role=tool 的消息，用 `tool_call_id=call_123` 回传结果。
OpenAI API 要求这两条消息**必须成对出现**，否则会报错。
所以 Message 必须能表达这两种格式。

**怎么改的**:
新增三个可选字段：
- `tool_calls` — assistant 消息携带的工具调用列表
- `tool_call_id` — tool 消息的关联 ID
- `name` — tool 消息的工具名

`to_dict()` 根据字段组合输出不同格式：
- assistant + tool_calls → 输出 tool_calls 结构
- tool + tool_call_id → 输出 tool_call_id + content
- 其他 → 普通 role + content

---

#### 问题 3：Agent 基类没有工具集成 ✅ 已修复

**改造前的代码** (`core/agent.py`):

```python
class Agent(ABC):
    def __init__(self, name, llm, system_prompt=None, config=None,
                 tools: Optional[Tool] = None):           # ← 类型错误：Tool 是抽象基类，不是注册表
        self.tools = tools or ToolRegistry().get_tools()   # ← ToolRegistry 没有 get_tools() 方法

    def chat(self, messages, use_tools=False) -> str:      # ← 返回类型错误
        tools = self.tools.get_all_schemas() if use_tools and self.tools else None
        return self.llm.chat(messages, tools=tools)        # ← LLMClient.chat() 不接受 tools（问题1）
```

**问题在哪**:
三层错误叠加：
1. `tools: Optional[Tool]` — 类型应该是 `ToolRegistry`，不是单个 `Tool`
2. `ToolRegistry().get_tools()` — `ToolRegistry` 没有这个方法（只有 `get_all_schemas()`）
3. `self.llm.chat(messages, tools=tools)` — 调用了不存在的参数（问题 1 导致的连锁反应）

**思考过程**:
这是一个典型的**接口不匹配**问题。Agent 基类写了工具集成的"壳"，
但底层 LLMClient 还不支持 tools，所以这层代码从未被真正运行和验证过。
修复顺序必须**从底向上**：先修 LLMClient（问题 1），再修 Message（问题 2），最后修 Agent。

**怎么改的**:
- `tools` 类型改为 `Optional[ToolRegistry]`
- 默认值改为 `tools or ToolRegistry()`（空注册表，而不是调用不存在的方法）
- `chat()` 返回类型改为 `LLMResponse`（与问题 1 的改造对齐）
- `run()` 加上 `@abstractmethod`，强制子类实现

---

#### 问题 4：ReactAgent 两套范式混用 ✅ 已修复

**改造前的代码** (`agents/reactAgent.py`):

```python
class reactAgent(Agent):
    def __init__(self, llm, tool_executor: ToolExecutor):  # ← ToolExecutor 类不存在
        ...

    # 范式 A：文本解析（ReAct 经典方式），已实现但 run() 中没用
    def _parse_action(self, text):
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.+)", text)
        ...

    # 范式 B：OpenAI Function Calling，在 run() 中使用但底层不支持
    def run(self, user_input):
        response = self.chat(messages, use_tools=True)
        if response.tool_calls:              # ← str 没有 .tool_calls
            for tool_call in response.tool_calls:
                tool_call.function.name      # ← str 没有 .function
```

**问题在哪**:
同一个类里混了两套完全不同的工具调用方式：
- **文本解析**（`_parse_action`）：让 LLM 输出 `Action: xxx` 格式文本，用正则提取 → 已写好但没被调用
- **Function Calling**（`run` 里的 `response.tool_calls`）：依赖 OpenAI API 原生返回 → 在用但底层不支持

结果：两套都不能工作。

**思考过程**:
这两种方式的**根本区别**：

| | 文本解析（ReAct 经典） | Function Calling（OpenAI 原生） |
|---|---|---|
| 工具选择靠谁 | LLM 输出特定格式文本，代码正则提取 | LLM 直接返回结构化 tool_calls |
| 可靠性 | 低（LLM 可能不按格式输出） | 高（API 保证结构化输出） |
| 适用场景 | 不支持 function calling 的模型 | 支持 function calling 的模型 |

既然选了方案 B（Function Calling），就应该**彻底删掉文本解析的代码**，
包括 system prompt 里的格式说明、`_parse_action()`、`_parse_final_answer()`。

**怎么改的**:
- 删掉 `_parse_action`、`_parse_final_answer`、文本格式的 SYSTEM_PROMPT_TEMPLATE
- 删掉对 `ToolExecutor` 的引用，改为接收 `ToolRegistry`
- `run()` 完全基于 `response.tool_calls` 驱动循环：
  1. `chat(messages, use_tools=True)` → 拿到 `LLMResponse`
  2. 有 `tool_calls` → 执行工具，把 assistant 消息和 tool 消息追加到 messages，继续循环
  3. 无 `tool_calls` → 返回 `response.content` 作为最终回答

---

#### 问题 5：Agent 基类类型引用错误 ✅ 已修复

**改造前**: `from ..tools import ToolRegistry` 在实际代码中引用的是 `Tool` 而不是 `ToolRegistry`，
且 import 了不存在的符号。

**怎么改的**: 修正 import，`tools` 参数类型改为 `Optional[ToolRegistry]`。

---

#### 问题 6：ReflectionAgent 未继承基类 ⬜ 待修复

**现状**: 独立实现，引用了不存在的 `from memories import reflectionMemory`。

**改造思路**:
- 继承 `Agent` 基类，复用 `self.chat()` 和 `self.memory`
- 这个 Agent **不需要工具**，只需要纯文本对话，所以 `chat()` 调用时不传 `use_tools`
- 保留"生成 → 评审 → 优化"三阶段循环

---

#### 问题 7：PlanSolveAgent 未继承基类 ⬜ 待修复

**现状**: 独立实现，Planner 和 Executor 是两个单独的类。

**改造思路**:
- 合并为一个继承 `Agent` 的类
- `run()` 实现两阶段：先调 `self.chat()` 生成计划，再逐步执行
- 可选是否在执行阶段使用工具

---

#### 问题 8：main.py 用旧接口 ⬜ 待修复

**现状**: 引用已删除的 `ToolExecutor`。

**改造思路**:
```python
llm = LLMClient()
tools = ToolRegistry()
tools.register(CalculatorTool())
agent = ReactAgent(llm=llm, tools=tools)
agent.run("23*47等于多少")
```

---

## 三、分层架构

```
┌─────────────────────────────────────────────────┐
│                   main.py                        │  ← 入口：组装各层，启动交互
├─────────────────────────────────────────────────┤
│                agents/ 层                        │  ← 具体策略：ReAct / Reflection / PlanSolve
│  （继承 Agent 基类，只关注"思考策略"）            │
├─────────────────────────────────────────────────┤
│                 core/ 层                         │  ← 基础能力：LLM、消息、记忆、工具调度
│  Agent基类 │ LLMClient │ Message │ Memory        │
├─────────────────────────────────────────────────┤
│                tools/ 层                         │  ← 工具定义 + 注册
│  Tool基类 │ ToolRegistry │ 内置工具              │
└─────────────────────────────────────────────────┘
```

### 核心原则

1. **Agent 基类管"公共能力"** — LLM 调用、消息历史、记忆、工具注册，子类不重复写
2. **子类只管"思考策略"** — ReAct 用工具循环，PlanSolve 用先规划后执行，Reflection 用生成+评审
3. **LLMClient 同时支持纯文本和 function calling** — 通过 `tools` 参数切换
4. **Message 覆盖所有消息类型** — user / assistant / system / tool，包括 tool_calls

---

## 四、核心模块设计

### 4.1 core/llm.py — LLMClient + LLMResponse

`LLMResponse` 统一封装 LLM 的返回值：

```python
@dataclass
class LLMResponse:
    content: Optional[str] = None        # 文本回复
    tool_calls: Optional[List[Any]] = None  # OpenAI tool_call 对象列表
```

`LLMClient.chat()` 根据是否传入 `tools` 切换行为：

| 场景 | 流式 | 返回值 |
|------|------|--------|
| 无 tools（纯文本） | 是 | `LLMResponse(content="文本")` |
| 有 tools（function calling） | 否 | `LLMResponse(content=..., tool_calls=[...])` |

支持的 provider：OpenAI、ModelScope、智谱、Ollama、vLLM、自定义，通过环境变量或参数自动检测。

### 4.2 core/message.py — Message

支持三种消息格式的 `to_dict()` 输出：

| 类型 | 字段 | 示例 |
|------|------|------|
| 普通消息 | role + content | `{"role": "user", "content": "..."}` |
| assistant 工具调用 | role + tool_calls | `{"role": "assistant", "tool_calls": [...]}` |
| tool 结果 | role + tool_call_id + content | `{"role": "tool", "tool_call_id": "call_123", "content": "结果"}` |

### 4.3 core/agent.py — Agent 基类

```python
class Agent(ABC):
    def __init__(self, name, llm, system_prompt=None, config=None, tools=None):
        # LLM、工具、记忆、对话历史 — 子类直接用

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """子类必须实现的运行策略"""

    def chat(self, messages, use_tools=False) -> LLMResponse:
        """调 LLM，可选是否带工具 schema"""

    def execute_tool(self, name, parameters) -> str:
        """执行工具的便捷方法"""

    def build_context(self) -> list[dict]:
        """构建完整上下文: system prompt + 三级记忆 + 对话历史"""
```

### 4.4 tools/ — 工具系统

```
Tool(ABC)  ──  定义接口: get_parameters() + run() + to_openai_schema()
    │
    ├── FunctionTool  ──  函数快速包装器
    ├── CalculatorTool
    └── SearchTool

ToolRegistry  ──  注册 + 查找 + Schema 批量生成 + 执行
```

注册两种方式：`register(Tool实例)` 或 `register_function(名称, 描述, 参数, 函数)`。

---

## 五、数据流：一次完整的工具调用

```
用户输入 "23*47等于多少"
         │
         ▼
┌─── ReactAgent.run() ──────┐
│  1. 构建 messages           │
│  2. chat(use_tools=True)   │──→ LLMClient.chat(messages, tools=schemas)
│         │                  │         │
│         ▼                  │         ▼ (非流式调用)
│    LLMResponse:            │    OpenAI API 返回
│      tool_calls: [{        │
│        name: "calculator"  │
│        arguments: {        │
│          "expression":     │
│          "23*47"           │
│        }                   │
│      }]                    │
│         │                  │
│  3. execute_tool()         │──→ ToolRegistry.execute("calculator", {...})
│         │                  │         │
│         │                  │         ▼
│         │                  │    CalculatorTool.run() → "1081"
│         │                  │
│  4. 把结果加入 messages     │
│     assistant msg: tool_calls
│     tool msg: result       │
│                            │
│  5. 再次 chat()             │──→ LLM 看到工具结果，生成最终回答
│         │                  │
│         ▼                  │
│  返回 "23×47=1081"         │
└────────────────────────────┘
```

---

## 六、实施进度

按**依赖关系从底向上**的顺序，每步完成后都可以独立测试：

```
  tools/base.py ──────────────────────── ✅ 已完成
  tools/registry.py ─────────────────── ✅ 已完成
  tools/builttin/* ──────────────────── ✅ 已完成
       │
  core/message.py ───────────────────── ✅ 已完成（支持 tool_calls/tool_call_id）
       │
  core/llm.py ──────────────────────── ✅ 已完成（LLMResponse + tools 参数）
       │
  core/agent.py ────────────────────── ✅ 已完成（ToolRegistry + @abstractmethod）
       │
  agents/reactAgent.py ────────────── ✅ 已完成（Function Calling 版）
  agents/reflectionAgent.py ────────── ⬜ 待接入基类
  agents/planSolveAgent.py ─────────── ⬜ 待接入基类
       │
  main.py ──────────────────────────── ⬜ 待更新入口
```

**为什么是这个顺序**: 每一层都依赖下面的层。先改 Message（所有层都用），
再改 LLMClient（Agent 要调），再改 Agent 基类（子类要继承），最后改具体 Agent。
反过来改的话，每一步都会因为底层没就绪而无法测试。

---

## 七、扩展指南

### 新增一个工具

```python
# tools/builttin/weather.py
class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="查询城市天气")

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="city", type="string", description="城市名")]

    def run(self, parameters: Dict[str, Any]) -> str:
        city = parameters["city"]
        return f"{city}今天晴，25°C"
```

然后注册：`tools.register(WeatherTool())`。不需要改 Agent、LLM 或 Registry。

### 新增一个 Agent

```python
# agents/my_agent.py
class MyAgent(Agent):
    def run(self, input_text: str, **kwargs) -> str:
        messages = self.build_context()
        messages.append({"role": "user", "content": input_text})
        response = self.chat(messages, use_tools=True)
        return response.content
```

继承 `Agent`，实现 `run()`。LLM 调用、工具执行、记忆管理全部由基类提供。
