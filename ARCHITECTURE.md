# MyAgents 架构设计文档

## 一、项目现状诊断

### 当前文件结构

```
myagents/
├── main.py                    # 入口（使用旧接口，已过时）
├── useModel.py                # Ollama 启动脚本（独立可用）
├── core/
│   ├── agent.py               # Agent 基类（有雏形，但没人继承）
│   ├── llm.py                 # LLM 客户端（能用，但不支持 function calling）
│   ├── message.py             # 消息类（基础可用）
│   ├── memory.py              # 记忆管理（设计不错）
│   ├── config.py              # 空文件
│   └── exceptions.py          # 空文件
├── tools/
│   ├── base.py                # Tool 基类（刚重写，已就绪 ✅）
│   ├── registry.py            # 工具注册中心（刚重写，已就绪 ✅）
│   └── builttin/
│       ├── calculator_tool.py # 计算器工具（刚重写，已就绪 ✅）
│       └── search.py          # 搜索工具（刚重写，已就绪 ✅）
└── agents/
    ├── reactAgent.py          # ReAct Agent（未继承基类，用旧接口）
    ├── reflectionAgent.py     # 反思 Agent（未继承基类，引用不存在的模块）
    └── planSolveAgent.py      # 计划 Agent（未继承基类）
```

### 核心问题清单

| # | 问题 | 位置 | 严重程度 |
|---|------|------|----------|
| 1 | **Agent 基类没人继承** — 三个 Agent 各写各的，完全绕过了 `core/agent.py` | agents/*.py | 🔴 高 |
| 2 | **LLM 不支持 function calling** — `chat()` 只能纯文本对话，没有 `tools` 参数 | core/llm.py | 🔴 高 |
| 3 | **Message 不支持工具消息** — `to_dict()` 只输出 role+content，缺少 tool_calls/tool_call_id | core/message.py | 🔴 高 |
| 4 | **Agent 基类没有工具集成** — `Agent.__init__` 里没有 `ToolRegistry` 参数 | core/agent.py | 🔴 高 |
| 5 | **类型引用错误** — agent.py 引用 `HelloAgentsLLM`，实际类名是 `LLMClient` | core/agent.py | 🟡 中 |
| 6 | **import 路径全乱** — reactAgent 用 `from llm import`，reflectionAgent 用 `from memories import` | agents/*.py | 🟡 中 |
| 7 | **config / exceptions 空文件** — 定义了但没内容 | core/ | 🟢 低 |
| 8 | **main.py 用旧接口** — 引用已删除的 ToolExecutor | main.py | 🟡 中 |

### 一句话总结

> **基础层（tools）刚修好，但中间层（core）和上层（agents）还是断裂的。**
> 三个 Agent 各自为政，没有共享 LLM 调用、工具执行、消息管理的统一路径。

---

## 二、目标架构

### 分层设计

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
│                tools/ 层                         │  ← 工具定义 + 注册（已完成 ✅）
│  Tool基类 │ ToolRegistry │ 内置工具              │
└─────────────────────────────────────────────────┘
```

### 核心原则

1. **Agent 基类管"公共能力"** — LLM 调用、消息历史、记忆、工具注册，子类不用重复写
2. **子类只管"思考策略"** — ReAct 用循环解析 Action，PlanSolve 用先规划后执行，Reflection 用生成+评审
3. **LLMClient 同时支持纯文本和 function calling** — 一个方法走天下
4. **Message 覆盖所有消息类型** — user / assistant / system / tool，包括 tool_calls

---

## 三、每个文件该怎么改

### 3.1 core/llm.py — 补上 function calling

**现状**: `chat()` 只能 `messages + temperature + max_tokens`，没法传 `tools`。

**改造要点**:

```python
class LLMClient:
    def chat(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,       # ← 新增: OpenAI function calling schema
        temperature: float = 0,
        max_tokens: int = 512,
        stream: bool = True,
    ) -> "LLMResponse":
        """
        统一调用入口。
        - 不传 tools → 普通对话，返回纯文本
        - 传 tools   → function calling，返回可能包含 tool_calls
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools
            params["stream"] = False     # function calling 通常不用流式
        else:
            params["stream"] = stream

        response = self.client.chat.completions.create(**params)
        # ... 解析返回
```

**为什么这样改**: ReAct Agent 需要 function calling（让 LLM 直接选择工具），
而 Reflection / PlanSolve Agent 只需要纯文本对话。一个方法兼顾两种模式。

### 3.2 core/message.py — 支持工具消息

**现状**: 只有 `role + content`。

**改造要点**:

```python
class Message(BaseModel):
    role: MessageRole                         # "user" / "assistant" / "system" / "tool"
    content: Optional[str] = None             # 可以为 None（当 assistant 返回 tool_calls 时）
    tool_calls: Optional[List[Dict]] = None   # assistant 消息中的工具调用
    tool_call_id: Optional[str] = None        # tool 消息回传时的关联 ID
    name: Optional[str] = None                # tool 消息中的工具名

    def to_dict(self) -> Dict[str, Any]:
        """转为 OpenAI API 格式，自动跳过 None 字段"""
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d
```

**为什么需要这些字段**: OpenAI function calling 的消息流是这样的:

```
assistant: {tool_calls: [{id: "call_123", function: {name: "calculator", arguments: ...}}]}
tool:      {tool_call_id: "call_123", name: "calculator", content: "计算结果: 42"}
```

Message 必须能表达这两种消息，否则工具调用链就断了。

### 3.3 core/agent.py — 集成工具系统

**现状**: `__init__` 没有 `ToolRegistry`，`run()` 是固定逻辑（不适合所有策略）。

**改造要点**:

```python
class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        tools: Optional[ToolRegistry] = None,  # ← 新增
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or ToolRegistry()
        self.memory = MemoryManager()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str) -> str:
        """子类必须实现自己的运行策略"""
        ...

    # --- 以下是公共能力，子类直接用 ---

    def chat(self, messages: list[dict], use_tools: bool = False):
        """调 LLM，可选是否带工具"""
        tools = self.tools.get_all_schemas() if use_tools and self.tools else None
        return self.llm.chat(messages, tools=tools)

    def execute_tool(self, name: str, parameters: dict) -> str:
        """执行工具的便捷方法"""
        return self.tools.execute(name, parameters)

    def build_context(self) -> list[dict]:
        """构建完整上下文（system prompt + 记忆 + 历史）"""
        # ... 和现在类似
```

**关键变化**:
- `run()` 变成 `@abstractmethod`，强制子类实现（不同策略差异太大）
- 新增 `chat()` 和 `execute_tool()` 公共方法，子类不用直接操作 llm/tools
- 工具注册表通过构造函数注入

### 3.4 agents/ — 三个 Agent 继承基类

#### ReActAgent（function calling 版）

```python
class ReActAgent(Agent):
    """ReAct: Thought → Action → Observation 循环"""

    def run(self, input_text: str) -> str:
        messages = self.build_context()
        messages.append({"role": "user", "content": input_text})

        for i in range(self.max_iterations):
            # 带工具调 LLM
            response = self.chat(messages, use_tools=True)

            # 如果 LLM 选择了工具
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self.execute_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments)
                    )
                    # 把工具结果加入消息
                    messages.append(...)  # assistant message with tool_calls
                    messages.append(...)  # tool message with result
            else:
                # 没选工具 = 最终回答
                return response.content

        return "达到最大迭代次数"
```

**对比现在**: 现在的 reactAgent 用正则解析 `Action:` / `Action Input:`（脆弱），
改成 function calling 后由 LLM API 原生支持工具选择，更可靠。

#### ReflectionAgent

```python
class ReflectionAgent(Agent):
    """生成 → 评审 → 优化 循环"""

    def run(self, task: str) -> str:
        # 1. 初始生成
        code = self.chat([{"role": "user", "content": f"编写: {task}"}])

        # 2. 迭代反思
        for i in range(self.max_iterations):
            feedback = self.chat([...])  # 评审
            if "无需改进" in feedback:
                break
            code = self.chat([...])      # 优化

        return code
```

**这个 Agent 不需要工具**，但仍然继承 Agent 基类，复用 LLM 调用和记忆。

#### PlanSolveAgent

```python
class PlanSolveAgent(Agent):
    """先规划，后逐步执行"""

    def run(self, question: str) -> str:
        # 1. 规划
        plan = self._plan(question)

        # 2. 逐步执行
        for step in plan:
            result = self.chat([...])     # 每步可选是否用工具

        return result
```

### 3.5 main.py — 简洁的组装入口

```python
from core.llm import LLMClient
from tools import ToolRegistry, CalculatorTool, SearchTool
from agents.react_agent import ReActAgent

def main():
    # 1. LLM
    llm = LLMClient()

    # 2. 工具
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(SearchTool())

    # 3. Agent
    agent = ReActAgent(name="助手", llm=llm, tools=tools)

    # 4. 交互循环
    while True:
        user_input = input("你: ")
        if user_input in ("quit", "exit"):
            break
        print(agent.run(user_input))

if __name__ == "__main__":
    main()
```

---

## 四、数据流全景图

一次完整的工具调用，数据是这样流动的:

```
用户输入 "23*47等于多少"
         │
         ▼
┌─── ReActAgent.run() ───┐
│  1. build_context()     │    构建 system prompt + 记忆 + 历史
│  2. chat(use_tools=True)│    ─→ LLMClient.chat(messages, tools=schemas)
│         │               │
│         ▼               │
│    LLM API 返回         │
│    tool_calls: [{       │
│      name: "calculator" │
│      arguments: {       │
│        "expression":    │
│        "23*47"          │
│      }                  │
│    }]                   │
│         │               │
│  3. execute_tool()      │    ─→ ToolRegistry.execute("calculator", {"expression": "23*47"})
│         │               │                │
│         │               │                ▼
│         │               │        CalculatorTool.run({"expression": "23*47"})
│         │               │                │
│         │               │                ▼
│         │               │           返回 "1081"
│         │               │
│  4. 把结果加入 messages  │
│  5. 再次 chat()          │    LLM 看到工具结果，生成最终回答
│         │               │
│         ▼               │
│  返回 "23×47=1081"      │
└─────────────────────────┘
```

---

## 五、实施顺序（从哪开始）

按**依赖关系从底向上**的顺序，每步完成后都可以独立测试:

### 第 1 步：core/message.py ⬅ 先改这个

> 所有上层都依赖消息格式，改动最小，影响最大。

- 加 `tool_calls`、`tool_call_id`、`name` 字段
- 改 `to_dict()` 支持完整消息格式
- 预计改动: ~20 行

### 第 2 步：core/llm.py

> Agent 的核心能力，必须先支持 function calling。

- `chat()` 加 `tools` 参数
- 返回结构化的响应（而不是纯字符串）
- 预计改动: ~40 行

### 第 3 步：core/agent.py

> 打通 LLM + 工具 + 记忆的统一基类。

- 加 `tools: ToolRegistry` 参数
- `run()` 改为 `@abstractmethod`
- 加 `chat()` / `execute_tool()` 公共方法
- 修复 `HelloAgentsLLM` → `LLMClient`
- 预计改动: ~30 行

### 第 4 步：agents/react_agent.py

> 最能体现工具系统价值的 Agent，优先改。

- 继承 `Agent`
- 用 function calling 替代正则解析
- 预计改动: 重写 ~60 行

### 第 5 步：agents/reflection_agent.py + agents/plan_solve_agent.py

> 这两个不依赖工具，改造简单。

- 继承 `Agent`
- 复用 `self.chat()` 和 `self.memory`
- 修复 import 路径

### 第 6 步：main.py

> 最后更新入口，用新接口组装。

### 第 7 步（可选）：core/config.py + core/exceptions.py

> 不急，等跑通了再补。

```
依赖方向（从底到顶）:

  tools/base.py ──────────────────────── ✅ 已完成
  tools/registry.py ─────────────────── ✅ 已完成
  tools/builttin/* ──────────────────── ✅ 已完成
       │
  core/message.py ───────────────────── 第 1 步
       │
  core/llm.py ──────────────────────── 第 2 步
       │
  core/agent.py ────────────────────── 第 3 步（依赖 message + llm + tools）
       │
  agents/react_agent.py ────────────── 第 4 步（依赖 agent 基类）
  agents/reflection_agent.py ────────── 第 5 步
  agents/plan_solve_agent.py ────────── 第 5 步
       │
  main.py ──────────────────────────── 第 6 步（最终组装）
```

---

## 六、扩展一个新工具只需要做什么

以"天气查询工具"为例，只需要:

```python
# tools/builttin/weather.py

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="查询城市天气")

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="city", type="string", description="城市名")]

    def run(self, parameters: Dict[str, Any]) -> str:
        city = parameters["city"]
        # 调用天气 API ...
        return f"{city}今天晴，25°C"
```

然后在 main.py 里一行注册:

```python
tools.register(WeatherTool())
```

不需要改 Agent、不需要改 LLM、不需要改 Registry。这就是架构的意义。

---

## 七、扩展一个新 Agent 只需要做什么

以 "ToolAgent"（纯工具调用 Agent，无复杂策略）为例:

```python
# agents/tool_agent.py

class ToolAgent(Agent):
    def run(self, input_text: str) -> str:
        messages = self.build_context()
        messages.append({"role": "user", "content": input_text})
        response = self.chat(messages, use_tools=True)
        return response.content
```

继承 `Agent`，实现 `run()`，就这么简单。
LLM 调用、工具执行、记忆管理全部由基类提供。
