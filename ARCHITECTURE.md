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

## 二、分层架构

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

## 三、核心模块设计

### 3.1 core/llm.py — LLMClient + LLMResponse

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

### 3.2 core/message.py — Message

支持三种消息格式的 `to_dict()` 输出：

| 类型 | 字段 | 示例 |
|------|------|------|
| 普通消息 | role + content | `{"role": "user", "content": "..."}` |
| assistant 工具调用 | role + tool_calls | `{"role": "assistant", "tool_calls": [...]}` |
| tool 结果 | role + tool_call_id + content | `{"role": "tool", "tool_call_id": "call_123", "content": "结果"}` |

### 3.3 core/agent.py — Agent 基类

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

### 3.4 tools/ — 工具系统

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

## 四、数据流：一次完整的工具调用

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

## 五、实施进度

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

---

## 六、扩展指南

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
