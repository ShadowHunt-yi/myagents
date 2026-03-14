# MCP 与 Skill 扩展指南

## 一、先搞清楚这两个东西是什么

### MCP（Model Context Protocol）

MCP 是一个**标准协议**，让你的 Agent 可以连接外部工具服务，而不需要把工具代码写在自己项目里。

```
没有 MCP 的世界:
  每个工具都要自己写代码，自己维护

  Agent
    ├── 计算器工具（自己写）
    ├── 搜索工具（自己写）
    ├── 数据库工具（自己写）
    ├── 文件操作工具（自己写）
    └── ...写不完

有 MCP 的世界:
  工具以"服务"的形式存在，Agent 通过协议连接即可

  Agent ──MCP协议──→ 计算器服务（别人写的/社区提供的）
    │
    ├──MCP协议──→ 数据库服务（一个独立进程）
    │
    ├──MCP协议──→ GitHub 服务（官方提供的）
    │
    └──MCP协议──→ 你自己写的任何服务
```

一句话：**MCP 就是工具界的 USB 接口** —— 统一标准，即插即用。

### Skill（技能）

Skill 是更高一层的概念：**一组工具 + 专属提示词 + 行为逻辑 的打包体**。

```
Tool = 一个函数
  例: calculator（只会算数）

Skill = 一套能力
  例: "数据分析师" 技能包含:
    ├── 工具: SQL查询、图表生成、CSV解析
    ├── 专属 system prompt: "你是一个数据分析师，擅长..."
    └── 行为: 自动把查询结果生成图表
```

一句话：**Tool 是螺丝刀，Skill 是工具箱 + 使用说明书**。

---

## 二、它们在架构中的位置

```
┌──────────────────────────────────────────────────────────┐
│                        main.py                            │
├──────────────────────────────────────────────────────────┤
│                      agents/ 层                           │
│              ReActAgent / PlanSolveAgent / ...             │
├──────────────────────────────────────────────────────────┤
│                       core/ 层                            │
│           Agent基类 │ LLMClient │ SkillManager            │  ← Skill 管理在这里
├──────────────────────────────────────────────────────────┤
│                      tools/ 层                            │
│  ToolRegistry                                             │
│    ├── 本地工具: CalculatorTool, SearchTool               │  ← 已有的
│    └── MCP工具: MCPTool(适配器，连接外部 MCP 服务)        │  ← 新增的
├──────────────────────────────────────────────────────────┤
│                     mcp/ 层（新增）                        │
│  MCPClient │ MCPTool │ MCPServerManager                   │  ← MCP 连接层
└──────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   外部 MCP Server 1              外部 MCP Server 2
   (数据库工具)                    (GitHub工具)
```

---

## 三、MCP 集成方案

### 3.1 整体思路

```
MCP Server（外部进程）                你的 Agent
┌──────────────┐                ┌──────────────────────┐
│  tool: query │◄──MCP协议──── │  MCPClient           │
│  tool: insert│                │    │                  │
└──────────────┘                │    ▼                  │
                                │  MCPTool(适配器)      │  ← 把 MCP 工具伪装成本地 Tool
                                │    │                  │
                                │    ▼                  │
                                │  ToolRegistry         │  ← 注册进去，Agent 无感知
                                │    │                  │
                                │    ▼                  │
                                │  Agent                │  ← 像用本地工具一样用
                                └──────────────────────┘
```

核心想法：**Agent 不需要知道工具是本地的还是远程的**。
通过 `MCPTool` 适配器，MCP 工具被伪装成普通的 `Tool` 子类，注册到 `ToolRegistry` 里。

### 3.2 需要新增的文件

```
myagents/
├── mcp/                         ← 新增目录
│   ├── __init__.py
│   ├── client.py                ← MCP 客户端，管理连接
│   ├── tool_adapter.py          ← MCPTool 适配器，MCP工具 → Tool 接口
│   └── config.py                ← MCP 服务器配置
```

### 3.3 各文件的设计

#### mcp/config.py — MCP 服务器配置

```python
"""
定义如何连接到一个 MCP Server

支持两种连接方式:
  - stdio: 启动一个子进程，通过标准输入/输出通信（本地工具常用）
  - http:  连接远程 HTTP 服务（远程工具常用）
"""
from pydantic import BaseModel
from typing import Optional, List

class MCPServerConfig(BaseModel):
    """一个 MCP Server 的连接配置"""

    name: str                                    # 服务名，如 "github"、"database"
    transport: str = "stdio"                     # "stdio" 或 "http"

    # stdio 方式的参数
    command: Optional[str] = None                # 启动命令，如 "python"
    args: Optional[List[str]] = None             # 命令参数，如 ["mcp_server.py"]

    # http 方式的参数
    url: Optional[str] = None                    # 如 "http://localhost:8000/mcp"
```

实际使用时，可以用一个 JSON 文件来管理所有 MCP Server：

```json
// mcp_servers.json
{
  "servers": [
    {
      "name": "github",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    },
    {
      "name": "database",
      "transport": "http",
      "url": "http://localhost:9000/mcp"
    }
  ]
}
```

#### mcp/client.py — MCP 客户端

```python
"""
MCP 客户端 — 负责连接 MCP Server、发现工具、调用工具

生命周期:
  1. connect()    → 启动连接，建立会话
  2. list_tools() → 发现服务器上有哪些工具
  3. call_tool()  → 调用某个工具
  4. close()      → 断开连接

基于官方 mcp Python SDK (pip install mcp)
"""
import asyncio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from .config import MCPServerConfig


class MCPClient:
    """管理与单个 MCP Server 的连接"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: ClientSession = None
        self._tools_cache = None          # 缓存工具列表，避免重复请求

    async def connect(self):
        """建立连接并初始化会话"""

        if self.config.transport == "stdio":
            # stdio: 启动子进程
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
            )
            self._transport = stdio_client(server_params)

        elif self.config.transport == "http":
            # http: 连接远程服务
            self._transport = streamable_http_client(self.config.url)

        # 建立会话
        read, write = await self._transport.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()

    async def list_tools(self):
        """发现服务器上的所有工具"""
        if self._tools_cache is None:
            result = await self.session.list_tools()
            self._tools_cache = result.tools
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用工具并返回结果文本"""
        result = await self.session.call_tool(name, arguments=arguments)
        # MCP 返回的是 content 列表，取第一个文本内容
        if result.content:
            return result.content[0].text
        return ""

    async def close(self):
        """断开连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._transport:
            await self._transport.__aexit__(None, None, None)
```

#### mcp/tool_adapter.py — 适配器（最关键的一层）

```python
"""
MCPTool 适配器 — 把远程 MCP 工具"伪装"成本地 Tool

这是 MCP 集成的核心:
  MCP 工具有自己的 schema 格式  →  MCPTool 翻译成我们的 Tool 接口
  Agent 和 ToolRegistry 完全不知道这个工具其实在远程

数据流:
  MCP Server 的 tool schema
    → MCPTool 翻译为 ToolParameter + to_openai_schema()
  Agent 调 execute("mcp_tool_name", params)
    → MCPTool.run() 内部走 MCPClient.call_tool() 远程调用
"""
import asyncio
from typing import Dict, Any, List
from tools.base import Tool, ToolParameter


class MCPTool(Tool):
    """把一个 MCP 远程工具包装成本地 Tool 对象"""

    def __init__(self, mcp_client, mcp_tool_schema):
        """
        Args:
            mcp_client: MCPClient 实例（已连接）
            mcp_tool_schema: MCP 协议返回的工具 schema 对象
        """
        super().__init__(
            name=mcp_tool_schema.name,
            description=mcp_tool_schema.description or "",
        )
        self._client = mcp_client
        self._mcp_schema = mcp_tool_schema

    def get_parameters(self) -> List[ToolParameter]:
        """把 MCP 的 JSON Schema 转成我们的 ToolParameter 列表"""
        params = []
        schema = self._mcp_schema.inputSchema  # MCP 工具的参数定义

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required,
                default=prop.get("default"),
            ))
        return params

    def run(self, parameters: Dict[str, Any]) -> str:
        """调用远程 MCP 工具

        注意: MCP 是异步的，这里用 asyncio.run() 桥接到同步
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有事件循环（如 Jupyter），用 nest_asyncio 或 create_task
                import nest_asyncio
                nest_asyncio.apply()
            return asyncio.run(
                self._client.call_tool(self.name, parameters)
            )
        except Exception as e:
            return f"MCP 工具 '{self.name}' 调用失败: {e}"
```

### 3.4 如何注册 MCP 工具到 ToolRegistry

```python
"""
完整流程: 连接 MCP Server → 发现工具 → 注册到 ToolRegistry
"""
import asyncio
from mcp_module.client import MCPClient
from mcp_module.config import MCPServerConfig
from mcp_module.tool_adapter import MCPTool
from tools import ToolRegistry


async def register_mcp_tools(registry: ToolRegistry, config: MCPServerConfig):
    """连接一个 MCP Server，把它的所有工具注册到 registry"""

    # 1. 连接
    client = MCPClient(config)
    await client.connect()

    # 2. 发现工具
    tools = await client.list_tools()

    # 3. 逐个注册
    for mcp_tool_schema in tools:
        adapter = MCPTool(client, mcp_tool_schema)
        registry.register(adapter)
        # 就这一行！MCP 工具变成了普通 Tool，Agent 毫无感知


# main.py 中使用
async def main():
    registry = ToolRegistry()

    # 注册本地工具
    registry.register(CalculatorTool())

    # 注册 MCP 远程工具
    github_config = MCPServerConfig(
        name="github",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
    )
    await register_mcp_tools(registry, github_config)

    # 现在 registry 里既有本地工具，也有 MCP 工具
    # Agent 一视同仁地使用它们
    print(registry.tool_names)
    # → ["calculator", "create_issue", "list_repos", "search_code", ...]
```

### 3.5 MCP 数据流全景

```
用户: "帮我在 GitHub 上创建一个 issue"
  │
  ▼
Agent.run()
  │
  ▼
LLM 看到所有工具的 schema（本地 + MCP 的混在一起）
  │
  ▼
LLM 返回 tool_call: { name: "create_issue", arguments: {...} }
  │
  ▼
ToolRegistry.execute("create_issue", {...})
  │
  ▼
找到 MCPTool 实例（不是本地 Tool）
  │
  ▼
MCPTool.run(parameters)
  │
  ▼
MCPClient.call_tool("create_issue", {...})
  │
  ▼ (MCP 协议 - stdio/http)
  │
MCP Server (GitHub)
  │ 实际调用 GitHub API
  │
  ▼
返回结果: "Issue #42 created"
  │
  ▼ (原路返回)
  │
Agent 拿到结果，喂给 LLM
  │
  ▼
LLM: "已在 GitHub 上创建了 Issue #42"
```

---

## 四、Skill 系统设计

### 4.1 Skill 是什么结构

```python
一个 Skill 包含三样东西:

┌────────────────────────────────────┐
│           Skill: "数据分析师"        │
│                                    │
│  1. 工具集                          │
│     ├── SQLQueryTool               │
│     ├── ChartGeneratorTool         │
│     └── CSVParserTool              │
│                                    │
│  2. 专属 system prompt              │
│     "你是一个数据分析师，擅长..."    │
│     "分析数据时请先查询再可视化..."   │
│                                    │
│  3. 行为配置                        │
│     max_iterations: 10             │
│     auto_visualize: true           │
└────────────────────────────────────┘
```

### 4.2 需要新增的文件

```
myagents/
├── skills/                      ← 新增目录
│   ├── __init__.py
│   ├── base.py                  ← Skill 基类
│   ├── manager.py               ← SkillManager 加载/卸载技能
│   └── builttin/                ← 内置技能
│       ├── __init__.py
│       ├── coder.py             ← "程序员"技能
│       └── researcher.py        ← "研究员"技能
```

### 4.3 Skill 基类设计

```python
# skills/base.py

from abc import ABC
from typing import List, Optional
from tools.base import Tool


class Skill(ABC):
    """技能基类

    一个 Skill = 工具集 + 专属提示词 + 行为配置

    子类只需覆盖三个属性即可:
        - get_tools()         → 这个技能包含哪些工具
        - system_prompt       → 这个技能的专属人设
        - config              → 这个技能的行为参数
    """

    name: str = ""
    description: str = ""
    system_prompt: str = ""

    def get_tools(self) -> List[Tool]:
        """返回该技能提供的工具列表"""
        return []

    def get_config(self) -> dict:
        """返回该技能的行为配置"""
        return {}

    def on_activate(self):
        """技能被激活时的回调（可选）"""
        pass

    def on_deactivate(self):
        """技能被停用时的回调（可选）"""
        pass
```

### 4.4 实际的 Skill 实现示例

```python
# skills/builttin/researcher.py

class ResearcherSkill(Skill):
    """研究员技能 — 让 Agent 变成一个能搜索和总结信息的研究员"""

    name = "researcher"
    description = "互联网研究员，擅长搜索、整理和总结信息"

    system_prompt = """你是一位专业的互联网研究员。
当用户提出问题时，你应该:
1. 先用搜索工具查找相关信息
2. 综合多个来源的信息
3. 用清晰的结构组织你的回答
4. 标注信息来源

请确保你的回答准确、全面、有条理。"""

    def get_tools(self) -> List[Tool]:
        return [
            SearchTool(),           # 网页搜索
            # WikipediaTool(),      # 维基百科（未来可加）
            # ArxivTool(),          # 论文搜索（未来可加）
        ]

    def get_config(self) -> dict:
        return {
            "max_iterations": 5,    # 最多搜索5轮
        }
```

```python
# skills/builttin/coder.py

class CoderSkill(Skill):
    """程序员技能 — 让 Agent 能写代码和做数学计算"""

    name = "coder"
    description = "Python 程序员，擅长编程和数学计算"

    system_prompt = """你是一位资深 Python 程序员。
请遵循以下规范:
- 遵循 PEP 8 编码规范
- 添加类型注解和文档字符串
- 优先考虑代码的可读性和性能
- 需要计算时使用 calculator 工具"""

    def get_tools(self) -> List[Tool]:
        return [
            CalculatorTool(),
            # CodeExecutorTool(),   # 代码执行（未来可加）
        ]
```

### 4.5 SkillManager — 技能管理器

```python
# skills/manager.py

class SkillManager:
    """技能管理器

    负责:
      - 加载/卸载技能
      - 把技能的工具注册到 ToolRegistry
      - 把技能的 prompt 注入 Agent
      - 支持同时激活多个技能

    使用示例:
        manager = SkillManager(registry=tools)
        manager.activate(ResearcherSkill())    # 激活研究员技能
        manager.activate(CoderSkill())         # 同时激活程序员技能
        # 现在 Agent 同时拥有搜索 + 计算能力
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._active_skills: Dict[str, Skill] = {}

    def activate(self, skill: Skill):
        """激活一个技能"""
        # 1. 注册技能的所有工具到 ToolRegistry
        for tool in skill.get_tools():
            self._registry.register(tool)

        # 2. 记录激活状态
        self._active_skills[skill.name] = skill

        # 3. 回调
        skill.on_activate()

    def deactivate(self, skill_name: str):
        """停用一个技能"""
        skill = self._active_skills.pop(skill_name, None)
        if skill:
            skill.on_deactivate()
            # 注意: 工具不从 registry 中移除
            # 因为可能有其他技能也注册了同名工具

    def get_combined_prompt(self) -> str:
        """合并所有激活技能的 system prompt"""
        prompts = []
        for skill in self._active_skills.values():
            if skill.system_prompt:
                prompts.append(f"## {skill.name} 能力\n{skill.system_prompt}")
        return "\n\n".join(prompts)

    @property
    def active_skills(self) -> List[str]:
        return list(self._active_skills.keys())
```

### 4.6 Agent 集成 SkillManager

```python
# core/agent.py 改造

class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: Optional[ToolRegistry] = None,
        skills: Optional[List[Skill]] = None,     # ← 新增
    ):
        self.tools = tools or ToolRegistry()
        self.skill_manager = SkillManager(self.tools)

        # 激活传入的技能
        if skills:
            for skill in skills:
                self.skill_manager.activate(skill)

    def build_context(self) -> list[dict]:
        context = []

        # 基础 system prompt
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})

        # 技能 prompt（自动注入）
        skill_prompt = self.skill_manager.get_combined_prompt()
        if skill_prompt:
            context.append({"role": "system", "content": skill_prompt})

        # 记忆 + 历史（和之前一样）
        ...

        return context
```

### 4.7 最终使用方式

```python
# main.py

from tools import ToolRegistry
from skills.builttin.researcher import ResearcherSkill
from skills.builttin.coder import CoderSkill
from agents.react_agent import ReActAgent
from core.llm import LLMClient

def main():
    llm = LLMClient()

    # 方式1: 用技能驱动（推荐）
    agent = ReActAgent(
        name="全能助手",
        llm=llm,
        skills=[ResearcherSkill(), CoderSkill()],
    )
    # Agent 自动获得: 搜索 + 计算 能力，并自带两个技能的 prompt

    # 方式2: 手动注册工具（也支持，兼容旧方式）
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    agent = ReActAgent(name="计算助手", llm=llm, tools=tools)

    # 方式3: 运行时动态加载技能
    agent.skill_manager.activate(ResearcherSkill())  # 临时获得搜索能力
    agent.run("帮我搜索...")
    agent.skill_manager.deactivate("researcher")     # 用完卸载
```

---

## 五、MCP + Skill 组合使用

Skill 里也可以包含 MCP 工具，实现远程能力的打包:

```python
class GitHubSkill(Skill):
    """GitHub 技能 — 通过 MCP 连接 GitHub"""

    name = "github"
    description = "GitHub 操作，包括创建 Issue、查看 PR 等"

    system_prompt = "你可以操作用户的 GitHub 仓库..."

    def get_tools(self) -> List[Tool]:
        # 连接 MCP Server 获取工具
        client = MCPClient(MCPServerConfig(
            name="github",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        ))
        asyncio.run(client.connect())

        mcp_tools = asyncio.run(client.list_tools())
        return [MCPTool(client, schema) for schema in mcp_tools]
```

使用:

```python
agent = ReActAgent(
    name="开发助手",
    llm=llm,
    skills=[CoderSkill(), GitHubSkill()],
)

agent.run("帮我在 myagents 仓库创建一个 bug issue")
# Agent 自动调用 MCP GitHub 工具完成
```

---

## 六、完整架构全景图

```
用户
 │
 ▼
main.py ──── 组装: LLM + Skills + Agent
 │
 ▼
Agent
 │
 ├── SkillManager
 │     ├── ResearcherSkill
 │     │     ├── system_prompt: "你是研究员..."
 │     │     └── tools: [SearchTool]
 │     │
 │     ├── CoderSkill
 │     │     ├── system_prompt: "你是程序员..."
 │     │     └── tools: [CalculatorTool]
 │     │
 │     └── GitHubSkill
 │           ├── system_prompt: "你可以操作GitHub..."
 │           └── tools: [MCPTool×N]  ← 来自 MCP Server
 │                         │
 │                         ▼
 │                   MCPClient ──MCP协议──→ GitHub MCP Server
 │
 ├── ToolRegistry（自动聚合所有技能的工具）
 │     ├── calculator     (本地，来自 CoderSkill)
 │     ├── web_search     (本地，来自 ResearcherSkill)
 │     ├── create_issue   (远程MCP，来自 GitHubSkill)
 │     ├── list_repos     (远程MCP，来自 GitHubSkill)
 │     └── ...
 │
 ├── LLMClient
 │     └── chat(messages, tools=registry.get_all_schemas())
 │
 └── MemoryManager
       └── 短期 → 长期 → 核心
```

---

## 七、实施顺序

在 ARCHITECTURE.md 第 1~6 步（core 层改造）完成之后:

```
第 7 步: mcp/config.py + mcp/client.py        ← MCP 连接能力
第 8 步: mcp/tool_adapter.py                   ← MCPTool 适配器
第 9 步: skills/base.py + skills/manager.py    ← Skill 系统
第10步: skills/builttin/researcher.py 等       ← 内置技能
第11步: Agent 基类集成 SkillManager             ← 打通
第12步: 写一个 MCP Server 示例                  ← 验证双向（可选）
```

优先级: **先完成 core 层改造（第1~6步）** → 再加 MCP 和 Skill。
因为 MCP 和 Skill 都依赖一个能正常工作的 Agent + ToolRegistry + LLM 链路。

---

## 八、你也可以把自己的工具暴露为 MCP Server

不只是连接别人的 MCP Server，你也可以把自己的工具发布出去:

```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyAgents Tools")

@mcp.tool()
def calculator(expression: str) -> str:
    """数学计算"""
    # 复用已有的计算逻辑
    from tools.builttin.calculator_tool import CalculatorTool
    tool = CalculatorTool()
    return tool.run({"expression": expression})

@mcp.tool()
def web_search(query: str) -> str:
    """网页搜索"""
    from tools.builttin.search import SearchTool
    tool = SearchTool()
    return tool.run({"query": query})

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

这样别人也可以通过 MCP 协议使用你的工具。
