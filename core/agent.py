"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional
from core.message import Message
from core.memory import MemoryManager
from core.llm import LLMClient, LLMResponse
from tools import ToolRegistry


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        tools: Optional[ToolRegistry] = None,
        max_iterations: int = 5,
        enable_tool_calls: bool = True,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        self.enable_tool_calls = enable_tool_calls

        # 对话历史（上下文）
        self._history: list[Message] = []

        # 记忆管理器
        self.memory = MemoryManager()

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        ...

    def chat(
        self,
        messages: list[dict],
        use_tools: bool = False,
    ) -> LLMResponse:
        """调 LLM，可选是否带工具"""
        tools = self.tools.get_all_schemas() if use_tools and self.tools else None
        return self.llm.chat(messages, tools=tools)

    def execute_tool(self, name: str, parameters: dict) -> str:
        """执行工具的便捷方法"""
        return self.tools.execute(name, parameters)

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def build_context(self) -> list[dict]:
        """构建LLM上下文"""
        context: list[dict] = []

        # system prompt
        if self.system_prompt:
            context.append(Message(role="system", content=self.system_prompt).to_dict())

        # 核心记忆
        core_memories = self.memory.get_core()
        if core_memories:
            core_text = "\n".join([m.content for m in core_memories])
            context.append(
                Message(role="system", content=f"核心记忆：\n{core_text}").to_dict()
            )

        # 长期记忆
        long_memories = self.memory.get_long()
        if long_memories:
            long_text = "\n".join([m.content for m in long_memories])
            context.append(
                Message(role="system", content=f"长期记忆：\n{long_text}").to_dict()
            )

        # 短期记忆
        short_memories = self.memory.get_short()
        if short_memories:
            short_text = "\n".join([m.content for m in short_memories])
            context.append(
                Message(role="system", content=f"短期记忆：\n{short_text}").to_dict()
            )

        # 历史对话
        context.extend([msg.to_dict() for msg in self._history])

        return context

    def update_memory_cycle(self):
        """更新记忆等级"""
        self.memory.memory_cycle(self)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
