"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        # 对话历史（上下文）
        self._history: list[Message] = []
        # 短期记忆（任务级）
        self._short_memory: list[Message] = []

        # 长期记忆（跨任务）
        self._long_memory: list[Message] = []

        # 核心记忆（系统级）
        self._core_memory: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        self.add_message(Message(role="user", content=input_text))
        context = self.build_context()
        response = self.llm.think(context)
        self.add_message(Message(role="assistant", content=response))
        return response

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def add_short_memory(self, message: Message):
        """添加短期记忆"""
        self._short_memory.append(message)

    def get_short_memory(self) -> list[Message]:
        return self._short_memory.copy()

    def clear_short_memory(self):
        """清空短期记忆"""
        self._short_memory.clear()

    def add_long_memory(self, message: Message):
        """添加长期记忆"""
        self._long_memory.append(message)

    def get_long_memory(self) -> list[Message]:
        return self._long_memory.copy()

    def clear_long_memory(self):
        self._long_memory.clear()

    def add_core_memory(self, message: Message):
        """添加核心记忆"""
        self._core_memory.append(message)

    def get_core_memory(self) -> list[Message]:
        return self._core_memory.copy()

    def build_context(self) -> list[Message]:
        """构建LLM上下文"""
        context = []
        context.extend(self._core_memory)
        context.extend(self._long_memory)
        context.extend(self._short_memory)
        context.extend(self._history)
        return context

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
