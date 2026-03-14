"""Agent基类"""

from abc import ABC
from typing import Optional
from .message import Message
from .memory import MemoryManager
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

        # 记忆管理器
        self.memory = MemoryManager()

    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
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