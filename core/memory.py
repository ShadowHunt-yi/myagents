from datetime import datetime,timedelta
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

MemoryType = Literal["short", "long", "core"]


class MemoryItem(BaseModel):
    """单条记忆"""

    content: str
    memory_type: MemoryType
    importance: int = 1
    source: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    access_count: int = 0
    last_access: Optional[datetime] = None

    def touch(self):
        self.access_count += 1
        self.last_access = datetime.now()

    def __str__(self) -> str:
        return f"[{self.memory_type}] {self.content}"


class MemoryManager:
    """记忆管理器"""

    def __init__(self):
        self._short_memory: list[MemoryItem] = []
        self._long_memory: list[MemoryItem] = []
        self._core_memory: list[MemoryItem] = []

    def add(self, memory: MemoryItem):
        if memory.memory_type == "short":
            self._short_memory.append(memory)
        elif memory.memory_type == "long":
            self._long_memory.append(memory)
        elif memory.memory_type == "core":
            self._core_memory.append(memory)
        else:
            raise ValueError(f"未知记忆类型: {memory.memory_type}")

    def add_short(self, content: str, importance: int = 1):
        self._short_memory.append(
            MemoryItem(content=content, memory_type="short", importance=importance)
        )

    def add_long(self, content: str, importance: int = 1):
        self._long_memory.append(
            MemoryItem(content=content, memory_type="long", importance=importance)
        )

    def add_core(self, content: str, importance: int = 1):
        self._core_memory.append(
            MemoryItem(content=content, memory_type="core", importance=importance)
        )

    def consolidate(self):
        """短期记忆整理为长期记忆"""

        new_long = []
        for mem in self._short_memory:
            score = mem.importance + mem.access_count
            if score >= 5:
                mem.memory_type = "long"
                new_long.append(mem)

        for mem in new_long:
            self._short_memory.remove(mem)
            self._long_memory.append(mem)

    def decay_short(self):
        """清理短期记忆"""

        now = datetime.now()

        self._short_memory = [
            m for m in self._short_memory if now - m.timestamp < timedelta(minutes=30)
        ]

    def promote_core(self):
        """长期记忆升级为核心记忆"""

        new_core = []
        for mem in self._long_memory:
            score = mem.importance + mem.access_count
            if score >= 10:
                mem.memory_type = "core"
                new_core.append(mem)

        for mem in new_core:
            self._long_memory.remove(mem)
            self._core_memory.append(mem)

    def get_short(self) -> list[MemoryItem]:
        return self._short_memory.copy()

    def get_long(self) -> list[MemoryItem]:
        return self._long_memory.copy()

    def get_core(self) -> list[MemoryItem]:
        return self._core_memory.copy()

    def get_all(self) -> list[MemoryItem]:
        return [
            *self._core_memory,
            *self._long_memory,
            *self._short_memory,
        ]

    def clear_short(self):
        self._short_memory.clear()

    def clear_long(self):
        self._long_memory.clear()

    def clear_core(self):
        self._core_memory.clear()

    def memory_cycle(self):
        self.consolidate()
        self.promote_core()
        self.decay_short()