"""
记忆管理器 — PostgreSQL 持久化

通过 SQLAlchemy 操作 memory_items 表，
对外接口保持不变（add_short / get_core / memory_cycle 等）。
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Literal, Optional, List

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.models import MemoryItemDB

MemoryType = Literal["short", "long", "core"]


class MemoryItem(BaseModel):
    """记忆业务对象（用于业务层传递，不直接操作数据库）"""

    id: Optional[int] = None
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


def _db_to_item(row: MemoryItemDB) -> MemoryItem:
    """ORM 行 → Pydantic 业务对象"""
    return MemoryItem(
        id=row.id,
        content=row.content,
        memory_type=row.memory_type,
        importance=row.importance,
        source=row.source,
        timestamp=row.timestamp,
        metadata=row.metadata_ or {},
        access_count=row.access_count,
        last_access=row.last_access,
    )


class MemoryManager:
    """记忆管理器 — PostgreSQL 版"""

    def __init__(self, session: Session):
        self._session = session

    # ---- 写入 ----

    def _add(self, content: str, memory_type: MemoryType, importance: int = 1, source: Optional[str] = None):
        row = MemoryItemDB(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source=source,
            timestamp=datetime.now(),
        )
        self._session.add(row)
        self._session.commit()

    def add(self, memory: MemoryItem):
        self._add(memory.content, memory.memory_type, memory.importance, memory.source)

    def add_short(self, content: str, importance: int = 1):
        self._add(content, "short", importance)

    def add_long(self, content: str, importance: int = 1):
        self._add(content, "long", importance)

    def add_core(self, content: str, importance: int = 1):
        self._add(content, "core", importance)

    # ---- 读取 ----

    def _query(self, memory_type: MemoryType) -> List[MemoryItem]:
        rows = (
            self._session.query(MemoryItemDB)
            .filter(MemoryItemDB.memory_type == memory_type)
            .order_by(MemoryItemDB.timestamp)
            .all()
        )
        return [_db_to_item(r) for r in rows]

    def get_short(self) -> List[MemoryItem]:
        return self._query("short")

    def get_long(self) -> List[MemoryItem]:
        return self._query("long")

    def get_core(self) -> List[MemoryItem]:
        return self._query("core")

    def get_all(self) -> List[MemoryItem]:
        return [*self.get_core(), *self.get_long(), *self.get_short()]

    # ---- 记忆生命周期 ----

    def consolidate(self):
        """短期记忆 → 长期记忆（importance + access_count >= 5）"""
        self._session.query(MemoryItemDB).filter(
            MemoryItemDB.memory_type == "short",
            (MemoryItemDB.importance + MemoryItemDB.access_count) >= 5,
        ).update({"memory_type": "long"}, synchronize_session="fetch")
        self._session.commit()

    def promote_core(self):
        """长期记忆 → 核心记忆（importance + access_count >= 10）"""
        self._session.query(MemoryItemDB).filter(
            MemoryItemDB.memory_type == "long",
            (MemoryItemDB.importance + MemoryItemDB.access_count) >= 10,
        ).update({"memory_type": "core"}, synchronize_session="fetch")
        self._session.commit()

    def decay_short(self):
        """清理 30 分钟前的短期记忆"""
        cutoff = datetime.now() - timedelta(minutes=30)
        self._session.query(MemoryItemDB).filter(
            MemoryItemDB.memory_type == "short",
            MemoryItemDB.timestamp < cutoff,
        ).delete(synchronize_session="fetch")
        self._session.commit()

    def memory_cycle(self):
        """执行一轮记忆整理"""
        self.consolidate()
        self.promote_core()
        self.decay_short()

    # ---- 清除 ----

    def clear_short(self):
        self._session.query(MemoryItemDB).filter(MemoryItemDB.memory_type == "short").delete()
        self._session.commit()

    def clear_long(self):
        self._session.query(MemoryItemDB).filter(MemoryItemDB.memory_type == "long").delete()
        self._session.commit()

    def clear_core(self):
        self._session.query(MemoryItemDB).filter(MemoryItemDB.memory_type == "core").delete()
        self._session.commit()
