"""
ORM 模型 — memory_items 表
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from core.db import Base


class MemoryItemDB(Base):
    """记忆条目 ORM 模型"""

    __tablename__ = "memory_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    memory_type = Column(String(10), nullable=False, index=True)  # short/long/core
    importance = Column(Integer, default=1)
    source = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    metadata_ = Column("metadata", JSON, default=dict)
    access_count = Column(Integer, default=0)
    last_access = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<MemoryItemDB(id={self.id}, type={self.memory_type}, content={self.content[:30]}...)>"
