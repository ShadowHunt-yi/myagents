"""
数据库连接管理

从 .env 读取 DATABASE_URL，创建 SQLAlchemy engine 和 session 工厂。
提供 init_db() 用于自动建表。
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/myagents",
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db():
    """根据 ORM 模型自动建表（已存在的表不会重建）"""
    from core.models import MemoryItemDB  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_session():
    """获取一个数据库 session，用完需要 close"""
    return SessionLocal()
