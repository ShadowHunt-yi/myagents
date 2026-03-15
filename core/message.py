from typing import Dict, Any, Optional, Literal, List
from datetime import datetime
from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """上下文消息

    支持三种消息格式:
    - 普通消息: role + content
    - assistant 工具调用: role="assistant", tool_calls=[...]
    - tool 结果: role="tool", tool_call_id="...", name="...", content="..."
    """

    role: MessageRole
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Function Calling 相关字段
    tool_calls: Optional[List[Any]] = None  # assistant 消息的工具调用列表
    tool_call_id: Optional[str] = None  # tool 消息的关联 ID
    name: Optional[str] = None  # tool 消息的工具名

    def to_dict(self) -> Dict[str, Any]:
        """转换为 OpenAI API 消息格式"""
        if self.role == "assistant" and self.tool_calls:
            # assistant 发起工具调用
            msg: Dict[str, Any] = {"role": "assistant"}
            if self.content:
                msg["content"] = self.content
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
            return msg

        if self.role == "tool" and self.tool_call_id:
            # 工具执行结果
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }

        # 普通消息
        return {"role": self.role, "content": self.content or ""}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
