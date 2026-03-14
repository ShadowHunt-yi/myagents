"""
tools 包 - 工具系统

对外暴露:
    - Tool, ToolParameter, FunctionTool  (定义工具)
    - ToolRegistry                        (注册和执行工具)
    - CalculatorTool, SearchTool          (内置工具)
"""

from .base import Tool, ToolParameter, FunctionTool
from .registry import ToolRegistry
from .builttin.calculator_tool import CalculatorTool
from .builttin.search import SearchTool

__all__ = [
    "Tool",
    "ToolParameter",
    "FunctionTool",
    "ToolRegistry",
    "CalculatorTool",
    "SearchTool",
]
