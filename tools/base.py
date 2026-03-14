"""
工具系统基类模块

设计思路:
    整个工具系统只有一条主线: 定义 → 注册 → 生成Schema → 执行

    本模块负责"定义"这一环:
    1. ToolParameter - 描述工具的每个参数(名称、类型、是否必填等)
    2. Tool(ABC)    - 工具的标准接口，所有工具都必须实现 run() 和 get_parameters()
                      同时自带 to_openai_schema()，让每个工具知道如何描述自己
    3. FunctionTool - 一个便捷包装器，把普通函数快速变成 Tool 对象
                      适合简单场景，不想写一个完整的类时使用

    为什么 to_openai_schema() 放在 Tool 上而不是 Registry 上?
    → 因为"如何描述自己"是工具自身的职责，Registry 只负责收集和管理
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数定义

    对应 OpenAI function calling schema 中 parameters.properties 的一项

    Attributes:
        name: 参数名，如 "expression"、"query"
        type: JSON Schema 类型，如 "string"、"number"、"array"、"boolean"
        description: 参数说明，LLM 靠这个理解该传什么值
        required: 是否必填，默认 True
        default: 默认值，可选
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具抽象基类

    所有工具(计算器、搜索、文件操作等)都继承这个类。
    只需实现两个方法:
        - get_parameters(): 告诉系统你需要哪些参数
        - run(): 拿到参数后执行具体逻辑

    schema 生成由基类统一处理，子类不用操心。

    使用示例:
        class MyTool(Tool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="做某件事"
                )

            def get_parameters(self) -> List[ToolParameter]:
                return [
                    ToolParameter(name="input", type="string", description="输入内容")
                ]

            def run(self, parameters: Dict[str, Any]) -> str:
                return f"处理了: {parameters['input']}"
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """声明工具需要的参数列表"""
        ...

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具，返回结果字符串

        Args:
            parameters: 参数字典，key 对应 ToolParameter.name
        """
        ...

    def to_openai_schema(self) -> Dict[str, Any]:
        """生成 OpenAI function calling 格式的 schema

        自动根据 get_parameters() 的返回值构建，子类无需重写。

        Returns:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { "type": "object", "properties": {...}, "required": [...] }
                }
            }
        """
        params = self.get_parameters()

        properties = {}
        required = []

        for param in params:
            prop = {"type": param.type, "description": param.description}
            if param.default is not None:
                prop["description"] += f" (默认: {param.default})"
            if param.type == "array":
                prop["items"] = {"type": "string"}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class FunctionTool(Tool):
    """函数包装器 - 把普通函数快速变成 Tool 对象

    适合不想写完整类的简单场景:
        registry.register_function(
            name="echo",
            description="回显输入",
            parameters=[ToolParameter(name="text", type="string", description="内容")],
            func=lambda text: text
        )

    内部会自动创建一个 FunctionTool 实例。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        func: Callable[..., str],
    ):
        super().__init__(name, description)
        self._parameters = parameters
        self._func = func

    def get_parameters(self) -> List[ToolParameter]:
        return self._parameters

    def run(self, parameters: Dict[str, Any]) -> str:
        """调用被包装的函数

        会自动将 parameters dict 解包为关键字参数传入
        """
        try:
            result = self._func(**parameters)
            return str(result)
        except Exception as e:
            return f"工具 '{self.name}' 执行出错: {e}"
