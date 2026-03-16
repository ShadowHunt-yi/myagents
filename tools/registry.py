"""
工具注册中心

设计思路:
    ToolRegistry 是整个工具系统的"中枢"，负责:
    1. 注册 - 收纳所有 Tool 实例（统一存放在一个 dict 里）
    2. 查找 - 按名称获取工具
    3. Schema - 批量生成所有工具的 OpenAI function calling schema
    4. 执行 - 根据名称 + 参数，找到工具并调用 run()

    为什么合并了原来的 ToolExecutor?
    → 注册和执行本质上是同一个生命周期里的事，拆成两个类
      反而需要在它们之间同步状态，增加复杂度。
      一个 Registry 全搞定，对外只暴露一个入口。

    支持两种注册方式:
    - register(tool)           → 传入 Tool 子类实例（推荐，适合复杂工具）
    - register_function(...)   → 传入普通函数（便捷，适合简单工具）
      内部会自动包装成 FunctionTool
"""

from typing import Dict, Any, List, Callable

from .base import Tool, ToolParameter, FunctionTool


class ToolRegistry:
    """工具注册中心

    使用示例:
        registry = ToolRegistry()

        # 方式1: 注册 Tool 子类实例
        registry.register(CalculatorTool())

        # 方式2: 快速注册函数
        registry.register_function(
            name="echo",
            description="回显",
            parameters=[ToolParameter(name="text", type="string", description="内容")],
            func=lambda text: text
        )

        # 获取所有工具的 OpenAI schema（传给 LLM）
        schemas = registry.get_all_schemas()

        # LLM 返回 tool_call 后，执行工具
        result = registry.execute("calculator", {"expression": "1+1"})
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, *tools: Tool):
        """注册一个 Tool 实例"""
        for tool in tools:
            self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        func: Callable[..., str],
    ):
        """快速注册一个函数为工具

        内部自动包装成 FunctionTool，不需要手写类。

        Args:
            name: 工具名称
            description: 工具描述（LLM 靠这个决定是否调用）
            parameters: 参数定义列表
            func: 实际执行函数，参数名需要和 parameters 中的 name 对应
        """
        tool = FunctionTool(name, description, parameters, func)
        self.register(tool)

    def get(self, name: str) -> Tool:
        """按名称获取工具，找不到则抛出 KeyError"""
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未注册")
        return self._tools[name]

    def execute(self, name: str, parameters: Dict[str, Any]) -> str:
        """查找并执行工具

        这是 Agent 调用工具的统一入口:
            LLM 返回 tool_call → 解析出 name 和 arguments → 调用本方法

        Args:
            name: 工具名称
            parameters: 参数字典

        Returns:
            工具执行结果字符串
        """
        try:
            tool = self.get(name)
        except KeyError:
            return f"错误: 未找到工具 '{name}'"
        try:
            return tool.run(parameters)
        except Exception as e:
            return f"工具 '{name}' 执行出错: {e}"

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """获取所有已注册工具的 OpenAI function calling schema

        直接传给 OpenAI API 的 tools 参数即可:
            response = client.chat.completions.create(
                model="...",
                messages=[...],
                tools=registry.get_all_schemas()
            )
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_descriptions(self) -> str:
        """获取所有工具的文本描述（用于非 function calling 的 prompt 注入）"""
        if not self._tools:
            return "暂无可用工具"
        lines = []
        for tool in self._tools.values():
            params = tool.get_parameters()
            param_str = ", ".join(f"{p.name}({p.type})" for p in params)
            lines.append(f"- {tool.name}: {tool.description}  参数: {param_str}")
        return "\n".join(lines)

    @property
    def tool_names(self) -> List[str]:
        """所有已注册的工具名称"""
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
