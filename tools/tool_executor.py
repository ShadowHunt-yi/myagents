from typing import Dict, Any, Callable, List


class ToolExecutor:
    """
    工具执行器：负责注册、描述和执行工具。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        func: Callable,
    ):
        """
        注册一个工具。

        Args:
            name: 工具名称
            description: 工具用途描述
            parameters: 参数说明，如 {"query": "搜索关键词"}
            func: 工具执行函数
        """
        self.tools[name] = {
            "description": description,
            "parameters": parameters,
            "func": func,
        }
        print(f"[ToolExecutor] 已注册工具: {name}")

    def execute(self, name: str, **kwargs) -> str:
        """
        根据名称执行工具，返回结果字符串。
        """
        tool = self.tools.get(name)
        if not tool:
            return f"错误: 未找到工具 '{name}'"
        try:
            result = tool["func"](**kwargs)
            return str(result)
        except Exception as e:
            return f"工具 '{name}' 执行出错: {e}"

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_prompt(self) -> str:
        """
        生成工具描述文本，用于注入到Agent的系统提示词中。
        """
        if not self.tools:
            return "当前没有可用工具。"

        lines = []
        for name, info in self.tools.items():
            params_desc = ", ".join(
                f"{k}({v})" for k, v in info["parameters"].items()
            )
            lines.append(f"- {name}: {info['description']}  参数: {params_desc}")
        return "\n".join(lines)
