"""
计算器工具 - 继承 Tool 基类的完整实现示例

演示了如何用"类继承"方式编写一个工具:
    1. 继承 Tool
    2. 实现 get_parameters() → 声明参数
    3. 实现 run() → 执行逻辑
"""

import ast
import operator
import math
from typing import Dict, Any, List
from ..base import Tool, ToolParameter


class CalculatorTool(Tool):
    """数学计算工具

    支持: 四则运算(+, -, *, /), sqrt(), pi
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="数学计算工具，支持四则运算(+,-,*,/)和 sqrt 函数",
        )

        self._operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        self._functions = {
            "sqrt": math.sqrt,
            "pi": math.pi,
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式，如 '2+3*4' 或 'sqrt(16)'",
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = parameters.get("expression", "").strip()
        if not expression:
            return "计算表达式不能为空"

        try:
            node = ast.parse(expression, mode="eval")
            result = self._eval_node(node.body)
            return str(result)
        except ZeroDivisionError:
            return "错误: 除数不能为零"
        except Exception:
            return f"计算失败，请检查表达式格式: {expression}"

    def _eval_node(self, node):
        """递归求值 AST 节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self._functions and callable(self._functions[func_name]):
                args = [self._eval_node(arg) for arg in node.args]
                return self._functions[func_name](*args)
            raise ValueError(f"不支持的函数: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self._functions:
                return self._functions[node.id]
            raise ValueError(f"未知变量: {node.id}")
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")
