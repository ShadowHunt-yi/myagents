import re
from typing import Optional
from llm import LLMClient
from tools import ToolExecutor

SYSTEM_PROMPT_TEMPLATE = """你是一个智能助手，可以使用以下工具来帮助回答用户的问题。

可用工具:
{tool_descriptions}

请严格按照以下格式回复:

如果你需要使用工具，请按此格式输出:
Thought: <你的思考过程>
Action: <工具名称>
Action Input: <工具参数值>

如果你已经得到了足够的信息可以直接回答，请按此格式输出:
Thought: <你的思考过程>
Final Answer: <最终回答>

注意:
- 每次只能调用一个工具
- Action 必须是可用工具之一
- 请务必严格遵守上述格式
"""

MAX_ITERATIONS = 5


class reactAgent:
    """
    基于ReAct模式的Agent：Think -> Act -> Observe 循环。
    """

    def __init__(self, llm: LLMClient, tool_executor: ToolExecutor):
        self.llm = llm
        self.tool_executor = tool_executor

    def _build_system_prompt(self) -> str:
        tool_desc = self.tool_executor.get_tool_prompt()
        return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_desc)

    def _parse_action(self, text: str) -> Optional[tuple]:
        """
        从LLM输出中解析 Action 和 Action Input。
        返回 (tool_name, tool_input) 或 None。
        """
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.+)", text)
        if action_match and input_match:
            return action_match.group(1).strip(), input_match.group(1).strip()
        return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """从LLM输出中解析 Final Answer。"""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def run(self, user_input: str) -> str:
        """
        运行Agent主循环。

        流程: 用户输入 -> LLM思考 -> 决定是否使用工具 -> 执行工具 -> 观察结果 -> 再次思考 -> ... -> 最终回答
        """
        print(f"\n{'='*50}")
        print(f"用户: {user_input}")
        print(f"{'='*50}")

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for i in range(MAX_ITERATIONS):
            print(f"\n--- 第 {i+1} 轮思考 ---")
            response = self.llm.chat(messages)

            if response is None:
                return "Agent执行失败: LLM无响应。"

            # 检查是否有最终答案
            final_answer = self._parse_final_answer(response)
            if final_answer:
                print(f"\n[Agent] 最终回答: {final_answer}")
                return final_answer

            # 检查是否需要调用工具
            action = self._parse_action(response)
            if action:
                tool_name, tool_input = action
                print(f"\n[Agent] 调用工具: {tool_name}({tool_input})")

                # 执行工具
                observation = self.tool_executor.execute(
                    tool_name, query=tool_input
                )
                print(f"[Agent] 工具返回: {observation[:200]}...")

                # 将LLM响应和工具结果加入对话历史
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请根据以上信息继续回答。如果信息足够请给出 Final Answer。",
                })
            else:
                # LLM没有按格式输出，直接把回复当作最终答案
                print(f"\n[Agent] 直接回答(无工具调用)")
                return response

        return "Agent已达到最大迭代次数，无法得出结论。"
