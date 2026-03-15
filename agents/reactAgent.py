import json
from core.llm import LLMClient
from core.agent import Agent
from tools import ToolRegistry

SYSTEM_PROMPT = "你是一个智能助手，请根据用户的问题，合理使用工具来获取信息并给出最终回答。"

MAX_ITERATIONS = 5


class ReactAgent(Agent):
    """
    基于 OpenAI Function Calling 的 ReAct Agent。

    流程: 用户输入 → LLM 决定调用工具 → 执行工具 → 观察结果 → 再次思考 → ... → 最终回答
    """

    def __init__(self, llm: LLMClient, tools: ToolRegistry):
        super().__init__(name="ReAct Agent", llm=llm, tools=tools)

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n{'='*50}")
        print(f"用户: {input_text}")
        print(f"{'='*50}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]

        for i in range(MAX_ITERATIONS):
            response = self.chat(messages, use_tools=True)

            # LLM 选择了工具
            if response.tool_calls:
                # 先把 assistant 的工具调用消息加入上下文
                assistant_msg = {"role": "assistant", "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]}
                if response.content:
                    assistant_msg["content"] = response.content
                messages.append(assistant_msg)

                # 逐个执行工具，把结果加入上下文
                for tc in response.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"\n[工具调用] {name}({args})")

                    result = self.execute_tool(name, args)
                    print(f"[工具结果] {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                # 没选工具 → 最终回答
                print(f"\n[最终回答] {response.content}")
                return response.content or "Agent 未能生成回答。"

        return "Agent 已达到最大迭代次数，无法得出结论。"
