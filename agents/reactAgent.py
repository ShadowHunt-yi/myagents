import json
from core.llm import LLMClient
from core.agent import Agent
from core.message import Message
from tools import ToolRegistry

SYSTEM_PROMPT = """你是一个智能助手，拥有以下工具可以使用。
当用户的问题涉及实时信息、价格、新闻、数据查询等你不确定的内容时，必须优先调用工具获取信息，而不是直接回答或反问用户。
只有在工具无法解决的问题时才直接回答。"""


class ReactAgent(Agent):
    """
    基于 OpenAI Function Calling 的 ReAct Agent。

    流程: 用户输入 → LLM 决定调用工具 → 执行工具 → 观察结果 → 再次思考 → ... → 最终回答
    对话历史通过基类的 _history 维护，跨轮次保持上下文。
    """

    def __init__(self, llm: LLMClient, tools: ToolRegistry):
        super().__init__(
            name="ReAct Agent",
            llm=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n{'='*50}")
        print(f"用户: {input_text}")
        print(f"{'='*50}")

        # 记录用户消息到历史
        self.add_message(Message(role="user", content=input_text))

        for i in range(self.max_iterations):
            # 用 build_context() 构建完整上下文（system + 记忆 + 历史）
            messages = self.build_context()
            response = self.chat(messages, use_tools=self.enable_tool_calls)

            if response.tool_calls:
                # 记录 assistant 的工具调用到历史
                self.add_message(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                # 逐个执行工具，把结果记录到历史
                for tc in response.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"\n[工具调用] {name}({args})")

                    result = self.execute_tool(name, args)
                    print(f"[工具结果] {result}")

                    self.add_message(Message(
                        role="tool",
                        tool_call_id=tc.id,
                        name=name,
                        content=result,
                    ))
            else:
                # 没选工具 → 最终回答，记录到历史
                answer = response.content or "Agent 未能生成回答。"
                self.add_message(Message(role="assistant", content=answer))
                print(f"\n[最终回答] {answer}")
                return answer

        return "Agent 已达到最大迭代次数，无法得出结论。"
