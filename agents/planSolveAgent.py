import ast
from core.llm import LLMClient
from core.agent import Agent
from core.message import Message

SYSTEM_PROMPT = "你是一个能够将复杂问题拆解为多个步骤并逐步执行的智能体。"

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class PlanAndSolveAgent(Agent):
    """
    Plan-and-Solve Agent: 先规划，后逐步执行。
    Planner 和 Executor 作为内部方法，共享 _history 上下文。
    """

    def __init__(self, llm: LLMClient):
        super().__init__(
            name="Plan-and-Solve Agent",
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
        )

    def run(self, question: str, **kwargs) -> str:
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 记录用户消息
        self.add_message(Message(role="user", content=question))

        # 1. 规划
        plan = self._plan(question)
        if not plan:
            msg = "无法生成有效的行动计划。"
            print(f"\n--- 任务终止 ---\n{msg}")
            self.add_message(Message(role="assistant", content=msg))
            return msg

        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        self.add_message(Message(
            role="assistant", content=f"计划:\n{plan_text}",
            metadata={"type": "plan"},
        ))

        # 2. 逐步执行
        step_history = ""
        result = ""
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")
            result = self._execute_step(question, plan, step_history, step)
            step_history += f"步骤 {i+1}: {step}\n结果: {result}\n\n"
            print(f"✅ 步骤 {i+1} 已完成")

            self.add_message(Message(
                role="assistant", content=result,
                metadata={"type": "step", "step": i + 1},
            ))

        # 3. 最终回答
        print(f"\n--- 任务完成 ---\n最终答案: {result}")
        return result

    def _plan(self, question: str) -> list[str]:
        """调用 LLM 生成行动计划"""
        print("--- 正在生成计划 ---")
        messages = self.build_context()
        messages.append({
            "role": "user",
            "content": PLANNER_PROMPT_TEMPLATE.format(question=question),
        })

        response = self.llm.chat(messages)
        text = response.content or ""

        try:
            plan_str = text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            if isinstance(plan, list):
                print(f"✅ 计划已生成，共 {len(plan)} 步")
                return plan
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
        return []

    def _execute_step(self, question: str, plan: list[str], history: str, step: str) -> str:
        """执行计划中的单个步骤"""
        messages = self.build_context()
        messages.append({
            "role": "user",
            "content": EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            ),
        })

        response = self.llm.chat(messages)
        return response.content or ""
