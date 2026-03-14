from llm import LLMClient
from tools import ToolExecutor
from tools import search
from agent import planSolveAgent, reactAgent, reflectionAgent


def main():
    # 1. 创建LLM客户端 (连接到 useModel.py 启动的本地服务)
    llm = LLMClient()

    # 2. 创建工具执行器并注册工具
    executor = ToolExecutor()
    executor.register(
        name="search",
        description="搜索互联网获取实时信息",
        parameters={"query": "搜索关键词"},
        func=search,
    )

    # 3. 创建Agent并注入LLM和工具链
    # agent = reactAgent(llm, executor)
    agent = planSolveAgent(llm)
    # agent = reflectionAgent(llm)
    # 4. 交互循环
    print("Agent 已就绪 (输入 quit 退出)")
    print(f"已加载工具: {executor.get_tool_names()}")
    print()

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        agent.run(user_input)


if __name__ == "__main__":
    main()
