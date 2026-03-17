from core.llm import LLMClient
from core.db import init_db
from tools import ToolRegistry
from tools.builttin.search import SearchTool
from tools.builttin.calculator_tool import CalculatorTool
from agents import PlanAndSolveAgent, ReactAgent, ReflectionAgent


def main():
    # 0. 初始化数据库（自动建表）
    init_db()

    # 1. 创建LLM客户端 (连接到 useModel.py 启动的本地服务)
    llm = LLMClient()

    # 2. 创建工具注册中心并注册工具实例
    executor = ToolRegistry()
    executor.register(SearchTool(), CalculatorTool())
    
    # 3. 创建Agent并注入LLM和工具链
    # agent = ReactAgent(llm, executor)
    agent = PlanAndSolveAgent(llm)
    # agent = reflectionAgent(llm)
    # 4. 交互循环
    print("Agent 已就绪 (输入 quit 退出)")
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
