import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()


class LLMClient:
    """
    LLM客户端，通过OpenAI兼容接口连接到本地模型服务(useModel.py)。
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID", "Qwen/Qwen1.5-0.5B-Chat")
        api_key = api_key or os.getenv("LLM_API_KEY", "local")
        base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "120"))

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers={
                "User-Agent": "",
                "x-stainless-lang": "",
                "x-stainless-package-version": "",
                "x-stainless-os": "",
                "x-stainless-arch": "",
                "x-stainless-runtime": "",
                "x-stainless-runtime-version": "",
                "x-stainless-async": "",
                "x-stainless-retry-count": "",
                "x-stainless-read-timeout": "",
            },
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """
        发送消息到LLM并返回完整响应文本。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            collected = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected.append(content)
            print()
            return "".join(collected)

        except Exception as e:
            print(f"LLM调用失败: {e}")
            return None


if __name__ == "__main__":
    llm = LLMClient()
    result = llm.chat(
        [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请做一下自我介绍。"},
        ]
    )
    if result:
        print(f"\n完整响应: {result}")
