import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()


class LLMClient:
    """
    LLM客户端，通过OpenAI兼容接口连接到本地模型服务(useModel.py)。
    """

    # 各 provider 的默认配置
    PROVIDER_DEFAULTS = {
        "openai": {"env_key": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
        "modelscope": {"env_key": "MODELSCOPE_API_KEY", "base_url": "https://api-inference.modelscope.cn/v1/"},
        "zhipu": {"env_key": "ZHIPU_API_KEY", "base_url": "https://open.bigmodel.cn/api/paas/v4/"},
        "ollama": {"env_key": None, "base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        "vllm": {"env_key": None, "base_url": "http://localhost:8000/v1", "api_key": "local"},
    }

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID", "Qwen/Qwen1.5-0.5B-Chat")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "120"))

        # 自动检测 provider，然后解析凭证
        self.provider = self._auto_detect_provider(api_key, base_url)
        resolved_key, resolved_url = self._resolve_credentials(api_key, base_url)

        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
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

    def _auto_detect_provider(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
        """自动检测LLM提供商"""
        # 1. 检查特定提供商的环境变量 (最高优先级)
        if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
        if os.getenv("OPENAI_API_KEY"): return "openai"
        if os.getenv("ZHIPU_API_KEY"): return "zhipu"

        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 2. 根据 base_url 判断
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
            if "open.bigmodel.cn" in base_url_lower: return "zhipu"
            if "api.openai.com" in base_url_lower: return "openai"
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower: return "ollama"
                if ":8000" in base_url_lower: return "vllm"
                return "local"
            # 有 base_url 但不匹配任何已知 provider → 自定义第三方服务
            return "custom"

        # 3. 根据 API 密钥格式辅助判断
        if actual_api_key:
            if actual_api_key.startswith("ms-"): return "modelscope"
            if actual_api_key.startswith("sk-"): return "custom"

        return "custom"

    def _resolve_credentials(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> tuple:
        """根据 provider 解析 API 密钥和 base_url"""
        defaults = self.PROVIDER_DEFAULTS.get(self.provider, {})
        env_key = defaults.get("env_key")

        resolved_key = (
            api_key
            or (os.getenv(env_key) if env_key else None)
            or os.getenv("LLM_API_KEY")
            or defaults.get("api_key", "local")
        )
        resolved_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or defaults.get("base_url", "http://127.0.0.1:8000/v1")
        )
        return resolved_key, resolved_url
        
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
