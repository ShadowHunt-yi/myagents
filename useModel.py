"""
本地模型服务启动脚本。
使用 Ollama 作为模型后端，提供 OpenAI 兼容的 API 接口。

使用前:
  1. 安装 Ollama: https://ollama.com/download/windows
  2. 拉取模型: ollama pull qwen2.5:0.5b
  3. 运行此脚本: python useModel.py

Ollama 会自动在 http://localhost:11434 提供服务，
其 OpenAI 兼容接口为 http://localhost:11434/v1 。
"""

import subprocess
import sys
import time
import httpx

MODEL_NAME = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434"


def check_ollama_running() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_model_exists() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(MODEL_NAME in m for m in models)
    except Exception:
        return False


def pull_model():
    print(f"正在拉取模型 {MODEL_NAME} ...")
    subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    print("模型拉取完成。")


def main():
    # 1. 检查 Ollama 是否已运行
    if not check_ollama_running():
        print("Ollama 服务未检测到，正在启动 ollama serve ...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            time.sleep(1)
            if check_ollama_running():
                break
        else:
            print("Ollama 启动失败，请手动运行: ollama serve")
            sys.exit(1)

    print(f"Ollama 服务运行中: {OLLAMA_URL}")

    # 2. 检查模型是否已拉取
    if not check_model_exists():
        pull_model()

    # 3. 预热模型 (发一个空请求让 Ollama 加载模型到内存)
    print(f"预热模型 {MODEL_NAME} ...")
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": "hi", "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            print("模型已加载就绪。")
    except Exception as e:
        print(f"预热失败: {e}，但服务仍可使用。")

    print(f"\nOpenAI 兼容接口: {OLLAMA_URL}/v1")
    print(f"模型: {MODEL_NAME}")
    print("llm_client.py 可直接连接此地址。")
    print("\n按 Ctrl+C 退出。")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n再见!")


if __name__ == "__main__":
    main()
