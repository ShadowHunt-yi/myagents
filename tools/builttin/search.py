"""
搜索工具 - 基于 SerpApi 的网页搜索

演示了如何把一个依赖外部 API 的功能封装为标准 Tool。
"""

import os
from typing import Dict, Any, List

from dotenv import load_dotenv

from ..base import Tool, ToolParameter

load_dotenv()


class SearchTool(Tool):
    """网页搜索工具

    基于 SerpApi，智能解析搜索结果，优先返回直接答案。
    """

    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索互联网获取实时信息，输入搜索关键词即可",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词",
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        query = parameters.get("query", "").strip()
        if not query:
            return "搜索关键词不能为空"

        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误: SERPAPI_API_KEY 未在 .env 文件中配置"

        try:
            from serpapi import Client as SerpApiClient

            client = SerpApiClient(api_key=api_key)
            results = client.search({
                "engine": "google",
                "q": query,
                "gl": "cn",
                "hl": "zh-cn",
            })
            return self._parse_results(query, results)

        except ImportError:
            return "错误: 需要安装 serpapi 包 (pip install serpapi)"
        except Exception as e:
            return f"搜索时发生错误: {e}"

    def _parse_results(self, query: str, results: dict) -> str:
        """智能解析搜索结果，优先返回最直接的答案"""
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"没有找到关于 '{query}' 的信息"
