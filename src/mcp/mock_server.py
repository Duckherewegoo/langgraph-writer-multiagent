from typing import Any

import jieba

from utils.context_summarizer import auto_trim_and_summarize


class MockMCPServer:
    """模拟MCP协议服务端，提供外部工具能力"""

    def __init__(self, llm: Any):
        """
        :param llm: LangChain 兼容的 Chat 模型，或原生 OpenAI client
        """
        self.llm = llm
        self._tools: dict[str, dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认MCP工具 (使用标准 JSON Schema 定义参数)"""
        self._tools["mcp_summarize"] = {
            "description": "对长文本进行摘要提炼",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待摘要的长文本内容"}
                },
                "required": ["text"]
            },
            "handler": self._summarize_handler
        }
        self._tools["mcp_keyword_extract"] = {
            "description": "提取文本中的核心关键词",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待提取关键词的文本"}
                },
                "required": ["text"]
            },
            "handler": self._keyword_handler
        }

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """列出所有可用工具（MCP标准接口）"""
        return self._tools

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """调用工具（MCP标准接口）"""
        if tool_name not in self._tools:
            return f"MCP错误：工具 {tool_name} 不存在"
        
        try:
            # 统一捕获工具执行异常，防止服务端崩溃
            return str(self._tools[tool_name]["handler"](**args))
        except Exception as e:
            return f"MCP工具执行异常: {str(e)}"

    def _summarize_handler(self, text: str) -> str:
        """文本摘要工具：调用注入的LLM对输入文本做摘要提炼"""
        messages = [{"role": "user", "content": text}]

        # 假设 auto_trim_and_summarize 内部兼容 LangChain Chat 模型
        summary = auto_trim_and_summarize(
            llm=self.llm,
            messages=messages,
            max_retain=8,
            threshold_tokens=1000
        )
        
        if not summary:
            return "摘要失败：文本为空或处理异常"

        return f"摘要：{summary[:50]}...（核心内容已提炼，共{len(summary)}字）"

    @staticmethod
    def _keyword_handler(text: str) -> str:
        """提取关键词工具"""
        if not text:
            return "提取失败：文本为空"
            
        # 修复：jieba.cut 返回生成器，必须转 list 才能切片和 len
        words = list(jieba.cut(text, cut_all=True))
        
        # 过滤掉单字和标点，保留真正的词
        valid_words = [w for w in words if len(w.strip()) > 1]
        
        keywords = valid_words[:5] if len(valid_words) >= 5 else valid_words
        if not keywords:
            return "提取关键词：未找到明显关键词"
            
        return f"提取关键词：{', '.join(keywords)}"
