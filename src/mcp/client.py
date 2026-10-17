import asyncio
import sys
from typing import Any

from langchain_core.tools import tool
from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters


class RealMCPClient:
    """真实 MCP 协议客户端 (同步 CLI 版本)"""
    
    def __init__(self, command: str, args: list[str]):
        """
        初始化客户端
        :param command: 启动 MCP 服务端的命令 (如 "python", "npx", "uv")
        :param args: 命令参数 (如 ["mcp_server.py"])
        """
        self.command = command
        self.args = args
        self._tools_cache: dict[str, Any] = {}
        self._connected = False

    def _run_async(self, coro):
        """同步 CLI 里跑异步 MCP 的桥梁"""
        try:
            return asyncio.run(coro)
        except Exception as e:
            print(f"❌ MCP 通信异常: {e}", file=sys.stderr)
            raise

    def connect(self) -> None:
        """测试连接并缓存工具列表"""
        server_params = StdioServerParameters(command=self.command, args=self.args)
        
        async def _connect():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    return response.tools

        try:
            tools = self._run_async(_connect())
            self._tools_cache = {t.name: t for t in tools}
            self._connected = True
        except Exception as e:
            self._connected = False
            raise RuntimeError(f"MCP 服务端连接失败 ({self.command} {' '.join(self.args)}): {e}") from e

    def list_tools(self) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("MCP客户端未连接")
        return self._tools_cache

    def call_tool(self, tool_name: str, args: dict) -> str:
        if not self._connected:
            raise RuntimeError("MCP客户端未连接")

        async def _call():
            server_params = StdioServerParameters(command=self.command, args=self.args)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # 真实调用
                    result = await session.call_tool(tool_name, args)
                    # 将 MCP 的返回格式转为纯字符串
                    return "\n".join([c.text for c in result.content if c.type == "text"])

        return self._run_async(_call())

    def get_langchain_tools(self) -> list:
        """将真实 MCP 工具转换为 LangChain Tool"""
        if not self._connected:
            self.connect()

        tools = []
        for name, info in self.list_tools().items():
            def make_tool(tool_name, tool_desc):
                @tool
                def mcp_tool(**kwargs):
                    try:
                        return self.call_tool(tool_name, kwargs)
                    except Exception as e:
                        # 错误即反馈，绝不崩溃
                        return f"工具执行失败: {str(e)}"
                
                mcp_tool.name = tool_name
                mcp_tool.description = tool_desc.description
                # 动态解析参数 schema（MCP 提供的是 JSON Schema）
                # LangChain 的 @tool 默认能处理普通 kwargs，这里做默认兼容
                return mcp_tool

            tools.append(make_tool(name, info))

        return tools
