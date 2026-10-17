from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """智能体全局状态定义"""
    # 对话消息历史，自动追加归约（核心）
    messages: Annotated[list[BaseMessage], add_messages]
    # 循环计数，防止死循环
    loop_count: int
    # 最大允许循环步数
    max_steps: int
    # 人工反馈内容（人机交互用）
    human_feedback: str | None
    # 分析结果缓存
    analysis_result: Any | None
    # 用户上传的CSV文件路径
    csv_file_path: str | None
    # 记录致命异常，供 CLI 顶层判断退出码（如 1 或 2）
    error: str | None
