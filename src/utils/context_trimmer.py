from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage

def trim_messages(messages: List[BaseMessage], max_count: int = 12) -> List[BaseMessage]:
    """
    截断对话历史，保留最近消息，控制上下文长度。
    增加了安全兜底：防止 ToolMessage 孤悬，防止单条超长。
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    chat_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # 1. 防御单条超长（比如巨大的 CSV 结果）：暴力截断内容，不删消息
    safe_chat_msgs = []
    for m in chat_msgs:
        # 限制单条消息最大 8000 字符（约 4k token）
        if len(str(m.content)) > 8000:
            m.content = str(m.content)[:8000] + "\n...【内容过长已截断】"
        safe_chat_msgs.append(m)

    # 2. 条数没超，直接返回
    if len(safe_chat_msgs) <= max_count:
        return system_msgs + safe_chat_msgs

    # 3. 截断并保证上下文合法性
    split_idx = len(safe_chat_msgs) - max_count
    
    # 如果切出来的第一条是 ToolMessage，往前多留一条（把它依赖的 AIMessage 带上）
    while split_idx > 0 and isinstance(safe_chat_msgs[split_idx], ToolMessage):
        split_idx -= 1

    recent_msgs = safe_chat_msgs[split_idx:]
    return system_msgs + recent_msgs
