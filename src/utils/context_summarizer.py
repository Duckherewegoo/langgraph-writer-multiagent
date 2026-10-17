import logging
from typing import List, Optional
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
你是对话历史压缩总结助手。
任务：把下面多轮用户与AI的对话，浓缩成一段精简、完整、信息不丢失的摘要文本。
要求：
1. 保留用户核心诉求、关键数据、提问目标、工具返回的重要结果；
2. 删除重复闲聊、冗余修饰；
3. 摘要简短精炼，不超过200字；
4. 不要新增不存在的信息，不编造内容。
    """.strip()),
    ("human", "待压缩对话记录：\n{chat_history}")
])


def summarize_chat_history(llm: BaseChatModel, chat_messages: List[BaseMessage]) -> str:
    """调用LLM对一段对话消息列表生成摘要文本"""
    history_blocks = []
    for msg in chat_messages:
        if isinstance(msg, HumanMessage):
            history_blocks.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            history_blocks.append(f"AI：{msg.content}")
        elif isinstance(msg, dict):
            #兼容 dict 格式的轻量消息
            role = "用户" if msg.get("role") == "user" else "AI"
            history_blocks.append(f"{role}：{msg.get('content', '')}")
            
    full_text = "\n".join(history_blocks)

    #限制喂给总结模型的最大长度，防止总结阶段炸裂
    # 简单粗暴截断到前 8000 字符（约 4k token）
    safe_text = full_text[:8000] 

    try:
        chain = SUMMARY_PROMPT | llm
        summary = chain.invoke({"chat_history": safe_text})
        return summary.content.strip()
    except Exception as e:
        logger.error(f"上下文摘要生成失败: {e}")
        # 果总结失败，直接返回一个简单的截断标记
        return f"【早前对话因过长已直接截断】"


def compress_context_messages(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    max_recent_rounds: int = 6,
    token_threshold: int = 1800
) -> List[BaseMessage]:
    """核心工具：自动压缩上下文消息"""
    system_msgs: List[SystemMessage] = []
    chat_msgs: List[BaseMessage] = []
    
    for m in messages:
        if isinstance(m, SystemMessage):
            system_msgs.append(m)
        else:
            chat_msgs.append(m)

    if not chat_msgs:
        return system_msgs

    #计算总长度时，必须把 SystemMessage 也算进去
    sys_text_len = sum(len(str(m.content)) for m in system_msgs)
    total_text_len = sum(len(str(m.content)) for m in chat_msgs) + sys_text_len
    total_token_est = total_text_len * 2  # 粗略估算

    # 未超阈值，直接原样返回
    if total_token_est < token_threshold:
        return system_msgs + chat_msgs

    #按“对”切分，避免截断导致一问一答分离
    # 我们假设 max_recent_rounds 代表“最近 N 条消息”，但必须保证最新的消息不被拆散
    split_idx = len(chat_msgs) - max_recent_rounds
    if split_idx <= 1:
        # 如果切分点太靠前，干脆全保留（总比只留一条好）
        return system_msgs + chat_msgs

    #对齐切分点。如果切在 AI 的消息上，往前挪一位，保证从 Human 开始
    while split_idx < len(chat_msgs) and isinstance(chat_msgs[split_idx], AIMessage):
        split_idx += 1
        
    if split_idx >= len(chat_msgs):
        # 如果挪到了末尾（说明全是 AI 消息，罕见），保守放弃总结
        return system_msgs + chat_msgs

    old_history = chat_msgs[:split_idx]
    recent_history = chat_msgs[split_idx:]

    # 旧历史生成摘要，封装成一条系统/摘要消息
    summary_text = summarize_chat_history(llm, old_history)
    summary_msg = SystemMessage(content=f"【历史对话摘要】{summary_text}")

    # 重组最终上下文
    return system_msgs + [summary_msg] + recent_history


def auto_trim_and_summarize(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    max_retain: int = 8,
    threshold_tokens: int = 2000
) -> List[BaseMessage]:
    """对外简易入口"""
    return compress_context_messages(
        llm=llm,
        messages=messages,
        max_recent_rounds=max_retain,
        token_threshold=threshold_tokens
    )
