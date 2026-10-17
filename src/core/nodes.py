from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from .state import AgentState
from utils.content_moderation import moderate_content
from utils.context_trimmer import trim_messages
from tools.calculator import calculator
from tools.csv_analyzer import csv_analyzer
from tools.weather import get_weather
from mcp.client import MockMCPClient

# 建立工具映射表，消灭 if/elif 地狱，新增工具只需在这里注册
LOCAL_TOOLS = {
    "calculator": calculator,
    "csv_analyzer": csv_analyzer,
    "get_weather": get_weather,
}

# TODO: 把超长的 Prompt 抽离出来，未来可改为读文件
SYSTEM_PROMPT = """
你是专业的写作分析助手，具备数据分析、数值计算、信息查询能力。
严格遵循ReAct流程：
1. 能直接回答的问题直接回答
2. 需要工具时，明确调用对应工具，不要编造结果
3. 工具返回结果后，基于结果生成专业分析报告
4. 禁止编造数据，所有结论必须有工具结果支撑
可用工具：计算器、CSV分析、天气查询、文本摘要、关键词提取
""".strip()


def _get_all_tools(mcp_client: MockMCPClient):
    """合并本地工具与MCP工具"""
    local_tools = list(LOCAL_TOOLS.values())
    mcp_tools = mcp_client.get_langchain_tools()
    return local_tools + mcp_tools


def input_moderation_node(state: AgentState) -> Dict[str, Any]:
    """内容审核节点：校验用户输入合法性"""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        return {"loop_count": state["loop_count"]}

    is_safe, reason = moderate_content(last_msg.content)
    if not is_safe:
        reject_msg = SystemMessage(content=f"请求被拦截：{reason}")
        # 记录拦截原因，供 CLI 退出时判断
        return {
            "messages": [reject_msg], 
            "loop_count": state["loop_count"], 
            "error": "内容审核未通过"
        }

    return {"loop_count": state["loop_count"]}


def agent_node(state: AgentState, llm, mcp_client: MockMCPClient) -> Dict[str, Any]:
    """ReAct推理节点：生成思考与工具调用"""
    # 先截断上下文，控制Token消耗
    trimmed_messages = trim_messages(state["messages"], state.get("max_steps", 12))
    system_prompt = SystemMessage(content=SYSTEM_PROMPT)

    # 绑定所有工具并调用LLM
    tools = _get_all_tools(mcp_client)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_prompt] + trimmed_messages)

    return {
        "messages": [response],
        "loop_count": state["loop_count"] + 1
    }


def tool_execution_node(state: AgentState, mcp_client: MockMCPClient) -> Dict[str, Any]:
    """工具执行节点：统一执行本地工具与MCP工具"""
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    tool_results = []
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        try:
            # 字典分发
            if tool_name in LOCAL_TOOLS:
                result = LOCAL_TOOLS[tool_name].invoke(tool_args)
            # 防止模型幻觉调用不存在的 MCP 工具
            elif hasattr(mcp_client, "has_tool") and not mcp_client.has_tool(tool_name):
                raise ValueError(f"工具 '{tool_name}' 不存在，请从可用列表中选择。")
            else:
                result = mcp_client.call_tool(tool_name, tool_args)

            tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_id))
        except Exception as e:
            # 错误即反馈，把错误信息喂回给大模型，逼它自我纠正
            tool_results.append(ToolMessage(
                content=f"工具执行失败：{str(e)}", 
                tool_call_id=tool_id
            ))

    return {"messages": tool_results}


def loop_check_node(state: AgentState) -> str:
    """循环条件判断：决定继续推理还是输出最终结果"""
    last_msg = state["messages"][-1]
    loop_count = state["loop_count"]
    max_steps = state["max_steps"]

    # 如果审核已经被拦截，直接结束
    if state.get("error") == "内容审核未通过":
        return "final_analysis"

    if loop_count >= max_steps:
        return "final_analysis"

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "human_confirm"

    return "final_analysis"


def final_analysis_node(state: AgentState) -> Dict[str, Any]:
    """最终分析写作节点：整合所有信息输出专业报告"""
    # 如果是审核未通过，直接输出拒绝信息
    if state.get("error") == "内容审核未通过":
        return {"analysis_result": state["messages"][-1].content}

    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls"):
        # 明确报告未完成的原因
        return {
            "messages": [SystemMessage(content="已达到最大推理步数，无法完成分析")], 
            "error": "超过最大步数"
        }

    return {"analysis_result": last_msg.content}
