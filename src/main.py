import argparse
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from core.graph import build_agent_graph
from mcp.client import RealMCPClient

# 加载环境变量必须放在最前面，防止内部模块导入时读不到
load_dotenv()

def init_dependencies():
    """初始化并校验依赖"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    
    if not api_key or not model:
        print("❌ 错误：未配置 LLM_API_KEY 或 LLM_MODEL，请检查 .env 文件。", file=sys.stderr)
        sys.exit(2)

    try:
        max_steps = int(os.getenv("MAX_LOOP_STEPS", "6"))
    except ValueError:
        max_steps = 6
        print("⚠️ 警告：MAX_LOOP_STEPS 格式错误，已回退默认值 6。", file=sys.stderr)

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
        timeout=30.0,
        max_retries=2
    )

    mcp_client = RealMCPClient(llm=llm)  # 注意：按我们之前改的，可能需要传 llm
    mcp_client.connect()
    
    agent_graph = build_agent_graph(llm, mcp_client, max_steps=max_steps)
    return agent_graph, max_steps


def run_task(agent_graph, max_steps, user_input: str):
    """执行单次任务"""
    # 每次任务独立 thread_id，保证记忆隔离
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4().hex[:8]}"}}
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "loop_count": 0,
        "max_steps": max_steps,
        "human_feedback": None,
        "analysis_result": None,
        "csv_file_path": None,
        "error": None
    }

    try:
        print("🤖 思考中...")
        result = agent_graph.invoke(initial_state, config=config)

        # 处理人机确认中断
        while result.get("__interrupt__"):
            print("\n⚠️ 智能体请求调用工具：")
            # 这里可以精细化打印工具详情，但为了极简，直接问
            try:
                choice = input("是否允许执行？(y/n): ").strip().lower()
            except EOFError:
                choice = "n"
                
            if choice in ('y', 'yes', ''):
                print("⏳ 执行工具...")
                result = agent_graph.invoke(None, config=config)
            else:
                print("❌ 已取消工具执行。", file=sys.stderr)
                sys.exit(1)

        if result.get("error"):
            print(f"\n❌ 任务失败: {result['error']}", file=sys.stderr)
            sys.exit(1)
            
        print("\n✅ 结果：")
        print(result.get("analysis_result", "无结果"))
        
    except KeyboardInterrupt:
        print("\n👋 操作已取消。", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 系统错误: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="LangGraph ReAct 写作智能体 CLI")
    parser.add_argument("query", nargs="?", help="直接传入要执行的指令（不传则进入交互模式）")
    args = parser.parse_args()

    # 初始化
    agent_graph, max_steps = init_dependencies()

    if args.query:
        # 模式一：单次执行 python main.py "帮我算一下"
        run_task(agent_graph, max_steps, args.query)
    else:
        # 模式二：交互式对话 python main.py
        print("欢迎使用智能体 CLI。输入 'exit' 或 'quit' 退出。")
        while True:
            try:
                query = input("\n> ").strip()
                if query.lower() in ("exit", "quit"):
                    break
                if not query:
                    continue
                run_task(agent_graph, max_steps, query)
                print("-" * 40)
            except KeyboardInterrupt:
                print("\n👋 再见。")
                sys.exit(130)
            except EOFError:
                break


if __name__ == "__main__":
    main()
