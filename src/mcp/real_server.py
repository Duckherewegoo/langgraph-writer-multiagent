import sys
import jieba
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务端，命名为 "TaskPlannerTools"
mcp = FastMCP("TaskPlannerTools")


@mcp.tool()
def mcp_summarize(text: str) -> str:
    """
    对长文本进行摘要提炼。
    """
    if not text or not text.strip():
        return "摘要失败：文本为空"
    
    # 在真实的独立进程中，如果没有传大模型的依赖，
    # 这里我们用一个严谨的启发式截断（取前 100 字）来保底。
    # 未来如果需要 LLM 摘要，可以在这里异步调用 OpenAI API。
    cleaned_text = text.strip().replace('\n', ' ')
    summary = cleaned_text[:100]
    
    suffix = "..." if len(cleaned_text) > 100 else ""
    return f"摘要：{summary}{suffix}（共 {len(cleaned_text)} 字）"


@mcp.tool()
def mcp_keyword_extract(text: str) -> str:
    """
    提取文本中的核心关键词。
    """
    if not text or not text.strip():
        return "提取失败：文本为空"
        
    try:
        # jieba.cut 返回生成器，必须转 list 才能切片
        words = list(jieba.cut(text, cut_all=True))
        # 过滤掉单字和纯空白字符，保留真正的词
        valid_words = [w for w in words if len(w.strip()) > 1]
        
        # 取前 5 个关键词
        keywords = valid_words[:5]
        
        if not keywords:
            return "提取关键词：未找到明显关键词"
            
        return f"提取关键词：{', '.join(keywords)}"
    except Exception as e:
        return f"关键词提取异常: {str(e)}"


if __name__ == "__main__":
    # 默认以 stdio 模式运行，完美适配 RealMCPClient 的 stdio_client
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\nMCP 服务端已优雅退出。", file=sys.stderr)
        sys.exit(0)
