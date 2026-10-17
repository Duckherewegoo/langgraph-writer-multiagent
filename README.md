LangGraph Writer MultiAgent

一个基于 LangGraph 构建的轻量级、工程化 CLI 智能体。支持 ReAct 推理循环、MCP 协议工具调用、人工介入确认与长上下文自动压缩。
核心能力

    开箱即用的 ReAct 循环：模型自主思考、调用工具，并支持最大步数限制防死循环。

    原生 Human-in-the-Loop (HITL)：在执行危险工具前触发原生中断，等待终端用户确认。

    无缝接入 MCP 生态：内置 MCP 客户端与 Mock 服务端，支持直接拉起外部 real_server.py 进程。

    丰富的本地工具：内置安全计算器、CSV 数据分析、天气查询。

    长上下文防线：内置基于 Token 估算的截断与 LLM 摘要记忆机制，防爆窗。

    前置内容审核：可扩展的敏感词拦截（支持 AC 自动机），在入口处阻断违规请求。

    兼容 OpenAI 协议：完美适配硅基流动等免费推理 API。

快速启动

    安装依赖：
    pip install -e .

    配置环境变量：
    复制 .env.example 为 .env，并填入你的 API 密钥：

LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Pro/zai-org/GLM-4.7

    运行：

单次命令模式：
python src/main.py "分析一下 data.csv，并告诉我明天的天气"

交互式终端：
python src/main.py
项目结构

    core/：LangGraph 图定义（State, Nodes, Graph 路由）

    mcp/：MCP 协议实现（客户端、Mock 服务端、真实 stdio 服务端）

    tools/：本地工具集（计算器、CSV分析、天气）

    utils/：横切逻辑（敏感词审核、上下文截断/摘要）

License

GPLv3
