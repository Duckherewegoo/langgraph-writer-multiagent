# LangGraph Writer MultiAgent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-orange.svg)](https://langchain-ai.github.io/langgraph/)

一个基于 [LangGraph](https://langchain-ai.github.io/langgraph/) 构建的轻量级、工程化 CLI 智能体。支持 ReAct 推理循环、MCP 协议工具调用、人工介入确认与长上下文自动压缩。

______________________________________________________________________

## 核心能力

- **开箱即用的 ReAct 循环** — 模型自主思考、调用工具，并支持最大步数限制防止死循环。
- **原生 Human-in-the-Loop (HITL)** — 在执行危险工具前触发原生中断，等待终端用户确认。
- **无缝接入 MCP 生态** — 内置 MCP 客户端与 Mock 服务端，支持直接拉起外部 `real_server.py` 进程。
- **丰富的本地工具** — 内置安全计算器、CSV 数据分析、天气查询。
- **长上下文防线** — 内置基于 Token 估算的截断与 LLM 摘要记忆机制，防止上下文溢出。
- **前置内容审核** — 可扩展的敏感词拦截（支持 AC 自动机），在入口处阻断违规请求。
- **兼容 OpenAI 协议** — 适配硅基流动（SiliconFlow）等兼容 OpenAI 接口的推理 API。

## 快速启动

### 环境要求

- Python 3.11+
- 可用的 LLM API 密钥（OpenAI 兼容接口）

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的 API 密钥：

```bash
cp .env.example .env
```

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Pro/zai-org/GLM-4.7
```

### 3. 运行

单次命令模式：

```bash
python src/main.py "分析一下 data.csv，并告诉我明天的天气"
```

交互式终端：

```bash
python src/main.py
```

## 项目结构

```text
langgraph-writer-multiagent/
├── src/
│   ├── main.py              # CLI 入口
│   ├── core/                # LangGraph 图定义（State, Nodes, Graph 路由）
│   ├── mcp/                 # MCP 协议实现（客户端、Mock 服务端、真实 stdio 服务端）
│   ├── tools/               # 本地工具集（计算器、CSV 分析、天气）
│   └── utils/               # 横切逻辑（敏感词审核、上下文截断/摘要）
├── .env.example             # 环境变量模板
└── pyproject.toml           # 依赖与打包配置
```

| 目录 | 职责 |
| --- | --- |
| `core/` | LangGraph 图定义：状态契约、节点函数、条件路由 |
| `mcp/` | MCP 协议实现：客户端、Mock 服务端、真实 stdio 服务端 |
| `tools/` | 本地工具集：安全计算器、CSV 数据分析、天气查询 |
| `utils/` | 横切逻辑：敏感词审核、上下文截断与摘要 |

## 配置说明

| 变量 | 说明 | 是否必填 |
| --- | --- | --- |
| `LLM_API_KEY` | LLM 服务商 API 密钥 | 必填 |
| `LLM_BASE_URL` | 兼容 OpenAI 的接口地址 | 必填 |
| `LLM_MODEL` | 模型标识符 | 必填 |

完整字段见 `.env.example`。

## 使用示例

```text
> 分析一下 data.csv，并告诉我明天的天气
[思考] 需要调用两个工具：读取 CSV 与查询天气
[工具] csv_analyze(path="data.csv")     → 返回 1,240 行，3 列
[工具] weather(city="Beijing", date="tomorrow") → 晴，18°C
[摘要] data.csv 含 1,240 条销售记录；明日北京晴，18°C
```

HITL 中断示例：

```text
[中断] 即将执行高危操作：rm -rf ./build
允许 / 拒绝 / 查看详情？
```

## 故障排查

**`ModuleNotFoundError: No module named 'src'`**

原因：未在项目根目录执行，或包未以可编辑模式安装。

```bash
pip install -e .
python src/main.py
```

**`AuthenticationError` / 401**

原因：`LLM_API_KEY` 为空或失效。

```bash
grep LLM_API_KEY .env
```

**上下文溢出 / `ContextLengthExceeded`**

原因：对话轮次过多触发窗口上限。项目内置截断与摘要机制，若仍报错，可减小单次输入或降低 `MAX_TOKENS`。

**MCP 服务端启动失败**

原因：`real_server.py` 路径未加入 `PATH`，或依赖缺失。

```bash
which real_server.py
pip install -e ".[mcp]"
```

## License

本项目以 GNU General Public License v3.0 发布。

- 源码仓库根目录应包含 [`LICENSE`](LICENSE) 文件；若缺失，可从 [GNU 官方文本](https://www.gnu.org/licenses/gpl-3.0.txt) 获取。
- 分发时须保留版权声明与许可声明，修改版本须标注变更。
- 本软件按"原样"提供，无任何明示或暗示担保，详见许可证全文。

## 相关链接

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol 规范](https://modelcontextprotocol.io/)
- [硅基流动 SiliconFlow](https://siliconflow.cn/)

______________________________________________________________________

Made with ❤️ by DakeGinger
