import os
import sys

from openai import OpenAI


def create_llm():
    # 启动前置校验，拒绝带病运行
    api_key = os.getenv("SILI_API_KEY")
    if not api_key:
        print("❌ 错误：未检测到 SILI_API_KEY，请先配置环境变量。", file=sys.stderr)
        sys.exit(1)

    # 从环境变量读取配置，不要硬编码
    base_url = os.getenv("SILI_BASE_URL", "https://api.siliconflow.cn/v1")
    model_name = os.getenv("LLM_MODEL", "Pro/zai-org/GLM-4.7")

    # 加上 timeout，防止 CLI 卡死
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

    # 注意：create_llm 只负责返回配置好的 client，不要在工具函数里直接发起请求
    return client, model_name

if __name__ == "__main__":
    # 测试连通性
    client, model = create_llm()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好"}]
    )
    print(response.choices[0].message.content)
