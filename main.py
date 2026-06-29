"""
天气智能体 (Weather Agent)
基于 LangGraph + 智谱 GLM-4.6V-FlashX 的 ReAct 反应式天气查询助手
命令行交互模式
"""

import os
import sys
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 导入拆分的模块
from prompts import SYSTEM_PROMPT_TEMPLATE
from tools import TOOLS


# ============================================================
# 1. Agent 构建
# ============================================================

def build_agent() -> object:
    """构建并返回 LangGraph ReAct Agent"""

    # 从环境变量读取智谱API Key
    api_key = os.environ.get("ZHIPUAI_API_KEY", "")
    if not api_key:
        print("错误: 请设置环境变量 ZHIPUAI_API_KEY")
        print("  Windows PowerShell: $env:ZHIPUAI_API_KEY='your_api_key_here'")
        print("  Linux/Mac: export ZHIPUAI_API_KEY='your_api_key_here'")
        sys.exit(1)

    # 初始化智谱大模型 (通过 OpenAI 兼容接口)
    llm = ChatOpenAI(
        model="glm-4.6v-flashx",  # 智谱 GLM-4.6V-FlashX 模型
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.7,
        max_tokens=2048,
    )

    # 生成带当前时间的系统提示词
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_time=current_time)

    # 创建 ReAct Agent
    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=system_prompt,
        checkpointer=MemorySaver(),  # 启用对话记忆
    )

    return agent


# ============================================================
# 2. 命令行交互
# ============================================================

def print_welcome():
    """打印欢迎信息"""
    print("=" * 50)
    print("  天气智能体 (Weather Agent)")
    print("  基于 LangGraph + 智谱 GLM-4.6V-FlashX")
    print("=" * 50)
    print()
    print("可用功能:")
    print("  - 查询城市实时天气 (如: 北京今天天气怎么样)")
    print("  - 查询天气预报     (如: 上海未来三天天气)")
    print("  - 查询空气质量     (如: 广州空气质量如何)")
    print("  - 日常闲聊与建议   (如: 35度穿什么)")
    print()
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 50)


def chat_loop(agent):
    """命令行交互主循环（带工具调用展示 + 逐token流式输出）"""

    # 对话线程配置 (用于记忆)
    config = {"configurable": {"thread_id": "weather-chat-session-001"}}

    print_welcome()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        if user_input.lower() in ("help", "帮助"):
            print("可用功能:")
            print("  - 查询城市实时天气 (如: 北京今天天气怎么样)")
            print("  - 查询天气预报     (如: 上海未来三天天气)")
            print("  - 查询空气质量     (如: 广州空气质量如何)")
            print("  - 输入 'clear' 清除对话记忆")
            print("  - 输入 'quit' 退出")
            continue

        if user_input.lower() in ("clear", "清除"):
            config = {"configurable": {"thread_id": f"weather-chat-session-{datetime.now().timestamp()}"}}
            print("对话记忆已清除，开始新对话。")
            continue

        # ====== 调用 Agent（stream_mode="messages" 实现逐 token 流式输出） ======
        try:
            inputs = {"messages": [("user", user_input)]}
            print()  # 空行

            # 记录上一条消息类型，用于区分不同阶段
            prev_type = None

            # LangGraph >=1.2 stream_mode="messages" 返回 (message_chunk, metadata) 元组
            for msg_chunk, metadata in agent.stream(inputs, config, stream_mode="messages"):
                # ---- 处理 AI 消息块（可能包含 tool_calls 或 content） ----
                if isinstance(msg_chunk, AIMessage):

                    # AI 正在调用工具 -> 打印工具调用信息
                    if msg_chunk.tool_call_chunks:
                        for tc in msg_chunk.tool_call_chunks:
                            if tc.get("name"):
                                # 新工具调用开始
                                print(f"\n  🔧 调用工具: {tc['name']}", end="", flush=True)
                            if tc.get("args"):
                                # 打印工具参数（增量拼接）
                                print(f"  参数: {tc['args']}", end="", flush=True)
                        prev_type = "tool_call"
                        continue

                    # AI 正在流式输出文本内容
                    if msg_chunk.content:
                        if prev_type != "content":
                            print("\n\n小天气: ", end="", flush=True)
                        print(msg_chunk.content, end="", flush=True)
                        prev_type = "content"
                        continue

                # ---- 处理工具返回结果 ----
                if isinstance(msg_chunk, ToolMessage):
                    print(f"\n  ✅ 工具返回: {msg_chunk.content[:100]}{'...' if len(msg_chunk.content) > 100 else ''}")
                    prev_type = "tool_result"
                    continue

            # 流式结束后换行
            if prev_type == "content":
                print()
            else:
                print("\n小天气: （无文本回复）")

        except Exception as e:
            print(f"\n小天气: 抱歉，发生了错误: {e}")
            print(f"  错误类型: {type(e).__name__}")

        print()  # 空行分隔对话


# ============================================================
# 3. 主入口
# ============================================================

def main():
    """程序主入口"""
    agent = build_agent()
    chat_loop(agent)


if __name__ == "__main__":
    main()
