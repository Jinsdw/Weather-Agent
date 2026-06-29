"""
天气智能体 (Weather Agent)
基于 LangGraph + 智谱 GLM-4.6V-FlashX 的 ReAct 反应式天气查询助手
命令行交互模式
"""

import os
import sys
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, AIMessage, HumanMessage, ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# 1. Prompt 模板
# ============================================================

# 系统提示词模板 (System Prompt Template)
SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["current_time"],
    template="""你是一个专业的天气助手智能体，名叫"小天气"。你的职责是帮助用户查询天气信息并给出专业建议。

## 你的能力
1. 查询指定城市的实时天气、温度、湿度、风力等信息
2. 查询指定城市未来几天的天气预报
3. 根据天气情况给出出行、穿衣、健康等实用建议

## 你的行为准则
1. 始终使用中文回答用户问题
2. 回答要简洁、准确、有条理
3. 如果用户没有指定城市，请主动询问
4. 在提供天气数据后，主动给出穿衣/出行建议
5. 如果工具查询失败，诚实告知用户并建议稍后重试
6. 当前时间是: {current_time}
""",
)

# QA 回答模板 (QA Response Template)
QA_RESPONSE_TEMPLATE = PromptTemplate(
    input_variables=["city", "weather_info", "suggestion"],
    template="""## {city} 天气信息

{weather_info}

## 出行建议
{suggestion}
""",
)

# 天气穿衣建议 Prompt
WEATHER_SUGGESTION_PROMPT = PromptTemplate(
    input_variables=["temperature", "weather", "humidity", "wind"],
    template="""根据以下天气数据，给出简洁的穿衣和出行建议（2-3句话）：
- 温度: {temperature}
- 天气: {weather}
- 湿度: {humidity}
- 风力: {wind}

请直接给出建议，不要重复天气数据。""",
)


# ============================================================
# 2. 工具定义
# ============================================================

@tool
def get_current_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息，包括温度、天气状况、湿度、风力等。

    参数:
        city: 城市名称，如"北京"、"上海"、"广州"等
    """
    # NOTE: 这里使用模拟数据。实际使用时可替换为真实天气API，
    # 如和风天气(https://dev.qweather.com)、心知天气等。
    weather_db = {
        "北京": {"temperature": "32°C", "weather": "晴转多云", "humidity": "45%", "wind": "南风3级"},
        "上海": {"temperature": "29°C", "weather": "多云", "humidity": "72%", "wind": "东南风2级"},
        "广州": {"temperature": "34°C", "weather": "雷阵雨", "humidity": "85%", "wind": "南风2级"},
        "深圳": {"temperature": "33°C", "weather": "多云", "humidity": "78%", "wind": "西南风2级"},
        "成都": {"temperature": "27°C", "weather": "阴", "humidity": "65%", "wind": "微风"},
        "杭州": {"temperature": "30°C", "weather": "晴", "humidity": "55%", "wind": "东风2级"},
        "武汉": {"temperature": "35°C", "weather": "晴", "humidity": "50%", "wind": "南风3级"},
        "南京": {"temperature": "31°C", "weather": "多云转晴", "humidity": "60%", "wind": "东风2级"},
        "重庆": {"temperature": "36°C", "weather": "晴", "humidity": "55%", "wind": "微风"},
        "西安": {"temperature": "33°C", "weather": "晴", "humidity": "35%", "wind": "北风2级"},
        "天津": {"temperature": "31°C", "weather": "多云", "humidity": "50%", "wind": "南风3级"},
        "苏州": {"temperature": "29°C", "weather": "小雨", "humidity": "80%", "wind": "东风1级"},
    }

    # 城市名匹配（支持模糊匹配）
    for key in weather_db:
        if city and key in city or city in key:
            data = weather_db[key]
            return (
                f"城市: {key}\n"
                f"温度: {data['temperature']}\n"
                f"天气: {data['weather']}\n"
                f"湿度: {data['humidity']}\n"
                f"风力: {data['wind']}\n"
                f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

    return f"抱歉，暂未收录城市'{city}'的天气数据。目前支持查询的城市包括: {', '.join(weather_db.keys())}。"


@tool
def get_weather_forecast(city: str, days: int = 3) -> str:
    """
    获取指定城市未来几天的天气预报。

    参数:
        city: 城市名称，如"北京"、"上海"等
        days: 预报天数，默认为3天，最大支持7天
    """
    days = min(max(days, 1), 7)

    # 模拟预报数据
    forecast_db = {
        "北京": [
            {"date": "明天", "weather": "多云", "high": "33°C", "low": "22°C"},
            {"date": "后天", "weather": "小雨", "high": "30°C", "low": "20°C"},
            {"date": "大后天", "weather": "晴", "high": "34°C", "low": "23°C"},
        ],
        "上海": [
            {"date": "明天", "weather": "小雨转阴", "high": "30°C", "low": "24°C"},
            {"date": "后天", "weather": "多云", "high": "31°C", "low": "25°C"},
            {"date": "大后天", "weather": "晴", "high": "32°C", "low": "25°C"},
        ],
        "广州": [
            {"date": "明天", "weather": "雷阵雨", "high": "33°C", "low": "26°C"},
            {"date": "后天", "weather": "多云", "high": "34°C", "low": "27°C"},
            {"date": "大后天", "weather": "晴", "high": "35°C", "low": "27°C"},
        ],
    }

    for key in forecast_db:
        if city and key in city or city in key:
            forecasts = forecast_db[key][:days]
            result_lines = [f"## {key} 未来 {len(forecasts)} 天天气预报\n"]
            for f in forecasts:
                result_lines.append(f"- {f['date']}: {f['weather']}, 最高 {f['high']}, 最低 {f['low']}")
            return "\n".join(result_lines)

    return f"抱歉，暂未收录城市'{city}'的天气预报数据。"


@tool
def get_air_quality(city: str) -> str:
    """
    获取指定城市的空气质量指数(AQI)信息。

    参数:
        city: 城市名称
    """
    aqi_db = {
        "北京": {"aqi": 85, "level": "良", "pm25": "38μg/m³"},
        "上海": {"aqi": 62, "level": "良", "pm25": "25μg/m³"},
        "广州": {"aqi": 55, "level": "良", "pm25": "22μg/m³"},
        "深圳": {"aqi": 48, "level": "优", "pm25": "18μg/m³"},
        "成都": {"aqi": 92, "level": "良", "pm25": "45μg/m³"},
    }

    for key in aqi_db:
        if city and key in city or city in key:
            data = aqi_db[key]
            return (
                f"城市: {key}\n"
                f"AQI: {data['aqi']} ({data['level']})\n"
                f"PM2.5: {data['pm25']}"
            )

    return f"抱歉，暂未收录城市'{city}'的空气质量数据。"


# ============================================================
# 3. Agent 构建与运行
# ============================================================

# 所有工具列表
TOOLS = [get_current_weather, get_weather_forecast, get_air_quality]


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


def _print_tool_call(tool_name: str, tool_args: dict):
    """格式化打印工具调用信息"""
    arg_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
    print(f"\n  🔧 调用工具: {tool_name}({arg_str})")


def _print_thinking(text: str):
    """格式化打印思考过程"""
    # 截断过长的思考内容
    if len(text) > 200:
        text = text[:200] + "..."
    print(f"  💭 思考: {text}")


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

                    # AI 正在调用工具 → 打印工具调用信息
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
# 4. 主入口
# ============================================================

def main():
    """程序主入口"""
    agent = build_agent()
    chat_loop(agent)


if __name__ == "__main__":
    main()
