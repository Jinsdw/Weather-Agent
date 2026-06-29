"""
Prompt 模板定义
包含系统提示词、QA 回答模板、穿衣建议模板
"""

from langchain_core.prompts import PromptTemplate


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
