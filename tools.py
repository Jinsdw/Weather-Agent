"""
Agent 工具定义
包含实时天气查询、天气预报查询、空气质量查询三个工具

NOTE: 当前使用模拟数据，实际使用时可替换为真实天气 API，
如和风天气(https://dev.qweather.com)、心知天气等。
"""

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息，包括温度、天气状况、湿度、风力等。

    参数:
        city: 城市名称，如"北京"、"上海"、"广州"等
    """
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


# 所有工具列表
TOOLS = [get_current_weather, get_weather_forecast, get_air_quality]
