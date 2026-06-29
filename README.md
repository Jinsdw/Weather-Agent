# Weather Agent

基于 **LangGraph** + **智谱 GLM-4.6V-FlashX** 的 ReAct 反应式天气查询智能体，命令行交互模式运行。
<img width="906" height="892" alt="QQ20260629-162851" src="https://github.com/user-attachments/assets/990c51b0-0ed5-4f9c-8121-a686c1296dcb" />

## 功能特性

- **ReAct 推理架构**：思考 -> 调用工具 -> 观察结果 -> 生成回答，完整反应式循环
- **工具调用可视化**：实时显示 Agent 调用了什么工具、传入参数、返回结果
- **逐 Token 流式输出**：AI 回复以打字机效果逐字输出
- **对话记忆**：基于 LangGraph MemorySaver，支持多轮上下文记忆
- **Prompt 模板化**：系统提示词、QA 回答模板、建议生成模板均独立定义

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangGraph 1.2.6 |
| LLM 接入 | LangChain OpenAI (ChatOpenAI) |
| 大模型 | 智谱 GLM-4.6V-FlashX |
| API 协议 | OpenAI 兼容接口 |
| 运行环境 | Python >= 3.10 |

## 项目结构

```
weather-Agent/
├── main.py       # 主程序入口：Agent构建 + 命令行交互

├── prompts.py   # Prompt模板定义：系统提示词、QA模板、建议模板
├── tools.py     # Agent工具定义：天气查询、天气预报、空气质量
└── README.md    # 项目说明
```

### 代码模块说明

**`prompts.py`** — Prompt 模板定义
- `SYSTEM_PROMPT_TEMPLATE`：系统提示词，定义智能体角色与行为准则
- `QA_RESPONSE_TEMPLATE`：QA 回答格式化模板
- `WEATHER_SUGGESTION_PROMPT`：穿衣/出行建议生成模板

**`tools.py`** — Agent 工具定义
- `get_current_weather(city)`：查询实时天气（温度/天气/湿度/风力）
- `get_weather_forecast(city, days)`：查询未来天气预报（1-7天）
- `get_air_quality(city)`：查询空气质量 AQI/PM2.5
- `TOOLS`：所有工具的统一导出列表

**`main.py`** — 主程序入口
- `build_agent()`：初始化智谱大模型，创建 ReAct 智能体
- `chat_loop(agent)`：命令行交互主循环，带工具调用可视化 + 逐 Token 流式输出

## 快速开始

### 1. 安装依赖

```bash
pip install langchain-openai langgraph langchain-core httpx_sse
```

### 2. 配置 API Key

前往 [智谱开放平台](https://open.bigmodel.cn/) 注册账号并获取 API Key。

```bash
# Windows PowerShell
$env:ZHIPUAI_API_KEY = "your_api_key_here"

# Linux / macOS
export ZHIPUAI_API_KEY="your_api_key_here"
```

### 3. 运行

```bash
python main.py
```

## 使用示例

```
==================================================
  天气智能体 (Weather Agent)
  基于 LangGraph + 智谱 GLM-4.6V-FlashX
==================================================

可用功能:
  - 查询城市实时天气 (如: 北京今天天气怎么样)
  - 查询天气预报     (如: 上海未来三天天气)
  - 查询空气质量     (如: 广州空气质量如何)
  - 日常闲聊与建议   (如: 35度穿什么)

输入 'quit' 或 'exit' 退出
--------------------------------------------------

你: 北京今天天气怎么样？

  🔧 调用工具: get_current_weather  参数: {"city": "北京"}
  ✅ 工具返回: 城市: 北京\n温度: 32°C\n天气: 晴转多云\n湿度: 45%\n风力: 南风3级...

小天气: 北京今天的天气是晴转多云，气温32°C，湿度45%，南风3级。天气较热，建议穿着轻薄透气的衣物，外出注意防晒...

你: 上海未来三天天气如何？

  🔧 调用工具: get_weather_forecast  参数: {"city": "上海", "days": 3}
  ✅ 工具返回: ## 上海 未来 3 天天气预报\n- 明天: 小雨转阴...

小天气: 上海未来三天的天气预报如下...

你: quit
再见!
```

## 命令行指令

| 指令 | 说明 |
|------|------|
| `help` / `帮助` | 显示可用功能列表 |
| `clear` / `清除` | 清除对话记忆，开始新对话 |
| `quit` / `exit` / `q` | 退出程序 |

## 扩展指南

### 接入真实天气 API

当前工具使用模拟数据。替换为真实 API 的步骤：

1. 注册天气 API 服务（如 [和风天气](https://dev.qweather.com/)）
2. 在工具函数中替换模拟数据为 HTTP 请求：

```python
import requests

@tool
def get_current_weather(city: str) -> str:
    """获取指定城市的实时天气信息"""
    API_KEY = "your_qweather_key"
    url = f"https://devapi.qweather.com/v7/weather/now?location={city}&key={API_KEY}"
    resp = requests.get(url).json()
    # 解析并返回天气数据...
```

### 添加新工具

使用 `@tool` 装饰器定义新工具，加入 `TOOLS` 列表即可：

```python
@tool
def get_uv_index(city: str) -> str:
    """获取指定城市的紫外线指数"""
    # 实现逻辑...
    return f"{city}当前紫外线指数: 6 (中等)"

TOOLS = [get_current_weather, get_weather_forecast, get_air_quality, get_uv_index]
```

## 注意事项

- 模型名称 `glm-4.6v-flashx` 需在智谱开放平台确认可用，如不可用可替换为 `glm-4-flash` 等其他模型
- API Key 请勿硬编码在代码中，务必通过环境变量传入
- 当前天气数据为模拟数据，仅供演示 Agent 交互流程
"# Weather-Agent" 
