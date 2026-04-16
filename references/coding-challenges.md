# AI 应用开发实操题集

> 面试最后环节可选的实操题。难度从简单到中等，覆盖 Prompt 工程、Agent 流程设计、RAG 实现、工具调用等方向。
>
> 面试官可根据候选人水平和剩余时间灵活选题，每题标注了难度和预估用时。

---

## 目录

- [1. Prompt 工程](#1-prompt-工程)
  - [1.1 设计 System Prompt](#11-设计-system-prompt)
  - [1.2 Prompt 改写优化](#12-prompt-改写优化)
- [2. Agent 流程设计](#2-agent-流程设计)
  - [2.1 ReAct 循环实现](#21-react-循环实现)
  - [2.2 Agent 决策逻辑](#22-agent-决策逻辑)
- [3. RAG 实战](#3-rag-实战)
  - [3.1 分块策略实现](#31-分块策略实现)
  - [3.2 混合检索实现](#32-混合检索实现)
  - [3.3 重排序实现](#33-重排序实现)
- [4. 工具调用与 MCP](#4-工具调用与-mcp)
  - [4.1 工具调用封装](#41-工具调用封装)
  - [4.2 MCP Server 实现](#42-mcp-server-实现)
- [5. 实用工具函数](#5-实用工具函数)
  - [5.1 Token 数量估算](#51-token-数量估算)
  - [5.2 上下文窗口压缩](#52-上下文窗口压缩)

---

## 1. Prompt 工程

### 1.1 设计 System Prompt

**难度**：简单 | **预估用时**：5-8 分钟 | **高频指数**：⭐⭐⭐⭐⭐

**题目**：为客服 Agent 设计一个 System Prompt，满足以下要求：

1. 角色定位：专业电商客服
2. 必须遵循的行为规范：仅回答商品相关问题，不知识时引导至人工
3. 输出格式：先确认用户问题，再给出解答
4. 包含一个思维链示例

**参考解答**：

```python
SYSTEM_PROMPT = """你是一位专业的电商平台客服助手，名为"小二"。

【角色定位】
- 专注于回答商品咨询、订单查询、物流跟踪、退换货等问题
- 语气亲切、专业、有耐心

【行为规范】
- 只回答与本平台商品和服务相关的问题
- 遇到以下情况时，礼貌引导至人工客服：
  * 涉及投诉、赔偿协商
  * 需要查询用户账户敏感信息
  * 问题超出商品咨询范围
- 不知道的问题不要瞎编，直接说"这个我不太清楚"

【输出格式】
请按以下格式回答：
1. 先确认用户问题："您好，您是想了解[问题摘要]对吗？"
2. 再给出解答或引导

【思维链示例】
用户问：这件衣服是纯棉的吗？
思考：这是商品属性问题，属于客服可回答范围
回答：您好，您是想了解这件衣服的材质对吗？这件商品详情页标注面料为95%纯棉+5%氨纶..." 
```

**考察点**：Prompt 结构化设计、角色设定、边界情况处理、输出格式控制

**追问方向**：
- 如何防止 Prompt Injection？（输入中加入"忽略上面的指令"等）
- 如何让模型在不知道时承认而不是 hallucinate？
- 如何设计 Few-shot Examples？

---

### 1.2 Prompt 改写优化

**难度**：中等 | **预估用时**：8-10 分钟 | **高频指数**：⭐⭐⭐⭐

**题目**：优化以下 Prompt，使其更清晰、更能避免幻觉、更容易被人可控：

```
原始 Prompt：
"帮我写一个函数，检查邮箱格式是否正确，然后把结果返回给我"
```

**参考解答**：

```python
OPTIMIZED_PROMPT = """你是一个 Python 编程助手。请根据用户需求编写函数。

【任务】
编写一个邮箱格式校验函数 check_email(email: str) -> bool

【具体要求】
1. 标准：符合 RFC 5322 规范（简化版）
2. 必须包含 @ 符号，@ 前后有字符
3. @ 后必须有点号 .，且 . 不在开头或结尾
4. 不能包含空格

【输出要求】
1. 只输出代码，不解释
2. 代码用 def check_email(email: str) -> bool: 开头
3. 附简要注释说明关键判断逻辑

【边界情况】
- 空字符串返回 False
- 单字符域名返回 False（如 a@b.c）
- IP 格式域名需要特殊处理（可选）

【示例】
check_email("user@example.com") -> True
check_email("invalid") -> False
check_email("@example.com") -> False
```

**考察点**：Prompt 细节打磨、边界情况覆盖、输出格式控制

---

## 2. Agent 流程设计

### 2.1 ReAct 循环实现

**难度**：中等 | **预估用时**：10-15 分钟 | **高频指数**：⭐⭐⭐⭐⭐

**题目**：实现一个简单的 ReAct Agent，包含以下组件：

```python
import anthropic
from typing import Literal

client = anthropic.Anthropic()

# 工具定义
tools = [
    {
        "name": "search",
        "description": "搜索网络获取信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "执行数学计算",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }
    }
]

def react_agent(user_query: str, max_iterations: int = 5) -> str:
    """
    实现 ReAct 循环：
    1. Thought：思考下一步行动
    2. Action：选择并执行工具
    3. Observation：观察结果
    4. 如果达到 max_iterations 或得到最终答案，返回结果
    """
    # TODO: 实现代码
    
    pass
```

**参考解答**：

```python
def react_agent(user_query: str, max_iterations: int = 5) -> str:
    messages = [
        {"role": "user", "content": f"""你是一个 AI 助手，可以调用工具来回答问题。

用户问题：{user_query}

你必须按以下格式回复：
```
Thought: 你现在需要思考做什么
Action: {{"name": "工具名", "input": {{"参数": "值"}}}}
Observation: [工具返回结果]
```

当问题已解决时，回复：
```
Thought: 我已经得到答案
Final Answer: [你的最终回答]
```
"""}
    ]
    
    for i in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # 解析响应中的工具调用
        for content in response.content:
            if content.type == "text":
                text = content.text
                print(f"[Iteration {i+1}] {text}")
                
                # 检查是否是最终答案
                if "Final Answer:" in text:
                    final_answer = text.split("Final Answer:")[1].strip()
                    return final_answer
                    
                messages.append({"role": "assistant", "content": text})
                
            elif content.type == "tool_use":
                tool_name = content.name
                tool_input = content.input
                
                # 执行工具（模拟）
                if tool_name == "search":
                    result = f"搜索结果：关于'{tool_input['query']}'的信息..."
                elif tool_name == "calculate":
                    result = f"计算结果：{eval(tool_input['expression'])}"
                else:
                    result = "未知工具"
                
                observation = f"Observation: {result}"
                print(f"  -> {observation}")
                messages.append({"role": "user", "content": observation})
    
    return "达到最大迭代次数，未能得出答案"
```

**考察点**：ReAct 范式理解、工具调用格式、循环终止条件、状态管理

**追问方向**：
- 如何避免无限循环？（最大迭代次数、重复检测）
- 如何处理工具执行失败？
- 如何让 Agent 自主选择工具而不是按照固定顺序？

---

### 2.2 Agent 决策逻辑

**难度**：中等 | **预估用时**：8-10 分钟 | **高频指数**：⭐⭐⭐

**题目**：设计一个路由 Agent，根据用户问题类型选择不同的处理策略：

```python
def route_intent(user_message: str) -> str:
    """
    根据用户消息判断意图类型
    返回值：
    - "order": 订单相关问题
    - "product": 商品咨询
    - "refund": 退换货
    - "human": 需要转人工
    - "other": 其他
    """
    # TODO: 实现代码
    pass
```

**参考解答**：

```python
import anthropic

client = anthropic.Anthropic()

def route_intent(user_message: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{
            "role": "user", 
            "content": f"""分析以下用户消息，判断其意图类型。

用户消息：{user_message}

意图类型定义：
- "order": 订单查询、物流跟踪、修改订单
- "product": 商品属性、功能、价格咨询
- "refund": 退换货申请、退款进度
- "human": 投诉、赔偿、账户问题等需要人工处理
- "other": 不属于以上类型的一般问题

请只输出一个词：order / product / refund / human / other
"""
        }]
    )
    
    intent = response.content[0].text.strip().lower()
    
    # 验证返回值的有效性
    valid_intents = {"order", "product", "refund", "human", "other"}
    if intent not in valid_intents:
        return "other"
    
    return intent
```

**考察点**：LLM 工具调用、意图识别、路由设计

**追问方向**：
- 如何处理多意图混合的情况？
- 如何支持自定义意图类型的扩展？
- 如何在分类不确定时做 fallback？

---

## 3. RAG 实战

### 3.1 分块策略实现

**难度**：简单 | **预估用时**：5-8 分钟 | **高频指数**：⭐⭐⭐⭐

**题目**：实现一个简单的滑动窗口分块器

```python
def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    将文本分割成块
    
    Args:
        text: 待分割文本
        chunk_size: 每块字符数
        overlap: 相邻块重叠字符数
    
    Returns:
        文本块列表
    """
    # TODO: 实现代码
    pass
```

**参考解答**：

```python
def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 滑动窗口：下次开始位置 = 结束位置 - 重叠长度
        start = end - overlap
        
        # 防止死循环
        if overlap >= chunk_size:
            break
    
    return chunks

# 测试
text = "这是一个测试文本，用于演示分块功能。" * 20
chunks = chunk_text(text, chunk_size=50, overlap=10)
for i, c in enumerate(chunks):
    print(f"Chunk {i}: {c}")
```

**考察点**：滑动窗口算法、边界情况处理、重叠机制理解

**追问方向**：
- 如何按句子/段落分割而不是字符？（更智能的分块）
- 如何避免在句子中间断开？（语义完整性）
- chunk_size 如何选择？（根据 LLM context window 和语义密度）

---

### 3.2 混合检索实现

**难度**：中等 | **预估用时**：10-15 分钟 | **高频指数**：⭐⭐⭐⭐⭐

**题目**：实现一个混合检索器，结合向量检索和关键词检索

```python
from typing import List, Tuple

class HybridRetriever:
    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store  # 向量数据库
        self.bm25_index = bm25_index      # BM25 索引
    
    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[float, str]]:
        """
        混合检索
        
        Args:
            query: 查询文本
            k: 返回结果数
            alpha: 向量权重 (0-1)，1-alpha 为 BM25 权重
        
        Returns:
            [(score, text), ...] 按得分降序
        """
        # TODO: 实现代码
        pass
```

**参考解答**：

```python
class HybridRetriever:
    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
    
    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[float, str]]:
        # 1. 向量检索
        vector_results = self.vector_store.similarity_search_with_score(query, k=k)
        vector_scores = {text: score for score, text in vector_results}
        
        # 2. BM25 检索
        bm25_results = self.bm25_index.search(query, k=k)
        bm25_scores = {text: score for text, score in bm25_results}
        
        # 3. 获取所有候选文档
        all_docs = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # 4. 归一化分数并融合
        results = []
        for doc in all_docs:
            # 向量分数归一化（假设是 0-1 的相似度）
            vec_score = vector_scores.get(doc, 0)
            
            # BM25 分数归一化（取相对排名）
            bm25_score = bm25_scores.get(doc, 0)
            
            # RRF (Reciprocal Rank Fusion) 方法也可以
            combined_score = alpha * vec_score + (1 - alpha) * bm25_score
            results.append((combined_score, doc))
        
        # 5. 排序返回
        results.sort(key=lambda x: x[0], reverse=True)
        
        return results[:k]
```

**考察点**：混合检索原理、分数归一化、RRF 融合算法

**追问方向**：
- alpha 参数如何选择？（根据召回率调整）
- RRF 方法相比加权平均有什么优势？（对异常值更鲁棒）
- 如何处理向量检索和 BM25 检索结果差异大的情况？

---

### 3.3 重排序实现

**难度**：中等 | **预估用时**：8-10 分钟 | **高频指数**：⭐⭐⭐⭐

**题目**：实现一个简单的 Cross-Encoder 重排序

```python
from sentence_transformers import CrossEncoder

def rerank(query: str, documents: List[str], top_k: int = 3) -> List[str]:
    """
    使用 Cross-Encoder 对文档重排序
    
    Args:
        query: 查询文本
        documents: 候选文档列表
        top_k: 返回前 k 个
    
    Returns:
        重排序后的文档列表
    """
    # TODO: 实现代码
    pass
```

**参考解答**：

```python
def rerank(query: str, documents: List[str], top_k: int = 3) -> List[str]:
    if not documents:
        return []
    
    # 初始化 Cross-Encoder（使用预训练模型）
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # 构造 query-document 对
    pairs = [(query, doc) for doc in documents]
    
    # 获取相关性分数
    scores = model.predict(pairs)
    
    # 按分数降序排列
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 返回 top_k
    return [doc for doc, _ in doc_scores[:top_k]]

# 使用示例
query = "如何提高 RAG 的召回率"
documents = [
    "RAG 系统可以通过优化嵌入模型来提高召回率",
    "使用混合检索结合向量和关键词检索",
    "Python 是一种编程语言",
    "上下文窗口压缩技术可以处理长文本"
]

reranked = rerank(query, documents, top_k=2)
print(reranked)
```

**考察点**：Cross-Encoder 原理、重排序流程、语义匹配

**追问方向**：
- Bi-Encoder vs Cross-Encoder 的区别？（延迟、精度权衡）
- 重排序的典型流程是怎样的？（先召回再排序两阶段）
- 如何训练自定义的 Cross-Encoder？

---

## 4. 工具调用与 MCP

### 4.1 工具调用封装

**难度**：简单 | **预估用时**：5-8 分钟 | **高频指数**：⭐⭐⭐⭐

**题目**：封装一个通用的工具调用装饰器

```python
import anthropic
from functools import wraps
from typing import Callable, Any

client = anthropic.Anthropic()

def tool_call(description: str):
    """
    工具调用装饰器，用于定义可以被 LLM 调用的工具
    
    用法：
        @tool_call(description="获取当前天气")
        def get_weather(location: str) -> str:
            '''获取指定位置的天气'''
            # 实际实现
            return f"{location} 天气晴朗，25°C"
    """
    def decorator(func: Callable) -> Callable:
        # TODO: 实现装饰器逻辑
        pass
    return decorator

# 定义工具列表供 LLM 使用
tools = []

@tool_call(description="获取当前天气")
def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    return f"{location} 天气晴朗，25°C"
```

**参考解答**：

```python
import anthropic
from functools import wraps
from typing import Callable, Any, List, Dict

client = anthropic.Anthropic()

# 全局工具注册表
_tool_registry: Dict[str, Callable] = {}

def tool_call(description: str):
    """
    工具调用装饰器，用于定义可以被 LLM 调用的工具
    """
    def decorator(func: Callable) -> Callable:
        # 注册到全局注册表
        _tool_registry[func.__name__] = func
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> str:
            return func(*args, **kwargs)
        
        # 添加工具元数据
        wrapper._is_tool = True
        wrapper._tool_description = description
        wrapper._tool_name = func.__name__
        
        return wrapper
    
    return decorator

def get_tools_schema() -> List[Dict]:
    """获取工具定义列表（供 LLM 使用）"""
    tools = []
    for name, func in _tool_registry.items():
        if hasattr(func, '_is_tool'):
            # 从函数签名构建 schema
            import inspect
            sig = inspect.signature(func)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                properties[param_name] = {
                    "type": "string",  # 简化处理
                    "description": f"参数 {param_name}"
                }
                required.append(param_name)
            
            tools.append({
                "name": func._tool_name,
                "description": func._tool_description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })
    return tools

def execute_tool(tool_name: str, tool_input: Dict) -> str:
    """执行工具调用"""
    if tool_name not in _tool_registry:
        return f"Error: Unknown tool {tool_name}"
    
    func = _tool_registry[tool_name]
    try:
        return func(**tool_input)
    except Exception as e:
        return f"Error: {str(e)}"

# 使用示例
@tool_call(description="获取当前天气")
def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    return f"{location} 天气晴朗，25°C"

print(get_tools_schema())
```

**考察点**：装饰器模式、工具注册、schema 生成

**追问方向**：
- 如何支持嵌套参数和复杂类型？
- 工具执行失败时如何处理？
- 如何实现工具调用的流式输出？

---

### 4.2 MCP Server 实现

**难度**：中等 | **预估用时**：10-15 分钟 | **高频指数**：⭐⭐⭐

**题目**：使用 MCP SDK 实现一个简单的天气查询服务器

```python
# mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
from pydantic import AnyUrl
import asyncio

# TODO: 实现 MCP Server
```

**参考解答**：

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
import asyncio

# 创建服务器实例
server = Server("weather-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """暴露工具列表"""
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的天气信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    }
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    if name == "get_weather":
        city = arguments.get("city", "")
        
        # 模拟天气查询（实际项目中调用天气 API）
        weather_data = {
            "北京": {"temp": 22, "condition": "晴", "humidity": 45},
            "上海": {"temp": 25, "condition": "多云", "humidity": 60},
            "深圳": {"temp": 28, "condition": "阵雨", "humidity": 75}
        }
        
        if city in weather_data:
            data = weather_data[city]
            result = f"{city}天气：{data['condition']}，温度{data['temp']}°C，湿度{data['humidity']}%"
        else:
            result = f"抱歉，暂不支持查询 {city} 的天气"
        
        return [TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """启动服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**考察点**：MCP 协议理解、工具定义、服务器实现

**追问方向**：
- MCP Server 和 LLM 之间如何通信？
- 如何处理工具执行的超时和错误？
- 如何实现多个工具的并行调用？

---

## 5. 实用工具函数

### 5.1 Token 数量估算

**难度**：简单 | **预估用时**：3-5 分钟 | **高频指数**：⭐⭐⭐

**题目**：实现一个简单的 token 估算函数（中文和英文分别估算）

```python
def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量
    
    规则：
    - 英文：1 token ≈ 4 characters
    - 中文：1 token ≈ 1-2 characters（GPT 约 2，中Claude约 1）
    
    Returns:
        估算的 token 数量
    """
    # TODO: 实现代码
    pass
```

**参考解答**：

```python
import re

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    
    # 分离中英文
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z0-9\s]', text))
    other_chars = len(text) - chinese_chars - english_chars
    
    # 英文按字符数/4估算（Claude 3.5 的近似值）
    english_tokens = english_chars / 4
    
    # 中文按字符数估算（Claude 中汉字约 1 token）
    chinese_tokens = chinese_chars
    
    # 其他字符按经验估算
    other_tokens = other_chars / 2
    
    return int(english_tokens + chinese_tokens + other_tokens)

# 测试
tests = [
    "Hello, world!",
    "你好，世界！",
    "Hello 你好 world 世界",
    "The quick brown fox jumps over the lazy dog"
]

for t in tests:
    print(f"{t!r}: {estimate_tokens(t)} tokens")
```

**追问方向**：
- 为什么中英文 token 估算方式不同？
- tiktoken 库了解吗？如何使用它精确计算？
- 为什么要做 token 预算？

---

### 5.2 上下文窗口压缩

**难度**：中等 | **预估用时**：8-10 分钟 | **高频指数**：⭐⭐⭐⭐

**题目**：实现一个简单的上下文压缩函数，当对话历史过长时压缩早期消息

```python
def compress_messages(messages: list, max_tokens: int = 100000) -> list:
    """
    压缩消息历史，保持最新的消息，压缩旧消息
    
    策略：
    1. 如果总 token 超过 max_tokens，从最旧的消息开始删除
    2. 优先保留 user 消息，压缩连续的 assistant 消息
    
    Returns:
        压缩后的消息列表
    """
    # TODO: 实现代码
    pass
```

**参考解答**：

```python
import anthropic

client = anthropic.Anthropic()

def estimate_tokens(text: str) -> int:
    """快速估算 token"""
    return len(text) // 4  # 简化估算

def compress_messages(messages: list, max_tokens: int = 100000) -> list:
    """压缩消息历史"""
    if not messages:
        return []
    
    # 计算当前总 token
    def calc_total_tokens(msgs):
        return sum(estimate_tokens(m.get("content", "")) for m in msgs)
    
    # 如果已经满足要求，直接返回
    if calc_total_tokens(messages) <= max_tokens:
        return messages
    
    compressed = messages.copy()
    
    while calc_total_tokens(compressed) > max_tokens and len(compressed) > 2:
        # 策略1：删除最早的 assistant 消息
        for i, msg in enumerate(compressed):
            if msg.get("role") == "assistant":
                compressed.pop(i)
                break
        else:
            # 策略2：如果没有 assistant 消息了，删除最早的消息
            compressed.pop(0)
    
    return compressed

# 压缩后的消息添加摘要说明（可选）
def compress_with_summary(messages: list, max_tokens: int = 100000) -> list:
    """带摘要的压缩版本"""
    if not messages:
        return []
    
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    
    if total_tokens <= max_tokens:
        return messages
    
    # 生成摘要
    summary_prompt = "请简要总结以下对话的要点：\n\n"
    for msg in messages[:-1]:  # 不包括最后一条（最新）
        role = "用户" if msg.get("role") == "user" else "助手"
        summary_prompt += f"{role}：{msg.get('content', '')[:200]}...\n"
    
    # 调用 LLM 生成摘要（这里简化处理）
    summary = "[对话历史摘要：用户和助手讨论了多个技术问题...]"
    
    # 返回摘要 + 最近的消息
    return [
        {"role": "system", "content": f"以下是对话历史摘要：{summary}"},
        messages[-1]  # 保留最新的消息
    ]
```

**考察点**：上下文窗口管理、消息压缩策略、Token 预算

**追问方向**：
- 为什么需要压缩上下文而不是直接截断？
- 如何选择保留哪些消息？（重要性、相关性、时间）
- 压缩后如何保证对话连贯性？

---

## 各身份难度参考

| 题目 | 实习 | 应届 | 社招 1-3 年 |
|------|:----:|:----:|:----------:|
| 1.1 设计 System Prompt | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| 1.2 Prompt 改写优化 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 2.1 ReAct 循环实现 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 2.2 Agent 决策逻辑 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 3.1 分块策略实现 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 3.2 混合检索实现 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 3.3 重排序实现 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 4.1 工具调用封装 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 4.2 MCP Server 实现 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 5.1 Token 估算 | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| 5.2 上下文压缩 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
