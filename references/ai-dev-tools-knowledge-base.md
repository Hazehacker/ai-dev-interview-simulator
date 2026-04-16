# AI 编程工具知识库（可选扩展）

> 本文件为**可选参考**。当候选人表示对 AI 编程工具有兴趣或经验时（如使用 Copilot/Cursor/Claude Code 等工具辅助 AI 应用开发），面试官可从本知识库中选题，作为 AI 工具模块的扩展提问。
>
> 覆盖方向：AI 编程工具使用经验、AI 辅助 AI 应用开发、Vibe Coding 实践。

---

## 目录

- [1. AI 编程工具使用](#1-ai-编程工具使用)
  - [1.1 主流 AI 编程工具](#11-主流-ai-编程工具)
  - [1.2 AI 编程原理](#12-ai-编程原理)
  - [1.3 编写有效的 AI 编程 Prompt](#13-编写有效的-ai-编程-prompt)
- [2. Vibe Coding 实践](#2-vibe-coding-实践)
  - [2.1 Vibe Coding 是什么](#21-vibe-coding-是什么)
  - [2.2 Vibe Coding 工作流](#22-vibe-coding-工作流)
  - [2.3 AI 应用开发中的 Vibe Coding](#23-ai-应用开发中的-vibe-coding)
- [3. AI Coding Agent 架构](#3-ai-coding-agent-架构)
  - [3.1 Agent Loop 核心原理](#31-agent-loop-核心原理)
  - [3.2 上下文管理策略](#32-上下文管理策略)
  - [3.3 工具调用与 Skill 机制](#33-工具调用与-skill-机制)
- [4. AI 辅助 AI 应用开发](#4-ai-辅助-ai-应用开发)
  - [4.1 RAG 项目中的 AI 辅助](#41-rag-项目中的-ai-辅助)
  - [4.2 Agent 开发中的 AI 辅助](#42-agent-开发中的-ai-辅助)
  - [4.3 Prompt 工程自动化](#43-prompt-工程自动化)

---

## 1. AI 编程工具使用

### 1.1 主流 AI 编程工具

**你用过哪些 AI 编程工具？**
- GitHub Copilot：基于 OpenAI Codex，深度集成 VS Code、JetBrains IDE
- Cursor：基于 Claude 的 AI IDE，支持多模态、Agent 模式
- Claude Code：Anthropic 官方 CLI 工具，擅长复杂任务分解
- Windsurf：Codeium 推出的 AI IDE
- Trae：字节跳动推出的 AI IDE
- Roo Code（前 Codium）：开源 AI 编程扩展

**各工具在 AI 应用开发中的表现？**
- Cursor：适合同时处理前端和 AI 应用后端，多模态能力强
- Claude Code：Agent Loop 架构清晰，适合学习 AI 编程原理
- Copilot：代码补全速度快，但复杂 Agent 能力较弱

### 1.2 AI 编程原理

**AI 编程工具的核心原理？**
- **上下文感知的代码补全**：基于当前文件、相关文件、注释生成建议
- **自然语言转代码**：将注释/Prompt 转化为可执行代码
- **代码审查与优化建议**：理解代码逻辑后提供改进意见
- **多模态理解**：支持截图、UI 设计稿等视觉输入

**AI 编程在 AI 应用开发中的独特价值？**
- Prompt 迭代优化：快速生成和测试不同 Prompt 变体
- Agent 流程调试：帮助理解 Agent Loop 的执行逻辑
- RAG 效果验证：快速生成测试用例验证召回质量

### 1.3 编写有效的 AI 编程 Prompt

**如何写出好的 AI 编程 Prompt？**
- **明确任务而非模糊描述**：❌ "帮我写个 RAG" ✅ "写一个 RAG 检索类，用 Faiss 向量数据库，支持混合检索"
- **提供完整上下文**：LangChain 版本、向量模型、索引配置
- **指定输出格式**：是否需要流式输出、返回格式
- **分步骤引导**：复杂任务拆分成多个小步骤

**Prompt 示例对比**

❌ 模糊 Prompt：
```
写一个 Agent
```

✅ 清晰 Prompt：
```
用 LangChain 写一个 ReAct Agent，包含：
- 工具：search（搜索网络）、calculator（计算）、read_file（读取文件）
- System Prompt 要求：角色是研究助手，思维链要求中文输出
- 使用 Claude 3.5 Sonnet 作为 LLM
- 最大迭代次数 10 次
```

---

## 2. Vibe Coding 实践

### 2.1 Vibe Coding 是什么

**Vibe Coding 定义**
- 由 Andrej Karpathy 提出的概念
- 核心思想：用自然语言对话的方式进行编程
- 人类负责"说出想要什么"，AI 负责"实现"
- 特点：快速原型开发、降低编程门槛、适合探索性开发

**Vibe Coding vs 传统编程**
- 传统编程：代码 → 运行 → 调试（人主导）
- Vibe Coding：对话 → 代码 → 运行（AI 主导）
- 本质是编程生产力的提升，而非替代程序员

### 2.2 Vibe Coding 工作流

**典型 Vibe Coding 流程**
1. **需求表达**：用自然语言描述想要的功能
2. **AI 生成**：AI 生成代码并解释
3. **迭代调整**：指出问题，AI 修复
4. **验证测试**：运行测试，确认功能

**Vibe Coding 的适用场景**
- 快速原型：验证想法的可行性
- 学习新框架：让 AI 解释代码，然后自己修改
- 自动化脚本：一次性脚本，数据处理任务
- 简单功能：CRUD 操作、API 调用封装

**Vibe Coding 的局限性**
- 代码质量参差不齐，需要有能力审核
- 复杂系统难以一次性生成
- 容易产生幻觉代码（看似正确但运行有问题）

### 2.3 AI 应用开发中的 Vibe Coding

**RAG 项目中的 Vibe Coding**
```
对话示例：
User: "我想做一个基于 RAG 的知识库问答，先用 LangChain 和 Faiss"
AI: "好的，我来帮你搭建基础架构..."
User: "分块策略我想用句子级别，不要重叠"
AI: "明白了，修改 chunk_size 为按句子分割..."
User: "添加一个重排序功能，用 BGE 模型"
AI: "好的，我来添加 reranker 模块..."
```

**Agent 项目中的 Vibe Coding**
```
对话示例：
User: "用 LangGraph 写一个客服 Agent"
AI: "我来设计一个基于 LangGraph 的客服 Agent 架构..."
User: "加入多轮对话记忆，用 Redis 存储"
AI: "好的，添加 Redis memory 模块..."
User: "当用户问退货政策时，优先用规则匹配而不是 LLM"
AI: "明白了，添加规则引擎作为优先路由..."
```

---

## 3. AI Coding Agent 架构

### 3.1 Agent Loop 核心原理

**AI Coding Agent 的核心架构是什么？**
- 本质是一个 **while 循环（Agent Loop）**
- 读取用户输入 → 组装上下文（System Prompt + 对话历史 + 工具定义）
- 调用 LLM → 解析响应
- 如果包含工具调用则执行工具，并将结果追加到上下文
- 循环，直到模型输出纯文本（不再调用工具）为止

**Agent Loop 关键组件**
1. **System Prompt**：定义角色、约束、可用工具列表
2. **对话历史（Message History）**：user/assistant/tool 三种角色的消息序列
3. **工具注册表（Tool Registry）**：通过 JSON Schema 描述工具签名
4. **工具执行器（Tool Executor）**：解析模型输出的工具调用 → 路由到对应实现 → 执行 → 返回结果
5. **终止条件**：模型不再发起工具调用 / 达到最大轮次 / 用户中断

**Claude Code 的工作流程**
```
用户输入 → 组装上下文 → 调用 Claude → 解析响应
    ↓
如果包含工具调用 → 执行工具 → 返回结果追加到历史
    ↓
循环直到终止
```

### 3.2 上下文管理策略

**为什么上下文管理很重要？**
- 对话越长，Token 消耗越多，成本越高
- 上下文窗口有限，需要优先保留重要信息
- 无关信息会稀释关键上下文，降低生成质量

**上下文管理策略**
1. **上下文压缩/截断（Truncation）**：当接近上下文窗口上限时，压缩早期对话为摘要
2. **按优先级保护**：
   - 高优先级：System Prompt 和最近几轮对话
   - 中优先级：工具执行结果
   - 低优先级：早期对话历史
3. **工具结果裁剪**：大文件读取结果可能被截断，只保留关键部分

**Prompt Cache 机制**
- Anthropic 的 Prompt Cache：当多次 API 调用的 prompt 前缀相同时，后续调用可以复用前缀的计算结果
- 缓存按 prompt 前缀严格匹配，前缀相同则缓存命中
- 缓存有 5 分钟 TTL
- 设计技巧：把稳定内容（System Prompt、工具定义）放前面，变化内容放后面

### 3.3 工具调用与 Skill 机制

**Function Calling 是什么？**
- 允许 LLM 调用外部工具/函数的能力
- 模型输出结构化的函数调用请求（而非自然语言）
- 应用解析并执行后返回结果
- 底层依赖 JSON Schema 描述函数签名

**Skill 机制（以 Claude Code 为例）**
- Skill 是以自然语言（Markdown）编写的可复用能力模块
- 与 Function Calling 的区别：Function Calling 通过 JSON Schema 定义参数；Skill 通过 SKILL.md 定义完整执行指令
- **两阶段加载**：
  1. 轻量注入：每个 Skill 的 name + description 始终注入 system prompt
  2. 按需激活：当用户请求匹配 Skill description 时，完整 SKILL.md 被加载

**好的 Skill description 怎么写？**
- 必须包含：触发条件（TRIGGER when）、不触发条件（DO NOT TRIGGER when）
- 核心能力描述要准确，避免歧义
- 写得太模糊 → 该激活时不激活
- 写得太宽泛 → 不该激活时误触发

---

## 4. AI 辅助 AI 应用开发

### 4.1 RAG 项目中的 AI 辅助

**适合 AI 辅助的 RAG 任务**
- 分块策略代码生成
- 向量数据库操作封装
- 检索结果后处理
- RAG 评测代码生成

**Prompt 示例：RAG 分块策略**
```
用 LangChain 写一个中文文档分块器，要求：
- 基于中文标点符号和换行符分割
- 每个 chunk 不超过 500 字符
- 相邻 chunk 之间保留 50 字符的重叠
- 返回 chunk 列表和对应的元数据
```

### 4.2 Agent 开发中的 AI 辅助

**适合 AI 辅助的 Agent 任务**
- ReAct 循环代码生成
- 工具定义 JSON Schema 生成
- System Prompt 优化
- Agent 评测用例生成

**Prompt 示例：ReAct Agent**
```
用 LangChain 的 ReAct 代理写一个研究助手 Agent，包含：
- 工具：search（搜索网络）、read_file（读取本地文件）
- System Prompt：角色是研究助手，擅长信息整理
- 最大迭代次数 5 次
- 当迭代超过 3 次时，总结当前进度并询问是否继续
```

### 4.3 Prompt 工程自动化

**AI 辅助 Prompt 优化**
```
Prompt: "优化以下 System Prompt，使其更清晰、更能避免幻觉：

原始 Prompt：
你是一个客服机器人，要回答用户的问题。

优化要求：
1. 明确角色定位
2. 定义边界（什么问题不回答）
3. 指定输出格式
4. 添加安全约束
```

**Prompt 版本管理**
- 用 AI 生成多个 Prompt 变体
- 用测试集评估效果
- 选择最佳版本

---

## 各身份难度参考

| 题目 | 实习 | 应届 | 社招 1-3 年 |
|------|:----:|:----:|:----------:|
| AI 编程工具使用经验 | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| Vibe Coding 理解 | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| Agent Loop 原理 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 上下文管理策略 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Skill 机制理解 | — | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Prompt 工程 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| AI 辅助 RAG/Agent 开发 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
