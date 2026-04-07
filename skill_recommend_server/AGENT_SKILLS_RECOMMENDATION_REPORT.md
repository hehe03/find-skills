# Agent Skills 推荐系统设计方案

## 一、项目背景

当前拥有一个包含数百个 Agent Skills 的货架，需要根据用户输入的问题或需求，自动推荐能够完成用户任务的 Skills。

---

## 二、业界解决方案分析

### 2.1 技能发现与推荐相关开源项目

| 项目名称 | 描述 | Stars | 技术特点 |
|---------|------|-------|----------|
| **EvoSkill** | 自动从失败轨迹中发现和合成可复用技能，提升编码 Agent 在长期任务中的表现 | 312 | 自动化技能发现、轨迹学习 |
| **SkillOrchestra** | 通过技能路由学习，动态选择合适的 Agent | 48 | 技能编排、动态路由 |
| **AutoSkill** | 终身学习框架，通过技能自演进持续提升 | 149 | 持续学习、技能演化 |
| **XSkill** | 多模态 Agent 的持续学习框架 | 145 | 多模态、经验学习 |
| **dmgrok/agent_skills_directory** | 智能技能发现，支持质量验证和维护追踪 | 12 | 智能推荐、安全扫描 |
| **SkillSmith** | 技能发现、安全、优化和管理 | 33 | TypeScript/全栈 |

### 2.2 MCP 生态相关方案

| 项目/平台 | 描述 | 特点 |
|----------|------|------|
| **MCP Registry** | 官方 MCP 服务器注册中心 | 标准化发现机制 |
| **Smithery** | MCP 服务器注册平台 | 社区驱动、易用 |
| **skillhub-mcp** | AI 资源智能推荐 MCP 服务器 | 离线可用、20k+ 资源 |
| **awesome-agent-skills-mcp** | 暴露 100+ AI Agent 技能 | 兼容 Claude/Copilot |

### 2.3 向量搜索与语义匹配方案

| 方案 | 描述 | 适用场景 |
|------|------|----------|
| **FAISS** | Facebook 开源向量搜索库 | 高性能、本地部署 |
| **Qdrant** | 高性能向量搜索引擎 | 大规模语义搜索 |
| **txtai** | 语义搜索 + LLM 编排框架 | 一站式 AI 框架 |
| **Milvus** | 开源向量数据库 | 生产级部署 |

### 2.4 监控与可观测性方案

| 项目 | 描述 | 特点 |
|------|------|------|
| **Langfuse** | LLM 应用可观测平台 | 开源、支持自托管 |
| **LangSmith** | LLM 应用调试监控 | 功能全面、付费 |
| **OpenTelemetry** | 标准可观测性框架 | 厂商中立 |

### 2.5 技能推荐核心方法论

1. **语义向量匹配**：将 Skills 描述和用户需求都编码为向量，用余弦相似度匹配
2. **技能图谱**：构建技能之间的依赖关系图，辅助推荐
3. **LLM 理解**：利用大语言模型理解用户意图和技能能力，进行语义推理
4. **元数据过滤**：基于标签、类别、版本等元数据进行精确筛选

---

## 三、系统设计方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                       │
│                  Web / API / SDK / CLI                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      推荐引擎层 (Recommendation Engine)          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Query理解   │  │ 意图识别    │  │ 技能匹配    │              │
│  │ (意图解析)   │  │ (分类/标签) │  │ (向量召回)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ LLM重排序   │  │ 反馈处理    │  │ 监控记录    │              │
│  │ (可配置)    │  │ (用户反馈)  │  │ (Langfuse) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       数据层 (Data Layer)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Skill注册表 │  │ 向量索引    │  │ 技能图谱    │              │
│  │ (元数据)    │  │ (FAISS)     │  │ (关联关系)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │ 反馈数据    │  │ 配置管理    │                               │
│  │ (用户反馈)  │  │ (YAML)      │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块设计

#### 3.2.1 技能注册与管理模块

**功能**：管理 Skills 的元数据存储和索引

```yaml
Skill Schema:
  id: string                    # 唯一标识
  name: string                  # 技能名称
  description: string           # 详细描述
  category: string[]            # 分类标签
  capabilities: string[]         # 能力列表
  input_schema: object          # 输入参数模式
  output_schema: object         # 输出参数模式
  version: string               # 版本号
  dependencies: string[]        # 依赖技能
  author: string                # 作者
  created_at: timestamp          # 创建时间
  updated_at: timestamp          # 更新时间
  usage_stats: object           # 使用统计
  embedding: float[]            # 向量嵌入（预计算）
```

#### 3.2.2 向量嵌入模块

**功能**：将 Skills 描述和用户查询转换为语义向量

| 方案 | 优点 | 缺点 |
|------|------|------|
| **BGE-M3** | 开源、可本地部署、多语言 | 需要运维 |
| **OpenAI text-embedding-3** | 效果好、生态成熟 | 需 API、隐私 |
| **Cohere** | 多语言支持好 | 需 API |

**推荐**：BGE-M3（本地部署，隐私优先）

#### 3.2.3 查询理解模块

**功能**：解析用户输入，提取关键信息

```python
# 处理流程
User Query → 意图分类 → 关键实体提取 → 查询重写 → 向量生成

# 意图分类示例
- "帮我写个API" → capability: code_generation, target: api
- "处理图片" → capability: image_processing
- "分析数据" → capability: data_analysis
```

#### 3.2.4 推荐匹配引擎

**功能**：多策略融合的智能推荐

```
推荐策略：
1. 向量召回 (Vector Recall)     # 语义相似度 TOP-K
2. 精确过滤 (Exact Filter)      # 分类/标签精确匹配
3. LLM rerank                   # 重排序优化（可配置）
4. 图谱扩展 (Graph Expand)      # 依赖技能补充
```

**推荐结果结构**：
```json
{
  "recommendations": [
    {
      "skill_id": "skill_001",
      "name": "API Generator",
      "confidence": 0.95,
      "match_reason": "语义匹配度高: code_generation + api",
      "alternatives": ["skill_002", "skill_005"]
    }
  ],
  "query_understanding": {
    "intent": "api_development",
    "entities": ["API", "code"],
    "rewritten_query": "生成 REST API 代码"
  }
}
```

#### 3.2.5 监控与反馈模块

**功能**：记录执行过程、收集用户反馈、可选启用 Langfuse

```python
# 监控事件类型
Events:
  - query_received      # 用户查询接收
  - query_understood     # 查询理解完成
  - vector_search_done  # 向量搜索完成
  - llm_rerank_done     # LLM 重排完成
  - recommendation_generated  # 推荐生成
  - feedback_received   # 用户反馈接收
```

**反馈数据结构**：
```json
{
  "recommendation_id": "rec_001",
  "user_id": "user_123",
  "rating": 4,
  "comment": "推荐结果符合预期",
  "selected_skill": "skill_001",
  "timestamp": "2026-04-01T10:00:00Z"
}
```

**Langfuse 集成（可选）**：
```python
# config.yaml
langfuse:
  enabled: true
  public_key: "pk-xxx"
  secret_key: "sk-xxx"
  host: "https://cloud.langfuse.com"  # 或自托管地址
```

```python
# 使用示例
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=config.langfuse.public_key,
    secret_key=config.langfuse.secret_key,
    host=config.langfuse.host
)

# 记录 trace
trace = langfuse.trace(name="skill_recommendation")
trace.event(query_received={"query": user_input})
trace.event(vector_search_done={"results_count": len(results)})
```

#### 3.2.6 大模型 API 接入模块

**功能**：支持多种 LLM 提供商的可插拔架构

```python
# LLM 适配器接口
class LLMAdapter(Protocol):
    def chat(self, messages: List[Message], **kwargs) -> str: ...
    def embed(self, text: str) -> List[float]: ...
```

```python
# 支持的 LLM 提供商
class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def chat(self, messages, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

class AnthropicAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def chat(self, messages, **kwargs):
        # 转换消息格式
        return self.client.messages.create(
            model=self.model,
            **kwargs
        )

class OllamaAdapter(LLMAdapter):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
    
    def chat(self, messages, **kwargs):
        # 本地模型调用
        pass
```

#### 3.2.7 配置管理模块

**功能**：统一的配置文件管理，支持多环境切换

**配置文件结构**：
```yaml
# config.yaml
app:
  name: "Skill Recommender"
  version: "1.0.0"
  debug: false

# 向量存储配置
vector_store:
  type: "faiss"
  dimension: 1024  # BGE-M3 输出维度
  index_path: "./data/faiss_index"
  metric: "ip"  # 内积相似度

# Embedding 模型配置
embedding:
  provider: "bge-m3"
  model_path: "./models/bge-m3"
  device: "cuda"  # 或 cpu
  # 备选方案：使用 API
  # provider: "openai"
  # model: "text-embedding-3-small"

# LLM 配置（可多选）
llm:
  default: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o"
      temperature: 0.7
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-5-sonnet-20241022"
    ollama:
      base_url: "http://localhost:11434"
      model: "qwen2.5"
    custom:
      api_key: "${CUSTOM_API_KEY}"
      base_url: "https://custom-llm.example.com/v1"
      model: "custom-model"

# 监控配置
langfuse:
  enabled: false  # 可选开启
  public_key: "${LANGFUSE_PUBLIC_KEY}"
  secret_key: "${LANGFUSE_SECRET_KEY}"
  host: "https://cloud.langfuse.com"

# 数据库配置
database:
  type: "sqlite"  # 或 postgresql
  path: "./data/skills.db"
  # postgresql:
  #   host: "localhost"
  #   port: 5432
  #   database: "skills"
  #   user: "admin"
  #   password: "${DB_PASSWORD}"

# 推荐配置
recommendation:
  vector_top_k: 50
  final_top_k: 5
  similarity_threshold: 0.5
  enable_llm_rerank: true
  enable_graph_expand: true

# API 服务配置
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "http://localhost:3000"
```

**环境变量支持**：
```bash
# .env 文件
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
DB_PASSWORD=secret
```

**配置加载实现**：
```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Any
import yaml

class AppConfig(BaseSettings):
    app: Dict[str, Any]
    vector_store: Dict[str, Any]
    embedding: Dict[str, Any]
    llm: Dict[str, Any]
    langfuse: Dict[str, Any]
    database: Dict[str, Any]
    recommendation: Dict[str, Any]
    server: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml"):
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

config = AppConfig.from_yaml("config.yaml")
```

### 3.3 技术选型

| 层级 | 技术选型 | 理由 |
|------|----------|------|
| **向量存储** | FAISS | 高性能、开源、本地部署、隐私优先 |
| **Embedding** | BGE-M3 | 开源多语言、可本地部署、效果优秀 |
| **LLM** | 可配置（OpenAI/Claude/Ollama/自定义） | 灵活切换、按需选择 |
| **监控** | Langfuse（可选） | 开源可自托管、追踪完整 |
| **API框架** | FastAPI | 轻量、易部署、类型安全 |
| **前端** | React / Vue | 用户界面 |
| **存储** | SQLite / PostgreSQL | 轻量/生产级 |
| **配置** | YAML + 环境变量 | 易于维护、多环境支持 |

### 3.4 核心流程

```
1. 离线构建
   Skill元数据 → 文本提取 → BGE-M3 Embedding → FAISS索引构建 → 图谱构建

2. 在线推荐
   用户输入 → 意图识别 → FAISS向量搜索 → 候选筛选 → LLM重排 → 结果返回
   ↓
   Langfuse追踪（可选）→ 记录完整执行链路
   ↓
   用户反馈 → 存储反馈数据 → 用于后续优化
```

### 3.5 关键技术实现

#### 3.5.1 FAISS 向量搜索实现

```python
import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension: int = 1024, index_path: str = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引
        self.metadata = []
        self.index_path = index_path
    
    def add_vectors(self, embeddings: np.ndarray, metadata: list):
        # L2 归一化
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({
                    "score": float(dist),
                    "metadata": self.metadata[idx]
                })
        return results
    
    def save(self, path: str):
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        self.index = faiss.read_index(path)
```

#### 3.5.2 BGE-M3 Embedding 实现

```python
from FlagEmbedding import BGEM3FlagModel

class BGEEmbedding:
    def __init__(self, model_path: str = "./models/bge-m3", device: str = "cuda"):
        self.model = BGEM3FlagModel(model_path, use_fp16=True)
        self.device = device
    
    def encode(self, texts: list[str]) -> np.ndarray:
        result = self.model.encode(
            texts,
            batch_size=32,
            max_length=1024,
            device=self.device
        )
        return result["dense_vecs"]
    
    def encode_query(self, query: str) -> np.ndarray:
        result = self.model.encode(
            [query],
            batch_size=1,
            max_length=1024,
            device=self.device
        )
        return result["dense_vecs"][0]
```

#### 3.5.3 多策略融合

```python
def hybrid_recommend(query: str, filters: dict, config: AppConfig):
    # 1. 向量召回
    query_embedding = embedding_model.encode_query(query)
    vector_results = faiss_store.search(query_embedding, top_k=config.recommendation.vector_top_k)
    
    # 2. 精确过滤
    if filters.get("category"):
        vector_results = filter_by_category(vector_results, filters["category"])
    
    # 3. LLM 重排序
    if config.recommendation.enable_llm_rerank:
        reranked = llm_rerank(query, vector_results, top_k=config.recommendation.final_top_k)
    else:
        reranked = vector_results[:config.recommendation.final_top_k]
    
    # 4. 补充依赖技能
    if config.recommendation.enable_graph_expand:
        expanded = expand_dependencies(reranked)
    
    return expanded
```

#### 3.5.4 反馈收集与处理

```python
class FeedbackHandler:
    def __init__(self, db: Database):
        self.db = db
    
    def collect_feedback(self, feedback: FeedbackData):
        # 存储反馈
        self.db.feedbacks.insert(feedback.model_dump())
        
        # 更新技能使用统计
        self.db.skills.update(
            {"id": feedback.selected_skill},
            {"$inc": {"usage_stats.times_selected": 1}}
        )
    
    def get_negative_feedback_analysis(self):
        # 分析低分反馈，用于优化推荐
        pipeline = [
            {"$match": {"rating": {"$lte": 2}}},
            {"$group": {
                "_id": "$selected_skill",
                "count": {"$sum": 1},
                "comments": {"$push": "$comment"}
            }},
            {"$sort": {"count": -1}}
        ]
        return list(self.db.feedbacks.aggregate(pipeline))
```

---

## 四、实施路线图

### 4.1 Phase 1：基础功能 MVP (2-3周)

- [ ] 技能元数据存储 (SQLite/PostgreSQL)
- [ ] 基础向量搜索 (FAISS + BGE-M3)
- [ ] 配置文件加载模块
- [ ] 简单 API 接口
- [ ] 基础 Web 界面

### 4.2 Phase 2：智能推荐增强 (2-3周)

- [ ] 意图识别模块
- [ ] 多策略融合推荐
- [ ] LLM 重排序（可配置多提供商）
- [ ] 推荐结果展示优化

### 4.3 Phase 3：监控与反馈 (2-3周)

- [ ] Langfuse 集成（可选开关）
- [ ] 用户反馈收集
- [ ] 反馈数据存储与分析
- [ ] 使用统计面板

### 4.4 Phase 4：高级功能 (2-3周)

- [ ] 技能图谱构建
- [ ] 依赖推荐
- [ ] 个性化推荐（基于历史反馈）

### 4.5 Phase 5：运维与优化 (持续)

- [ ] 性能优化
- [ ] 监控告警
- [ ] 本地化部署支持
- [ ] 持续迭代优化

---

## 五、总结

本方案综合了业界主流的技能发现与推荐方案，提出了一套完整的 Agent Skills 推荐系统设计。核心思路是：

1. **向量语义搜索**：基于 BGE-M3 + FAISS 的本地化语义匹配，隐私优先
2. **多策略融合**：结合向量召回、精确过滤、可配置 LLM 重排，提升准确率
3. **可扩展架构**：支持后续扩展到技能图谱、个性化推荐等高级功能
4. **监控反馈**：Langfuse 可选集成 + 用户反馈闭环，持续优化
5. **灵活配置**：YAML 配置 + 环境变量，支持多环境多提供商

技术选型遵循「开源、本地部署、灵活配置」原则，便于后续维护和扩展。

---

*文档版本：v1.1*  
*创建时间：2026-04-01*  
*更新内容：增加监控反馈、LLM可配置、配置文件支持，技术选型调整为FAISS + BGE-M3*
