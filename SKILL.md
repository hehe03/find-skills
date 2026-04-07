---
name: find-skills
description: 当用户想查找适合完成某项任务的 skills，或想查看当前已注册的 skills 时，调用本地已启动的 skill_recommend_server API 服务完成查询与推荐。
---

# Find Skills

当用户需要“找适合当前任务的 skill”“看看现在库里有哪些 skill”“根据任务推荐可用 skill”时，使用这个技能。

默认假设 `skill_recommend_server` 已经启动，并监听：

- `http://localhost:8000`

如果接口不可用，再提示用户检查服务是否已启动。

## 何时使用

- 用户描述一个任务，想找到最合适的 skills。
- 用户想查看当前已注册的 skills 列表。
- 用户想通过 API 查询推荐结果，而不是手动阅读 `skills-hub`。

## 工作方式

### 1. 查询推荐

当用户输入“我现在要做什么”这一类任务描述时，调用：

```bash
curl -X POST http://localhost:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"用户的任务描述\"}"
```

PowerShell 可用：

```powershell
$body = @{ query = "用户的任务描述" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/recommend" -ContentType "application/json" -Body $body
```

返回结果里重点关注：

- `recommendations`
- `score`
- `score_band`
- `match_reason`
- `matched_terms`
- `query_understanding`

### 2. 查询当前 skills 列表

当用户想知道“现在系统里有哪些 skills”时，调用：

```bash
curl http://localhost:8000/skills
```

PowerShell 可用：

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/skills"
```

### 3. 查看反馈统计

当用户想查看推荐反馈统计时，调用：

```bash
curl http://localhost:8000/stats/feedback
```

## 使用要求

- 优先调用 API，不要直接假设 skill 内容。
- 推荐类问题优先用 `/recommend`，而不是自己从文件夹里猜。
- 列表类问题优先用 `/skills`。
- 如果用户关心“为什么推荐这些 skill”，要把 `score`、`score_band`、`match_reason` 和 `matched_terms` 一起解释给用户。
- 如果 API 请求失败，明确说明是服务不可用，而不是伪造推荐结果。

## 返回建议

向用户汇报推荐结果时，优先使用这种结构：

1. 用户任务的简短理解
2. 推荐到的前几个 skills
3. 每个 skill 的分数与匹配原因
4. 如果结果较弱，指出“当前没有高分匹配”

## 典型示例

### 示例 1：推荐 skill

用户说：

```text
分析产品的竞争力和改进方向
```

Agent 应调用：

```bash
curl -X POST http://localhost:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"分析产品的竞争力和改进方向\"}"
```

### 示例 2：查看已注册 skills

用户说：

```text
看看现在都注册了哪些 skills
```

Agent 应调用：

```bash
curl http://localhost:8000/skills
```

## 注意事项

- 本技能依赖本地运行中的 `skill_recommend_server`。
- 默认地址是 `http://localhost:8000`。
- 如果后续服务地址变更，应同步修改这里的接口地址。
