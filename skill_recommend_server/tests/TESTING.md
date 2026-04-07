# Testing Guide

本目录包含 `skill_recommend_server` 的最小回归测试集，目标是覆盖项目的核心主链路，而不是只验证入口是否存在。

## 测试文件说明

- `test_entrypoints.py`
  覆盖包入口、`python -m skills_recommender` 入口、以及 `SkillRecommenderApp._replace_catalog()` 的关键同步逻辑。

- `test_catalog.py`
  覆盖 `SkillCatalog` / `SkillSpec` 的基础行为，包括添加技能、分类搜索、使用统计和序列化往返。

- `test_recommendation.py`
  用 stub 的 embedding / vector store / LLM 覆盖推荐引擎行为，包括分类过滤、LLM rerank 和索引构建。

- `test_storage.py`
  覆盖 `skills-hub` 导入、`SKILL.md` 解析、JSON/SQLite 持久化、数据库查询以及向量索引文件存在性判断。

- `test_config_monitoring.py`
  覆盖配置加载、环境变量覆盖和反馈监控统计。

## 运行方法

在项目根目录下运行全部测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

单独运行某一类测试：

```bash
python -m unittest tests.test_storage -v
python -m unittest tests.test_recommendation -v
```

## 跳过说明

有些测试依赖可选库：

- 缺少 `fastapi` 时，API 相关条件测试会自动跳过。
- 缺少 `faiss` 时，依赖向量库导入的条件测试会自动跳过。

这类跳过不表示失败，只表示当前环境没有安装对应依赖。

## 建议流程

开发中建议至少运行：

```bash
python -m unittest tests.test_catalog tests.test_storage tests.test_recommendation -v
```

提交前建议运行全部测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
