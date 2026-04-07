#!/usr/bin/env python3
"""
API 调用示例脚本
在启动 API 服务后运行此脚本进行测试
"""
import requests
import json

API_BASE = "http://localhost:8000"


def import_skills():
    """导入 skills"""
    response = requests.post(f"{API_BASE}/skills/import")
    return response.json()


def update_skills():
    """导入 skills"""
    response = requests.post(f"{API_BASE}/skills/update-all")
    return response.json()



def recommend_skills(query: str):
    """推荐 skills"""
    response = requests.post(
        f"{API_BASE}/recommend",
        json={"query": query}
    )
    response.raise_for_status()
    return response.json()


def get_all_skills():
    """获取所有 skills"""
    response = requests.get(f"{API_BASE}/skills")
    response.raise_for_status()
    return response.json()


def submit_feedback(recommendation_id: str, rating: int, comment: str = None, selected_skill: str = None):
    """提交反馈"""
    data = {
        "recommendation_id": recommendation_id,
        "rating": rating,
    }
    if comment:
        data["comment"] = comment
    if selected_skill:
        data["selected_skill"] = selected_skill
    
    response = requests.post(
        f"{API_BASE}/feedback",
        json=data
    )
    response.raise_for_status()
    return response.json()


def get_feedback_stats():
    """获取反馈统计"""
    response = requests.get(f"{API_BASE}/stats/feedback")
    response.raise_for_status()
    return response.json()


def add_skill(
    skill_id: str,
    name: str,
    description: str,
    category: list = None,
    capabilities: list = None,
    version: str = "1.0.0",
    dependencies: list = None
):
    """添加 skill"""
    data = {
        "id": skill_id,
        "name": name,
        "description": description,
    }
    if category:
        data["category"] = category
    if capabilities:
        data["capabilities"] = capabilities
    if version:
        data["version"] = version
    if dependencies:
        data["dependencies"] = dependencies
    
    response = requests.post(
        f"{API_BASE}/skills",
        json=data
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("=" * 50)
    print("API 调用示例")
    print("=" * 50)


    # 1. 获取所有 skills
    # print("\n[1] 获取所有 skills:")
    # skills = get_all_skills()
    # print(f"  共 {len(skills)} 个 skills")
    # for s in skills[:3]:
    #     print(f"  - {s['name']}: {s['description'][:30]}...")
    #
    #
    # # 2. 添加新的 skill
    # print("\n[2] 添加新的 skill:")
    # new_skill = add_skill(
    #     skill_id="skill_email_sender",
    #     name="Email Sender",
    #     description="发送电子邮件，支持 SMTP 协议",
    #     category=["communication", "email"],
    #     capabilities=["send_email", "smtp"]
    # )
    # print(f"  添加成功: {new_skill}")

    # # 3. 再次获取 skills 验证
    # print("\n[3] 验证添加:")
    # skills = get_all_skills()
    # print(f"  现在共有 {len(skills)} 个 skills")

    # 2. 推荐 skills
    query = ['我要设计一个商标', '分析产品的竞争力和改进方向']
    print(f"\n[2] 推荐 skills (query: {query[1]}):")
    result = recommend_skills(query[1])
    print(f"  找到 {len(result['recommendations'])} 个推荐:")
    for rec in result["recommendations"]:
        print(f"  - {rec['name']} (置信度: {rec['confidence']:.2f})")
        print(f"  - description: {rec['description']}")
    #
    # # 3. 提交反馈
    # print("\n[3] 提交反馈:")
    # if result["recommendations"]:
    #     rec = result["recommendations"][0]
    #     feedback = submit_feedback(
    #         recommendation_id="test_rec",
    #         rating=5,
    #         comment="推荐很准确",
    #         selected_skill=rec["skill_id"]
    #     )
    #     print(f"  反馈提交成功: {feedback}")
    #
    # # 4. 获取反馈统计
    # print("\n[4] 反馈统计:")
    # stats = get_feedback_stats()
    # print(f"  总反馈数: {stats['total']}")
    # print(f"  平均评分: {stats['avg_rating']:.1f}")
    #
    # print("\n" + "=" * 50)
    # print("测试完成!")
    # print("=" * 50)
