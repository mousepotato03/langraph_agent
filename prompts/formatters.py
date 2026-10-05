"""
Formatters - 프롬프트용 포맷팅 유틸리티 함수
"""


def format_search_results(results: list[dict]) -> str:
    """검색 결과를 프롬프트용 문자열로 포맷팅 (출처 표시 포함)"""
    if not results:
        return "검색 결과 없음"

    formatted = []
    for idx, r in enumerate(results, 1):
        source = r.get("source", "json")

        if source == "pdf":
            # PDF 결과 포맷
            content = r.get("content", "N/A")
            # 내용이 길면 500자로 제한
            if len(content) > 500:
                content = content[:500] + "..."

            formatted.append(
                f"""
### {idx}. [PDF 참고자료] {r.get("filename", "Unknown")}
- **출처**: PDF 문서 (페이지 {r.get("page", "N/A")})
- **내용**: {content}
- **유사도 점수**: {r.get("score", 0):.2f}
""".strip()
            )
        else:
            # JSON 도구 또는 웹 검색 결과
            source_label = "[AI 도구]" if source == "json" else "[웹 검색]"
            formatted.append(
                f"""
### {idx}. {source_label} {r.get("name", "Unknown")}
- **카테고리**: {r.get("category", "N/A")}
- **설명**: {r.get("description", "N/A")}
- **가격**: {r.get("pricing", "N/A")}
- **유사도 점수**: {r.get("score", 0):.2f}
- **URL**: {r.get("url", "N/A")}
""".strip()
            )

    return "\n\n".join(formatted)


def format_plan_summary(subtasks: list[dict]) -> str:
    """계획을 요약 문자열로 포맷팅"""
    if not subtasks:
        return "계획 없음"

    lines = []
    for task in subtasks:
        if isinstance(task, dict):
            lines.append(
                f"- **{task.get('id', '')}**: {task.get('description', '')} ({task.get('category', '')})"
            )
        else:
            # 문자열인 경우
            lines.append(f"- {task}")

    return "\n".join(lines)


def format_user_profile(profile: dict | None) -> str:
    """사용자 프로필을 문자열로 포맷팅"""
    if not profile:
        return "신규 사용자 (프로필 없음)"

    lines = []
    if profile.get("preferred_categories"):
        lines.append(f"- 선호 카테고리: {', '.join(profile['preferred_categories'])}")
    if profile.get("price_preference"):
        lines.append(f"- 가격 선호도: {profile['price_preference']}")
    if profile.get("interests"):
        lines.append(f"- 관심 분야: {', '.join(profile['interests'])}")
    if profile.get("skill_level"):
        lines.append(f"- 기술 수준: {profile['skill_level']}")

    return "\n".join(lines) if lines else "프로필 정보 없음"


def format_guides(guides: dict[str, str]) -> str:
    """각 작업별 가이드를 통합 문자열로 포맷팅"""
    if not guides:
        return "가이드 없음"

    formatted = []
    for task_id, guide in guides.items():
        formatted.append(f"### {task_id}\n{guide}")

    return "\n\n---\n\n".join(formatted)
