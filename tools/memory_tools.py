"""
Memory Tools - 사용자 메모리 읽기/쓰기 도구
"""

import json

from langchain_core.tools import tool

from core.memory import get_memory_manager
from core.utils import merge_preferences


@tool
def read_memory(user_id: str) -> str:
    """
    사용자의 장기 메모리(선호도, 히스토리)를 읽어옵니다.

    Args:
        user_id: 사용자 ID

    Returns:
        사용자 프로필 정보 (JSON 문자열)
        - preferred_categories: 선호 카테고리
        - price_preference: 가격 선호도
        - interests: 관심 분야
        - skill_level: 기술 수준
    """
    memory = get_memory_manager()
    profile = memory.load_user_profile(user_id)

    if profile:
        return json.dumps(profile, ensure_ascii=False, indent=2)
    else:
        return json.dumps(
            {"message": "사용자 프로필이 없습니다.", "user_id": user_id}, ensure_ascii=False
        )


@tool
def write_memory(user_id: str, preferences: str) -> str:
    """
    사용자의 선호도를 장기 메모리에 저장합니다.

    Args:
        user_id: 사용자 ID
        preferences: 저장할 선호도 (JSON 문자열)
            예: {"preferred_categories": ["video-generation"], "price_preference": "무료선호"}

    Returns:
        저장 결과 메시지
    """
    memory = get_memory_manager()

    try:
        prefs = merge_preferences(memory.load_user_profile(user_id), json.loads(preferences))
        success = memory.save_user_profile(user_id, prefs)

        if success:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"사용자 {user_id}의 프로필이 저장되었습니다.",
                    "saved_data": prefs,
                },
                ensure_ascii=False,
            )
        else:
            return json.dumps(
                {"status": "error", "message": "프로필 저장에 실패했습니다."}, ensure_ascii=False
            )

    except json.JSONDecodeError as e:
        return json.dumps(
            {"status": "error", "message": f"JSON 파싱 오류: {str(e)}"}, ensure_ascii=False
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"저장 오류: {str(e)}"}, ensure_ascii=False
        )
