"""
Time Tools - 시간 및 최신성 확인 도구
"""

from datetime import datetime

from langchain_core.tools import tool


@tool
def check_tool_freshness(tool_name: str, updated_date: str) -> str:
    """
    AI 도구 정보의 최신성을 확인합니다.

    Args:
        tool_name: 도구 이름
        updated_date: 도구 정보 업데이트 날짜 (YYYY-MM-DD 형식)

    Returns:
        최신 여부 판단 결과
    """
    try:
        today = datetime.now()
        update_dt = datetime.strptime(updated_date, "%Y-%m-%d")
        days_old = (today - update_dt).days

        if days_old <= 30:
            return f"✅ '{tool_name}' 정보는 최신입니다. ({days_old}일 전 업데이트)"
        elif days_old <= 90:
            return f"⚠️ '{tool_name}' 정보가 다소 오래되었습니다. ({days_old}일 전 업데이트) 공식 사이트에서 최신 정보를 확인하세요."
        else:
            return f"❌ '{tool_name}' 정보가 오래되었습니다. ({days_old}일 전 업데이트) 반드시 공식 사이트에서 확인하세요."

    except ValueError:
        return f"⚠️ '{tool_name}'의 업데이트 날짜 형식이 올바르지 않습니다."


@tool
def get_current_time() -> str:
    """
    현재 날짜와 시간을 반환합니다.
    도구 정보의 최신성을 확인하거나 시간 관련 정보가 필요할 때 사용하세요.

    Returns:
        현재 날짜/시간 문자열 (YYYY-MM-DD HH:MM:SS 형식)
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def check_freshness_simple(updated_date: str) -> dict:
    """
    최신 여부를 간단히 확인 (내부 함수)

    Returns:
        {"is_fresh": bool, "days_old": int, "message": str}
    """
    try:
        today = datetime.now()
        update_dt = datetime.strptime(updated_date, "%Y-%m-%d")
        days_old = (today - update_dt).days

        return {
            "is_fresh": days_old <= 30,
            "days_old": days_old,
            "message": "최신" if days_old <= 30 else "확인 필요",
        }
    except ValueError:
        return {"is_fresh": False, "days_old": -1, "message": "날짜 형식 오류"}
