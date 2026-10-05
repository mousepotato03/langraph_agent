"""
Intent Analysis 프롬프트 - Human-in-the-Loop 의도 분석 및 계획 수정
"""

# ==================== 의도 분석 프롬프트 ====================

INTENT_ANALYSIS_SYSTEM_PROMPT = """사용자의 자연어 응답에서 의도를 분석하세요.

## 상황
에이전트가 작업 계획을 제시했고, 사용자가 이에 대해 응답했습니다.

## 의도 분류
1. **approve**: 승인/진행 의도
   - 예: "좋아", "진행해", "OK", "네", "ㅇㅇ", "그래", "시작해", "해줘", "고마워 진행해"

2. **modify**: 수정 요청 의도
   - 예: "그건 빼줘", "이것만 해줘", "순서 바꿔줘", "무료만", "가격 저렴한 걸로"

3. **cancel**: 취소 의도
   - 예: "취소", "됐어", "그만", "안할래", "아니"

4. **clarify**: 의도가 명확하지 않거나 계획에 대한 질문인 경우
   - 예: "이 단계는 왜 필요해?", "음...", "잘 모르겠어"
   - 명확한 승인 표현이 없으면 approve로 판단하지 마세요.

## 응답 형식 (JSON만 출력)
{"intent": "approve", "feedback": ""}
또는
{"intent": "modify", "feedback": "수정 요청 상세 내용"}
또는
{"intent": "cancel", "feedback": ""}
또는
{"intent": "clarify", "feedback": ""}
"""

INTENT_ANALYSIS_USER_TEMPLATE = """## 현재 계획
{plan_summary}

## 사용자 응답
{user_response}

위 응답의 의도를 JSON으로 분석하세요."""


# ==================== 계획 수정 프롬프트 ====================

MODIFY_PLAN_SYSTEM_PROMPT = """현재 계획을 사용자의 피드백에 맞게 수정하세요.

## 규칙
1. 피드백에 따라 작업을 제거, 수정, 또는 추가
2. 기존 형식 유지 (id, description, category, priority, status)
3. 제거된 작업은 완전히 삭제
4. priority는 순서대로 재정렬

## 응답 형식 (JSON 배열만)
[{"id": "task_1", "description": "...", "category": "...", "priority": 1, "status": "pending"}, ...]
"""

MODIFY_PLAN_USER_TEMPLATE = """## 현재 계획
{current_plan}

## 사용자 수정 요청
{feedback}

위 피드백을 반영하여 수정된 계획을 JSON 배열로 출력하세요."""


# ==================== Human Review 메시지 ====================

HUMAN_REVIEW_MESSAGE = """## 📋 작업 계획 검토

다음과 같은 작업 계획을 수립했습니다:

{plan_summary}

### 진행하시겠습니까?

- ✅ **승인**: 이대로 진행합니다
- ✏️ **수정**: 계획을 수정하고 싶습니다
- ❌ **취소**: 작업을 취소합니다

아래에서 선택해주세요."""
