"""
Memory Manager - ChromaDB 기반 벡터 저장소 및 사용자 프로필 관리
"""

import glob
import json
import logging
import os
from datetime import datetime
from threading import Lock

from langchain_core.messages import HumanMessage, SystemMessage

from core.categories import matches_category
from core.config import DB_PATH, EMBEDDING_MODEL
from core.llm import get_llm
from core.utils import extract_json, merge_preferences
from prompts.reflection import MEMORY_EXTRACTOR_SYSTEM_PROMPT, MEMORY_EXTRACTOR_USER_TEMPLATE

logger = logging.getLogger(__name__)


class MemoryManager:
    """ChromaDB를 활용한 RAG 및 장기 메모리 관리 클래스"""

    def __init__(self, persist_dir: str = None, *, client=None, embedding_model=None):
        """
        Args:
            persist_dir: ChromaDB 영구 저장소 경로
        """
        self.persist_dir = persist_dir or DB_PATH

        # ChromaDB 클라이언트 초기화 (Persistent)
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_dir)
        self.client = client

        # 임베딩 모델 초기화 (다국어 지원)
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_model = embedding_model

        # 컬렉션 초기화 (cosine distance 사용으로 유사도 계산 개선)
        self.tools_collection = self.client.get_or_create_collection(
            name="ai_tools",
            metadata={"description": "AI tools knowledge base", "hnsw:space": "cosine"},
        )

        self.profile_collection = self.client.get_or_create_collection(
            name="user_profile", metadata={"description": "User preferences and history"}
        )

        # PDF 지식베이스 컬렉션 (cosine distance 사용)
        self.pdf_collection = self.client.get_or_create_collection(
            name="pdf_knowledge",
            metadata={"description": "PDF documents knowledge base", "hnsw:space": "cosine"},
        )

    def _embed_text(self, text: str) -> list[float]:
        """텍스트를 임베딩 벡터로 변환"""
        return self.embedding_model.encode(text).tolist()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 임베딩 벡터로 변환"""
        return self.embedding_model.encode(texts).tolist()

    # ==================== AI Tools 관련 메서드 ====================

    def load_tools_from_json(self, json_path: str) -> int:
        """
        JSON 파일에서 AI 도구 데이터를 로드하여 ChromaDB에 저장

        Args:
            json_path: tools_base.json 파일 경로

        Returns:
            저장된 도구 수
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        tools = data.get("tools", [])
        if not tools:
            return 0

        # 기존 데이터 확인 (중복 방지)
        existing_count = self.tools_collection.count()
        if existing_count > 0:
            logger.info("Reusing %s catalog tools", existing_count)
            return existing_count

        # 문서 준비
        documents = []
        metadatas = []
        ids = []

        for idx, tool in enumerate(tools):
            # 새 구조: categories, domains, scores
            categories = tool.get("categories", [])
            domains = tool.get("domains", [])
            scores = tool.get("scores", {})

            categories_text = ", ".join(categories)
            domains_text = ", ".join(domains)
            pricing_notes = scores.get("pricing_notes", "")
            pricing_model = scores.get("pricing_model", "")

            # 검색 키워드와 매칭이 잘 되도록 다양한 표현 포함
            doc_text = f"""{tool["name"]} - {categories_text}
{tool["description"]}
카테고리: {categories_text}
적용 분야: {domains_text}
가격: {pricing_notes}
{tool["name"]}은 {categories_text} 분야의 AI 도구입니다.""".strip()

            documents.append(doc_text)

            # 메타데이터
            metadatas.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "categories": categories_text,
                    "domains": domains_text,
                    "pricing_model": pricing_model,
                    "pricing_notes": pricing_notes,
                    "scores": json.dumps(scores, ensure_ascii=False),
                }
            )

            # ID
            ids.append(f"tool_{idx}_{tool['name'].lower().replace(' ', '_')}")

        # 임베딩 생성 및 저장
        embeddings = self._embed_texts(documents)

        self.tools_collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        logger.info("Indexed %s catalog tools", len(tools))
        return len(tools)

    def search_tools(
        self, query: str, k: int = 5, threshold: float = 0.4, category: str | None = None
    ) -> tuple[list[dict], bool]:
        """
        AI 도구 검색 (RAG)

        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            threshold: 유사도 임계값 (0.7)
            category: 카테고리 필터 (선택)

        Returns:
            (검색 결과 리스트, fallback 필요 여부)
        """
        # 도구 데이터가 없으면 빈 리스트 반환 (HNSW 오류 방지)
        count = self.tools_collection.count()
        if count == 0:
            return [], True
        if k < 1:
            raise ValueError("k must be positive")

        # 쿼리 임베딩
        query_embedding = self._embed_text(query)

        # Chroma metadata cannot apply $contains to our comma-separated string.
        # Exact token matching also supports databases built by the original app.
        where_filter = None
        if category:
            metadata = self.tools_collection.get(include=["metadatas"])["metadatas"]
            matches = [
                item["name"]
                for item in metadata
                if matches_category(item.get("categories", ""), category)
            ]
            if not matches:
                return [], True
            where_filter = {"name": {"$in": matches}}
            count = len(matches)

        # ChromaDB 검색
        results = self.tools_collection.query(
            query_embeddings=[query_embedding], n_results=min(k, count), where=where_filter
        )

        # 결과 처리
        search_results = []

        if results["documents"] and results["documents"][0]:
            for idx, _doc in enumerate(results["documents"][0]):
                # 거리 → 유사도 변환 (ChromaDB cosine distance 사용)
                distance = results["distances"][0][idx] if results["distances"] else 1.0
                # cosine distance를 유사도로 변환: similarity = 1 - distance
                # cosine distance 범위: 0 (동일) ~ 2 (반대), 일반적으로 0~1
                similarity = max(0, 1 - distance)

                metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                try:
                    catalog_scores = json.loads(metadata.get("scores", "{}"))
                except (ValueError, TypeError):
                    catalog_scores = {}
                if not isinstance(catalog_scores, dict):
                    catalog_scores = {}

                search_results.append(
                    {
                        "name": metadata.get("name", "Unknown"),
                        "description": metadata.get("description", ""),
                        "categories": metadata.get("categories", ""),
                        "domains": metadata.get("domains", ""),
                        "pricing_model": metadata.get("pricing_model", ""),
                        "pricing_notes": metadata.get("pricing_notes", ""),
                        "source_urls": catalog_scores.get("source_urls", []),
                        "catalog_updated_at": catalog_scores.get("last_updated", ""),
                        "scores": metadata.get("scores", "{}"),
                        "score": round(similarity, 3),
                    }
                )

        # Fallback 필요 여부 판단
        if not search_results:
            should_fallback = True
        else:
            top_score = max(r["score"] for r in search_results)
            avg_score = sum(r["score"] for r in search_results) / len(search_results)
            should_fallback = top_score < threshold or avg_score < 0.5

        return search_results, should_fallback

    def get_tool_by_name(self, name: str) -> dict | None:
        """도구 이름으로 상세 정보 조회"""
        results = self.tools_collection.get(where={"name": name}, limit=1)

        if results["metadatas"]:
            return results["metadatas"][0]
        return None

    # ==================== 사용자 프로필 관련 메서드 ====================

    def load_user_profile(self, user_id: str) -> dict | None:
        """
        장기 메모리에서 사용자 프로필 로드

        Args:
            user_id: 사용자 ID

        Returns:
            사용자 프로필 딕셔너리 또는 None
        """
        try:
            results = self.profile_collection.get(
                ids=[f"profile_{user_id}"], include=["documents", "metadatas"]
            )

            if results["documents"] and results["documents"][0]:
                profile_json = results["documents"][0]
                profile = merge_preferences(None, json.loads(profile_json))
                return profile
        except Exception as e:
            logger.warning("Profile load failed: %s", e)

        return None

    def save_user_profile(self, user_id: str, preferences: dict) -> bool:
        """
        사용자 프로필을 ChromaDB에 저장 (Upsert)

        Args:
            user_id: 사용자 ID
            preferences: 사용자 선호도 딕셔너리

        Returns:
            저장 성공 여부
        """
        try:
            preferences = merge_preferences(None, preferences)
            profile_id = f"profile_{user_id}"
            profile_json = json.dumps(preferences, ensure_ascii=False)

            # 프로필 텍스트를 임베딩 (향후 유사 사용자 검색용)
            profile_text = f"""
            선호 카테고리: {", ".join(preferences.get("preferred_categories", []))}
            선호 가격대: {preferences.get("price_preference", "")}
            관심사: {", ".join(preferences.get("interests", []))}
            기술 수준: {preferences.get("skill_level", "")}
            """.strip()

            embedding = self._embed_text(profile_text)

            # Upsert (있으면 업데이트, 없으면 추가)
            self.profile_collection.upsert(
                ids=[profile_id],
                documents=[profile_json],
                embeddings=[embedding],
                metadatas=[{"user_id": user_id, "updated_at": datetime.now().isoformat()}],
            )

            return True
        except Exception as e:
            logger.warning("Profile storage failed: %s", e)
            return False

    def extract_preferences(
        self, messages: list[dict[str, str]], existing_profile: dict | None = None
    ) -> dict:
        """Compatibility helper uses the same prompt and validation as Reflection."""
        conversation = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
        )
        try:
            response = get_llm(temperature=0.3).invoke(
                [
                    SystemMessage(content=MEMORY_EXTRACTOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content=MEMORY_EXTRACTOR_USER_TEMPLATE.format(
                            conversation=conversation,
                            existing_profile=json.dumps(existing_profile, ensure_ascii=False),
                        )
                    ),
                ]
            )
            return merge_preferences(existing_profile, json.loads(extract_json(response.content)))
        except Exception:
            logger.warning("Could not extract preferences", exc_info=True)
            return existing_profile or {}

    def get_tools_count(self) -> int:
        """저장된 AI 도구 수 반환"""
        return self.tools_collection.count()

    def get_profiles_count(self) -> int:
        """저장된 사용자 프로필 수 반환"""
        return self.profile_collection.count()

    # ==================== PDF 지식베이스 관련 메서드 ====================

    def load_pdfs_from_directory(self, pdf_dir: str) -> int:
        """
        디렉토리 내 모든 PDF 파일을 로드하여 ChromaDB에 저장

        Args:
            pdf_dir: PDF 파일이 있는 디렉토리 경로

        Returns:
            저장된 청크 수
        """
        # 중복 방지 체크
        existing_count = self.pdf_collection.count()
        if existing_count > 0:
            logger.info("Reusing %s PDF chunks", existing_count)
            return existing_count

        # PDF 파일 탐색
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in %s", pdf_dir)
            return 0

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pypdf import PdfReader

        # 텍스트 스플리터 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        )

        documents = []
        metadatas = []
        ids = []

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            try:
                pages = PdfReader(pdf_path).pages

                for page_num, page in enumerate(pages):
                    chunks = text_splitter.split_text(page.extract_text() or "")

                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk.strip():
                            documents.append(chunk)
                            metadatas.append(
                                {
                                    "source": "pdf",
                                    "filename": filename,
                                    "page": page_num,
                                    "chunk_idx": chunk_idx,
                                }
                            )
                            ids.append(f"pdf_{filename}_{page_num}_{chunk_idx}")

                logger.info("Loaded %s (%s pages)", filename, len(pages))

            except Exception as e:
                logger.warning("PDF load failed (%s): %s", filename, e)
                continue

        if not documents:
            return 0

        # 임베딩 생성 및 저장
        embeddings = self._embed_texts(documents)

        self.pdf_collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        logger.info("Indexed %s PDF chunks", len(documents))
        return len(documents)

    def search_pdf_knowledge(self, query: str, k: int = 3, threshold: float = 0.03) -> list[dict]:
        """
        PDF 지식베이스 검색

        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            threshold: 유사도 임계값

        Returns:
            검색 결과 리스트
        """
        # PDF 데이터가 없으면 빈 리스트 반환
        count = self.pdf_collection.count()
        if count == 0:
            return []

        query_embedding = self._embed_text(query)

        results = self.pdf_collection.query(
            query_embeddings=[query_embedding], n_results=min(k, count)
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for idx, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][idx] if results["distances"] else 1.0
                # cosine distance를 유사도로 변환: similarity = 1 - distance
                similarity = max(0, 1 - distance)

                if similarity >= threshold:
                    metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                    search_results.append(
                        {
                            "content": doc,
                            "source": "pdf",
                            "filename": metadata.get("filename", "Unknown"),
                            "page": metadata.get("page", 0),
                            "score": round(similarity, 3),
                        }
                    )

        return search_results

    def get_pdf_count(self) -> int:
        """저장된 PDF 청크 수 반환"""
        return self.pdf_collection.count()

    def search_pdf_for_tool(self, tool_name: str, categories: str, k: int = 3) -> float:
        """
        특정 도구에 대한 PDF 관련도 점수 계산

        도구명과 카테고리를 조합한 쿼리로 PDF를 검색하여
        해당 도구가 최신 트렌드에서 얼마나 언급되는지 점수화

        Args:
            tool_name: 도구 이름
            categories: 도구 카테고리 (쉼표 구분 문자열)
            k: 검색할 PDF 청크 수

        Returns:
            관련도 점수 (0~1)
        """
        # PDF 데이터가 없으면 0 반환
        count = self.pdf_collection.count()
        if count == 0:
            return 0.0

        # 도구명 + 카테고리로 검색 쿼리 구성
        query = f"{tool_name} {categories}"
        query_embedding = self._embed_text(query)

        results = self.pdf_collection.query(
            query_embeddings=[query_embedding], n_results=min(k, count)
        )

        if not results["documents"] or not results["documents"][0]:
            return 0.0

        # 검색 결과의 유사도 점수 평균 계산
        scores = []
        for idx, _doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][idx] if results["distances"] else 1.0
            similarity = max(0, 1 - distance)
            scores.append(similarity)

        # 평균 점수 반환 (0~1 범위)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return round(avg_score, 3)


# 메모리 매니저 싱글톤
_memory_manager = None
_memory_lock = Lock()


def get_memory_manager() -> MemoryManager:
    """메모리 매니저 싱글톤 반환"""
    global _memory_manager
    if _memory_manager is None:
        with _memory_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager()
    return _memory_manager
