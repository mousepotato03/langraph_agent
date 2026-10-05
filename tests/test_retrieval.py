import copy
import json
from types import SimpleNamespace

import chromadb
import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from core.memory import MemoryManager
from tools import search


class LocalEmbedding:
    def encode(self, value):
        def vector(text):
            return [1.0, 0.0, 0.0] if "video" in text.lower() else [0.0, 1.0, 0.0]

        return np.array(
            [vector(text) for text in value] if isinstance(value, list) else vector(value)
        )


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(
        client=chromadb.PersistentClient(path=str(tmp_path / "chroma")),
        embedding_model=LocalEmbedding(),
    )


def test_real_chroma_category_filter_matches_exact_tokens_and_bounds_result_count(memory, tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "Video",
                        "description": "video creation",
                        "categories": ["Video AI"],
                        "scores": {
                            "last_updated": "2025-08-21",
                            "source_urls": ["https://example.org"],
                        },
                    },
                    {
                        "name": "Writer",
                        "description": "writing",
                        "categories": ["Writing Assistant"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert memory.load_tools_from_json(str(catalog)) == 2
    results, fallback = memory.search_tools("video", k=5, category="video-generation")
    assert [item["name"] for item in results] == ["Video"]
    assert not fallback
    assert results[0]["catalog_updated_at"] == "2025-08-21"
    assert results[0]["source_urls"] == ["https://example.org"]
    assert memory.search_tools("writing", category="text-generation")[0][0]["name"] == "Writer"
    assert memory.search_tools("video", category="video") == ([], True)
    assert memory.search_tools("video", category="unknown") == ([], True)


def test_real_chroma_profile_is_persistent_and_scoped_by_user_id(memory):
    assert memory.save_user_profile("alice", {"interests": ["영상"], "notes": "무료 선호"})
    assert memory.load_user_profile("alice")["interests"] == ["영상"]
    assert memory.load_user_profile("bob") is None


def test_invalid_profile_does_not_overwrite_the_previous_preferences(memory):
    assert memory.save_user_profile("alice", {"interests": ["영상"]})
    assert not memory.save_user_profile("alice", {"interests": [{"invalid": "value"}]})
    assert memory.load_user_profile("alice")["interests"] == ["영상"]


def make_pdf(path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (video research and tool trends) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def test_real_pdf_loader_indexes_source_page_and_can_be_reloaded_without_duplicates(
    memory, tmp_path
):
    make_pdf(tmp_path / "research.pdf")
    assert memory.load_pdfs_from_directory(str(tmp_path)) == 1
    assert memory.load_pdfs_from_directory(str(tmp_path)) == 1
    results = memory.search_pdf_knowledge("video", k=3)
    assert len(results) == 1
    assert results[0]["filename"] == "research.pdf"
    assert results[0]["page"] == 0
    assert "video research" in results[0]["content"]
    assert memory.search_pdf_for_tool("video", "video-generation") == pytest.approx(1)


def candidates_memory(fallback=False):
    rows = [
        {"name": "A", "categories": "video-generation", "score": 0.8},
        {"name": "B", "categories": "video-generation", "score": 0.7},
    ]
    return SimpleNamespace(
        rows=rows,
        search_tools=lambda **kwargs: (rows, fallback),
        search_pdf_for_tool=lambda tool_name, **kwargs: 0.0 if tool_name == "A" else 1.0,
    )


def test_two_stage_ranking_preserves_catalog_records_and_uses_pdf_weight():
    memory = candidates_memory()
    before = copy.deepcopy(memory.rows)
    top, candidates, fallback = search.two_stage_search(memory, "video", use_web_fallback=False)
    assert top["name"] == "B"
    assert top["scores"]["final_score"] == pytest.approx(0.79)
    assert [item["name"] for item in candidates] == ["B", "A"]
    assert not fallback
    assert memory.rows == before


def test_weak_catalog_results_use_web_when_fallback_is_enabled(monkeypatch):
    searches = []

    def web(query, **kwargs):
        searches.append(query)
        return [{"name": "Web", "description": "web result", "url": "https://example.org"}]

    monkeypatch.setattr(search, "google_search", SimpleNamespace(is_available=True, search=web))
    top, _, fallback = search.two_stage_search(candidates_memory(True), "video")
    assert fallback
    assert searches == ["video"]
    assert top["name"] == "Web"


def test_hybrid_search_honors_include_pdf_false():
    results, _ = search.hybrid_search(
        candidates_memory(), "video", include_pdf=False, use_web_fallback=False
    )
    assert results[0]["name"] == "A"
    assert results[0]["scores"]["pdf_score"] == 0
    assert results[0]["scores"]["final_score"] == pytest.approx(0.8)
