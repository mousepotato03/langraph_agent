"""Environment configuration; default paths do not depend on the working directory."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MAX_TOOL_CALLS_PER_TASK = 3
MAX_SUBTASKS = 5
SIMILARITY_THRESHOLD = 0.4
# Upper bound: router + planning + approval + 5 * (3 tool pairs + finalization)
# + guide + reflection, with room for a modified plan before re-approval.
GRAPH_RECURSION_LIMIT = 80
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", "100"))
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "db"))
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data"))
TOOLS_JSON_PATH = str(Path(DATA_PATH) / "ai_tools.json")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "7860"))
