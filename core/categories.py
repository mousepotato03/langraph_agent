"""Canonical tool categories and explicit aliases used by the supplied catalog."""

CATEGORY_ALIASES = {
    "text-generation": {"writing-assistant", "language-model", "nlp", "translation"},
    "image-generation": {"image-generation"},
    "video-generation": {"video-ai", "video-generation", "video-editing"},
    "audio-generation": {"transcription", "meeting-assistant"},
    "code-generation": {"coding-assistant", "code-editor", "developer-tools"},
    "productivity": {
        "productivity",
        "ai-assistant",
        "automation",
        "workflow-automation",
        "work-management",
        "project-management",
        "note-taking",
    },
    "design": {"design", "ui-design", "web-design", "prototyping", "image-editing"},
    "research": {"research", "research-assistant", "knowledge-work", "knowledge-management"},
}


def matches_category(categories: str, requested: str) -> bool:
    def normalize(value):
        return "-".join(value.strip().lower().split())

    requested = normalize(requested)
    accepted = {requested, *CATEGORY_ALIASES.get(requested, set())}
    return bool(accepted & {normalize(part) for part in categories.split(",")})
