import os

# Set before importing UI/storage modules. Test doubles never call a hosted model.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# These tests use in-process ASGI transports. Do not inherit a developer proxy.
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)
