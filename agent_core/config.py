"""Config - single source of truth for agent."""
import os

OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai")
OPENCODE_UA = "opencode"

# Default model - free, supports tool calling and vision (Response API for binary files)
DEFAULT_MODEL = os.environ.get("OPENCODE_MODEL", "muse-spark-1.2-contributor-free")

RESPONSES_MODELS = {
    "muse-spark-1.2-contributor-free",
    "muse-spark-1.3-contributor-free",
}

# Context manager limits
MAX_CONTEXT_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "120000"))
COMPACTION_THRESHOLD = 0.85  # compact when >85% limit
KEEP_RECENT_MESSAGES = 16
MAX_TOOL_OUTPUT_CHARS = 20000  # truncate tool output > this
SUMMARY_TRIGGER_TOKENS = 80000

# Agent loop
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "30"))
STREAM = True

# Paths - absolute based on project root (where this file lives is agent_core/, so parent is project root)
import pathlib as _pathlib
_PROJECT_ROOT = _pathlib.Path(__file__).parent.parent
SKILLS_DIR = str(_PROJECT_ROOT / ".agent" / "skills")
HISTS_DIR = str(_PROJECT_ROOT / ".agent" / "hists")
PROVIDER_STATE_FILE = str(_PROJECT_ROOT / ".agent" / "llm_provider_state.json")

# Safety: max file read bytes
MAX_READ_BYTES = 1024 * 500  # 500KB per read

# Fallback provider URLs (used if SDK differs)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
