"""Config - single source of truth for agent."""
import os

OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai")
OPENCODE_UA = "opencode"

# Default model - free & supports tool calling
DEFAULT_MODEL = os.environ.get("OPENCODE_MODEL", "mimo-v2.5-free")

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

# Paths
SKILLS_DIR = ".agent/skills"
HISTS_DIR = ".agent/hists"
PROVIDER_STATE_FILE = ".agent/llm_provider_state.json"

# Safety: max file read bytes
MAX_READ_BYTES = 1024 * 500  # 500KB per read

# Fallback provider URLs (used if SDK differs)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
