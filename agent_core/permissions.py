"""Permission manager for tool calling - hardware-related tools need approval."""
import sys
from typing import Set

# Tool classification by hardware risk
FS_READ_TOOLS: Set[str] = {"read", "glob", "grep", "skill_list", "skill_load"}
FS_WRITE_TOOLS: Set[str] = {"write", "edit"}
EXEC_TOOLS: Set[str] = {"bash"}

ALL_FS_TOOLS = FS_READ_TOOLS | FS_WRITE_TOOLS | EXEC_TOOLS
# All that need permission (except skill_list/load which are safe read-only? still included in ask for consistency, but can be auto)
# Definition: hardware = filesystem read/change + run command
HARDWARE_TOOLS = {"read", "write", "edit", "glob", "grep", "bash"}

VALID_MODES = {"accept-all", "accept-fs", "ask"}

class PermissionManager:
    def __init__(self, mode: str = "ask"):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode}")
        self.mode = mode

    def set_mode(self, mode: str):
        mode = mode.strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"invalid mode: {mode}. Choices: {', '.join(VALID_MODES)}")
        self.mode = mode

    def is_auto_allowed(self, tool_name: str) -> bool:
        """Check whether tool is auto-allowed without prompt based on mode."""
        name = tool_name.strip().lower()
        # skill and sub-agent tools always auto (not directly dangerous, sub-agents handle their own permissions)
        if name in {"skill_list", "skill_load", "spawn_agents"}:
            return True
        if self.mode == "accept-all":
            return True
        if self.mode == "accept-fs":
            # fs: read/write/glob/grep auto, bash still asks
            if name in FS_READ_TOOLS and name not in EXEC_TOOLS:
                return True
            if name in FS_WRITE_TOOLS:
                return True
            # bash still needs ask
            return False
        # ask -> all hardware needs prompt
        return False

    def prompt(self, tool_name: str, args_str: str) -> bool:
        """Prompt user for permission. Return True if allowed."""
        if self.is_auto_allowed(tool_name):
            return True
        # Show prompt to stderr so it doesn't interfere with stdout streaming
        # Create human-readable preview without cutting mid-JSON
        try:
            import json as _json
            parsed = _json.loads(args_str) if args_str.strip().startswith("{") else {}
            if isinstance(parsed, dict):
                # Show key fields like filePath, command, pattern
                keys = ["filePath", "filepath", "path", "command", "pattern", "content"]
                parts = []
                for k in keys:
                    if k in parsed and parsed[k]:
                        v = str(parsed[k])[:100].replace("\n", " ")
                        parts.append(f"{k}={v}")
                if parts:
                    preview = ", ".join(parts)
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                else:
                    preview = args_str[:300].replace("\n", " ")
                    if len(args_str) > 300:
                        preview += "..."
            else:
                preview = args_str[:300].replace("\n", " ")
                if len(args_str) > 300:
                    preview += "..."
        except:
            preview = args_str[:300].replace("\n", " ")
            if len(args_str) > 300:
                preview += "..."
        print(f"\n[permission] Tool '{tool_name}' wants to run", file=sys.stderr)
        print(f"  args: {preview}", file=sys.stderr)
        print(f"  mode: {self.mode} | allow? (y=yes, n=no, a=always-accept-all, f=accept-fs) [y/n/a/f]: ", end="", file=sys.stderr, flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[permission denied - interrupt]", file=sys.stderr)
            return False
        if ans in ("y", "yes", "1"):
            return True
        if ans in ("a", "always", "accept-all"):
            self.mode = "accept-all"
            print("[permission] mode -> accept-all (all tools auto)", file=sys.stderr)
            return True
        if ans in ("f", "fs", "accept-fs"):
            self.mode = "accept-fs"
            print("[permission] mode -> accept-fs (filesystem auto)", file=sys.stderr)
            return True
        # n, no, or empty -> deny
        print("[permission denied]", file=sys.stderr)
        return False

    def check_or_prompt(self, tool_name: str, args_str: str) -> bool:
        """Shortcut: if auto, return True directly, else prompt."""
        if self.is_auto_allowed(tool_name):
            return True
        return self.prompt(tool_name, args_str)

    def status(self) -> str:
        desc = {
            "accept-all": "all tools automatically allowed",
            "accept-fs": "filesystem (read/write/glob/grep) auto, bash/run command still asks",
            "ask": "all hardware (filesystem + bash) must be confirmed"
        }
        return f"{self.mode} ({desc.get(self.mode, '')})"
