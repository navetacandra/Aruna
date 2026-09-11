"""Permission manager untuk tool calling - hardware related tools perlu approval."""
import sys
from typing import Set

# Klasifikasi tool berdasarkan risiko hardware
FS_READ_TOOLS: Set[str] = {"read", "glob", "grep", "skill_list", "skill_load"}
FS_WRITE_TOOLS: Set[str] = {"write", "edit"}
EXEC_TOOLS: Set[str] = {"bash"}

ALL_FS_TOOLS = FS_READ_TOOLS | FS_WRITE_TOOLS | EXEC_TOOLS
# Semua yang perlu permission (kecuali skill_list/load yang read-only aman? tetap masuk ask untuk konsistensi, tapi bisa auto)
# Definisi: hardware = filesystem read/change + run command
HARDWARE_TOOLS = {"read", "write", "edit", "glob", "grep", "bash"}

VALID_MODES = {"accept-all", "accept-fs", "ask"}

class PermissionManager:
    def __init__(self, mode: str = "ask"):
        if mode not in VALID_MODES:
            raise ValueError(f"mode harus salah satu {VALID_MODES}, got {mode}")
        self.mode = mode

    def set_mode(self, mode: str):
        mode = mode.strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"mode tidak valid: {mode}. Pilihan: {', '.join(VALID_MODES)}")
        self.mode = mode

    def is_auto_allowed(self, tool_name: str) -> bool:
        """Cek apakah tool boleh auto tanpa prompt berdasarkan mode."""
        name = tool_name.strip()
        # skill tools selalu auto (tidak berbahaya, hanya baca .agent/skills)
        if name in {"skill_list", "skill_load"}:
            return True
        if self.mode == "accept-all":
            return True
        if self.mode == "accept-fs":
            # fs: read/write/glob/grep auto, bash tetap ask
            if name in FS_READ_TOOLS and name not in EXEC_TOOLS:
                return True
            if name in FS_WRITE_TOOLS:
                return True
            # bash tetap perlu ask
            return False
        # ask -> semua hardware perlu prompt
        return False

    def prompt(self, tool_name: str, args_str: str) -> bool:
        """Prompt user untuk izin. Return True jika diizinkan."""
        if self.is_auto_allowed(tool_name):
            return True
        # Tampilkan prompt ke stderr agar tidak ganggu stdout streaming
        # Format ringkas args
        preview = args_str[:300].replace("\n", " ")
        if len(args_str) > 300:
            preview += "..."
        print(f"\n[permission] Tool '{tool_name}' ingin dijalankan", file=sys.stderr)
        print(f"  args: {preview}", file=sys.stderr)
        print(f"  mode: {self.mode} | izinkan? (y=yes, n=no, a=always-accept-all, f=accept-fs) [y/n/a/f]: ", end="", file=sys.stderr, flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[permission denied - interrupt]", file=sys.stderr)
            return False
        if ans in ("y", "yes", "1"):
            return True
        if ans in ("a", "always", "accept-all"):
            self.mode = "accept-all"
            print("[permission] mode -> accept-all (semua tool auto)", file=sys.stderr)
            return True
        if ans in ("f", "fs", "accept-fs"):
            self.mode = "accept-fs"
            print("[permission] mode -> accept-fs (filesystem auto)", file=sys.stderr)
            return True
        # n, no, atau kosong -> deny
        print("[permission denied]", file=sys.stderr)
        return False

    def check_or_prompt(self, tool_name: str, args_str: str) -> bool:
        """Shortcut: jika auto, langsung True, else prompt."""
        if self.is_auto_allowed(tool_name):
            return True
        return self.prompt(tool_name, args_str)

    def status(self) -> str:
        desc = {
            "accept-all": "semua tool otomatis diizinkan",
            "accept-fs": "filesystem (read/write/glob/grep) auto, bash/run command tetap tanya",
            "ask": "semua hardware (filesystem + bash) harus konfirmasi"
        }
        return f"{self.mode} ({desc.get(self.mode, '')})"
