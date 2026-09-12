"""Tool definitions & implementations - robust filesystem, no external dependencies."""
import fnmatch
import glob as globmod
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List

from .config import MAX_READ_BYTES, MAX_TOOL_OUTPUT_CHARS

# ---------- Tool Schemas (OpenAI compatible) ----------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents. Use for inspecting code, logs, config. Supports offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Absolute or relative path to file"},
                    "offset": {"type": "integer", "description": "Starting line (1-indexed)", "default": 1},
                    "limit": {"type": "integer", "description": "Maximum number of lines, default 2000"}
                },
                "required": ["filePath"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a new file or overwrite. ALWAYS read first if file already exists (via read) unless intentional. Create parent directories automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Destination file path"},
                    "content": {"type": "string", "description": "Complete file contents"}
                },
                "required": ["filePath", "content"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit file using exact string search. oldString must appear exactly once. For global rename set replaceAll=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string", "description": "String to replace, must be exact including whitespace"},
                    "newString": {"type": "string", "description": "Replacement string"},
                    "replaceAll": {"type": "boolean", "default": False}
                },
                "required": ["filePath", "oldString", "newString"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob pattern (e.g. **/*.py, src/**/*.ts). Fast for codebase exploration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Base directory, default '.'"}
                },
                "required": ["pattern"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for regex inside files. Returns file:line with snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Base directory/file, default '.'"},
                    "include": {"type": "string", "description": "File filter e.g. *.py"}
                },
                "required": ["pattern"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute bash/shell command. Use for git, npm, python, ls, etc. Always quote paths containing spaces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "workdir": {"type": "string", "description": "Working directory, default '.'"},
                    "timeout": {"type": "integer", "description": "Timeout ms, default 120000"}
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "skill_list",
            "description": "List available skills in .agent/skills. Call before working on tasks that require skill guidance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "skill_load",
            "description": "Load full skill contents (SKILL.md) for detailed guidance. Argument: name = skill folder name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name (folder in .agent/skills)"}
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    },
]

# ---------- Implementation ----------

def _truncate(s: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    return s

def tool_read(filePath: str = None, offset: int = 1, limit: int = 2000, **kwargs) -> str:
    # Handle parameter aliases to prevent loss (LLM may send filepath, path, file_path)
    if not filePath:
        filePath = kwargs.get("filepath") or kwargs.get("file_path") or kwargs.get("path") or kwargs.get("filename")
    if not filePath:
        return "Error: filePath is required (received empty). Expected {filePath: 'path/to/file'}"
    p = pathlib.Path(filePath)
    if not p.exists():
        return f"Error: file not found: {filePath}"
    if p.is_dir():
        try:
            entries = sorted(os.listdir(p))
            lines = []
            for e in entries:
                full = p / e
                suffix = "/" if full.is_dir() else ""
                # also show hidden (os.listdir already does)
                lines.append(f"{e}{suffix}")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error listing dir {filePath}: {e}"
    try:
        size = p.stat().st_size
        # Detect binary/image/pdf via magic + ext for b.md (only Responses supported)
        is_image = False
        is_pdf = False
        mime = None
        ext = p.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"):
            is_image = True
        if ext == ".pdf":
            is_pdf = True
        try:
            with open(p, "rb") as fh:
                head = fh.read(12)
                if head.startswith(b"\xFF\xD8\xFF"):
                    is_image = True
                    mime = "image/jpeg"
                elif head.startswith(b"\x89PNG"):
                    is_image = True
                    mime = "image/png"
                elif head.startswith(b"GIF8"):
                    is_image = True
                    mime = "image/gif"
                elif head.startswith(b"RIFF") and b"WEBP" in head:
                    is_image = True
                    mime = "image/webp"
                elif head.startswith(b"%PDF"):
                    is_pdf = True
                    mime = "application/pdf"
        except:
            pass
        if is_image:
            import mimetypes as _mt
            if not mime:
                mime,_ = _mt.guess_type(str(p))
                mime = mime or "image/jpeg"
            abs_path = str(p.resolve())
            # Marker per b.md - will be rejected if not a Response (providers.py)
            return f"Image read successfully\n[File: {filePath} | type: {mime} | {size} bytes]\n[[VISION_IMAGE:{abs_path}]]"
        if is_pdf:
            mime = mime or "application/pdf"
            abs_path = str(p.resolve())
            return f"PDF read successfully\n[File: {filePath} | type: {mime} | {size} bytes]\n[[INPUT_FILE:{abs_path}]]"
        # limit size for text
        if size > MAX_READ_BYTES * 5:
            return f"Error: file too large ({size} bytes), use offset/limit or grep"
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        # offset 1-indexed
        start = max(0, (offset or 1) - 1)
        end = start + (limit or 2000)
        chosen = all_lines[start:end]
        # prefix line numbers per read tool spec
        out = []
        for i, line in enumerate(chosen, start=start+1):
            # truncate long lines >2000 chars
            if len(line) > 2000:
                line = line[:2000] + "...[truncated]\n"
            out.append(f"{i}: {line.rstrip(chr(10))}")
        header = f"[File: {filePath} | Lines {start+1}-{min(end, total)}/{total} | {size} bytes]"
        if not out:
            return header + "\n(empty range)"
        return header + "\n" + "\n".join(out)
    except Exception as e:
        return f"Error read {filePath}: {e}"

def tool_write(filePath: str = None, content: str = None, **kwargs) -> str:
    # Alias handling
    if not filePath:
        filePath = kwargs.get("filepath") or kwargs.get("file_path") or kwargs.get("path")
    if not content:
        content = kwargs.get("Content") or kwargs.get("text")
    if not filePath:
        return "Error: filePath is required"
    if content is None:
        return "Error: content is required"
    p = pathlib.Path(filePath)
    try:
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content)} chars to {filePath}"
    except Exception as e:
        return f"Error write {filePath}: {e}"

def tool_edit(filePath: str = None, oldString: str = None, newString: str = None, replaceAll: bool = False, **kwargs) -> str:
    # Alias handling
    if not filePath:
        filePath = kwargs.get("filepath") or kwargs.get("file_path") or kwargs.get("path")
    if oldString is None:
        oldString = kwargs.get("old_string") or kwargs.get("oldstring") or kwargs.get("oldString")
    if newString is None:
        newString = kwargs.get("new_string") or kwargs.get("newstring") or kwargs.get("newString")
    if not filePath:
        return "Error: filePath is required"
    if oldString is None:
        return "Error: oldString is required"
    if newString is None:
        return "Error: newString is required"
    p = pathlib.Path(filePath)
    if not p.exists():
        return f"Error: file not found: {filePath}"
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if oldString not in text:
            return f"Error: oldString not found in {filePath}"
        if not replaceAll:
            count = text.count(oldString)
            if count > 1:
                return f"Error: oldString found {count} times, use replaceAll=true or narrow context"
            text = text.replace(oldString, newString, 1)
        else:
            text = text.replace(oldString, newString)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return f"OK: edited {filePath} (replaceAll={replaceAll})"
    except Exception as e:
        return f"Error edit {filePath}: {e}"

def tool_glob(pattern: str = None, path: str = ".", **kwargs) -> str:
    if not pattern:
        pattern = kwargs.get("query") or kwargs.get("glob") or kwargs.get("Pattern")
    if not pattern:
        return "Error: pattern is required"
    if not path or path == ".":
        path = kwargs.get("base") or kwargs.get("dir") or kwargs.get("directory") or path
    base = pathlib.Path(path) if path else pathlib.Path(".")
    if not base.exists():
        return f"Error: base path does not exist: {path}"
    try:
        # use recursive glob
        # fnmatch + os.walk for no dependencies
        matches: List[str] = []
        # if pattern contains **, use pathlib.rglob
        if "**" in pattern:
            # pathlib glob
            for m in base.glob(pattern):
                matches.append(str(m).replace("\\", "/"))
        else:
            # normal glob
            full_pat = str(base / pattern)
            for m in globmod.glob(full_pat, recursive=True):
                matches.append(str(pathlib.Path(m)).replace("\\", "/"))
        # fallback: walk + fnmatch for simple patterns
        if not matches and ("*" in pattern or "?" in pattern):
            for root, dirs, files in os.walk(base):
                for name in files + dirs:
                    rel = os.path.relpath(os.path.join(root, name), str(base)).replace("\\", "/")
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                        matches.append(os.path.join(str(base), rel).replace("\\", "/"))
                if len(matches) > 500:
                    break
        matches = sorted(set(matches))
        if len(matches) > 500:
            matches = matches[:500]
            note = "\n...[truncated 500+ matches]..."
        else:
            note = ""
        if not matches:
            return f"(no match) pattern={pattern} path={path}"
        return "\n".join(matches) + note
    except Exception as e:
        return f"Error glob {pattern}: {e}"

def tool_grep(pattern: str = None, path: str = ".", include: str = "", **kwargs) -> str:
    if not pattern:
        pattern = kwargs.get("query") or kwargs.get("regex") or kwargs.get("Pattern")
    if not pattern:
        return "Error: pattern is required"
    if not path or path == ".":
        path = kwargs.get("dir") or kwargs.get("directory") or kwargs.get("base") or path
    base = pathlib.Path(path) if path else pathlib.Path(".")
    if not base.exists():
        return f"Error: path does not exist: {path}"
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: regex invalid '{pattern}': {e}"
    results: List[str] = []
    max_hits = 200
    # determine file list
    files: List[pathlib.Path] = []
    if base.is_file():
        files = [base]
    else:
        for root, dirs, filenames in os.walk(base):
            # skip .git, node_modules, .venv, __pycache__
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".venv", "venv", "__pycache__", ".agent")]
            for fn in filenames:
                if include and not fnmatch.fnmatch(fn, include):
                    continue
                files.append(pathlib.Path(root) / fn)
                if len(files) > 5000:
                    break
            if len(results) >= max_hits:
                break
    for fp in files:
        try:
            # skip binary
            if fp.suffix in (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".bin"):
                continue
            if fp.stat().st_size > MAX_READ_BYTES * 2:
                continue
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        # truncate line
                        line = line.strip()
                        if len(line) > 300:
                            line = line[:300] + "..."
                        results.append(f"{str(fp).replace(chr(92),'/')}:{lineno}: {line}")
                        if len(results) >= max_hits:
                            break
        except:
            continue
        if len(results) >= max_hits:
            break
    if not results:
        return f"(no match) pattern={pattern} path={path} include={include}"
    out = "\n".join(results)
    if len(results) >= max_hits:
        out += f"\n...[truncated {max_hits} hits]..."
    return out

def tool_bash(command: str = None, workdir: str = ".", timeout: int = 120000, **kwargs) -> str:
    if not command:
        command = kwargs.get("cmd") or kwargs.get("command") or kwargs.get("script")
    if not command:
        return "Error: command is required"
    if not workdir or workdir == ".":
        workdir = kwargs.get("cwd") or kwargs.get("dir") or kwargs.get("path") or workdir
    wd = pathlib.Path(workdir) if workdir else pathlib.Path(".")
    if not wd.exists():
        return f"Error: workdir does not exist: {workdir}"
    try:
        # timeout ms -> s
        timeout_s = max(1, (timeout or 120000) / 1000)
        # Windows: shell=True required, use bash if available else cmd
        # We use shell=True so command string is executed directly
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="ignore"
        )
        out = ""
        if result.stdout:
            out += result.stdout
        if result.stderr:
            out += ("\n[stderr]\n" + result.stderr if out else result.stderr)
        if not out:
            out = f"(no output) exit={result.returncode}"
        else:
            out = f"[exit {result.returncode}]\n" + out
        return _truncate(out)
    except subprocess.TimeoutExpired:
        return f"Error: timeout after {timeout}ms: {command}"
    except Exception as e:
        return f"Error bash '{command}': {e}"

# Skill tools will be injected via skills.py at runtime, but provide stubs
def tool_skill_list() -> str:
    from .skills import list_skills
    return list_skills()

def tool_skill_load(name: str) -> str:
    from .skills import load_skill
    return load_skill(name)


# Dispatcher
TOOL_IMPL: Dict[str, Any] = {
    "read": tool_read,
    "write": tool_write,
    "edit": tool_edit,
    "glob": tool_glob,
    "grep": tool_grep,
    "bash": tool_bash,
    "skill_list": lambda **kw: tool_skill_list(),
    "skill_load": tool_skill_load,
}

def execute_tool(name: str, arguments: Any) -> str:
    """Execute tool safely, arguments can be dict or json string."""
    fn = TOOL_IMPL.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOL_IMPL.keys())}"
    # parse arguments if JSON string
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON arguments for {name}: {e} | raw={arguments[:500]}"
    if not isinstance(arguments, dict):
        return f"Error: arguments must be object for {name}, got {type(arguments)}"
    try:
        result = fn(**arguments)
        # ensure string
        if not isinstance(result, str):
            result = str(result)
        return _truncate(result)
    except TypeError as e:
        return f"Error: argument mismatch for {name} {arguments}: {e}"
    except Exception as e:
        return f"Error executing {name}: {e}"

def get_tool_definitions() -> List[Dict[str, Any]]:
    return TOOL_DEFINITIONS
