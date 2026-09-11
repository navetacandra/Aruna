"""Tool definitions & implementations - filesystem kuat, tanpa external deps."""
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
            "description": "Baca isi file. Gunakan untuk inspeksi kode, log, config. Support offset/limit untuk file besar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path absolut atau relatif ke file"},
                    "offset": {"type": "integer", "description": "Baris mulai (1-indexed)", "default": 1},
                    "limit": {"type": "integer", "description": "Jumlah baris maksimal, default 2000"}
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
            "description": "Tulis file baru atau overwrite. SELALU baca dulu jika file sudah ada (via read) kecuali memang disengaja. Buat direktori induk otomatis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path file tujuan"},
                    "content": {"type": "string", "description": "Isi file lengkap"}
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
            "description": "Edit file dengan pencarian string eksak. oldString harus muncul tepat sekali. Untuk rename global set replaceAll=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string", "description": "String yang diganti, harus eksak termasuk whitespace"},
                    "newString": {"type": "string", "description": "String pengganti"},
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
            "description": "Cari file berdasarkan pola glob (mis. **/*.py, src/**/*.ts). Cepat untuk eksplorasi codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Direktori basis, default '.'"}
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
            "description": "Cari regex di dalam file. Mengembalikan file:line dengan cuplikan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Direktori/file basis, default '.'"},
                    "include": {"type": "string", "description": "Filter file mis. *.py"}
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
            "description": "Eksekusi perintah bash/shell. Gunakan untuk git, npm, python, ls, dll. Selalu quote path yang mengandung spasi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Perintah shell"},
                    "workdir": {"type": "string", "description": "Direktori kerja, default '.'"},
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
            "description": "List skill yang tersedia di .agent/skills. Panggil sebelum mengerjakan tugas yang butuh panduan skill.",
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
            "description": "Muat isi lengkap skill (SKILL.md) untuk panduan detail. Argumen: name = nama folder skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nama skill (folder di .agent/skills)"}
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    },
]

# ---------- Implementasi ----------

def _truncate(s: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    # Truncate cerdas - hemat token tapi pertahankan head + tail penting
    if len(s) <= max_chars:
        return s
    # Untuk output tool panjang, simpan head + tail dengan notice di tengah
    # Head 60% + tail 40% agar header dan awal file (yang penting) tetap ada
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len - 100  # 100 untuk notice
    notice = f"\n\n...[TRUNCATED cerdas {len(s)-max_chars} chars, total {len(s)} -> {max_chars} (hemat {(1-max_chars/len(s))*100:.0f}%)]...\n\n"
    return s[:head_len] + notice + s[-tail_len:] if tail_len > 0 else s[:max_chars] + notice

def tool_read(filePath: str, offset: int = 1, limit: int = 2000) -> str:
    p = pathlib.Path(filePath)
    if not p.exists():
        return f"Error: file tidak ditemukan: {filePath}"
    if p.is_dir():
        try:
            entries = sorted(os.listdir(p))
            lines = []
            for e in entries:
                full = p / e
                suffix = "/" if full.is_dir() else ""
                # tampilkan hidden juga (os.listdir sudah)
                lines.append(f"{e}{suffix}")
            return "\n".join(lines) if lines else "(direktori kosong)"
        except Exception as e:
            return f"Error list dir {filePath}: {e}"
    try:
        size = p.stat().st_size
        # Deteksi binary/image/pdf via magic + ext untuk b.md (hanya Responses yang didukung)
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
            # Marker sesuai b.md - akan ditolak jika bukan Response (providers.py)
            return f"Image read successfully\n[File: {filePath} | type: {mime} | {size} bytes]\n[[VISION_IMAGE:{abs_path}]]"
        if is_pdf:
            mime = mime or "application/pdf"
            abs_path = str(p.resolve())
            return f"PDF read successfully\n[File: {filePath} | type: {mime} | {size} bytes]\n[[INPUT_FILE:{abs_path}]]"
        # batasi ukuran untuk text
        if size > MAX_READ_BYTES * 5:
            return f"Error: file terlalu besar ({size} bytes), gunakan offset/limit atau grep"
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        # offset 1-indexed
        start = max(0, (offset or 1) - 1)
        end = start + (limit or 2000)
        chosen = all_lines[start:end]
        # prefix nomor baris seperti read tool spec
        out = []
        for i, line in enumerate(chosen, start=start+1):
            # truncate line panjang >2000 chars
            if len(line) > 2000:
                line = line[:2000] + "...[truncated]\n"
            out.append(f"{i}: {line.rstrip(chr(10))}")
        header = f"[File: {filePath} | Lines {start+1}-{min(end, total)}/{total} | {size} bytes]"
        if not out:
            return header + "\n(empty range)"
        return header + "\n" + "\n".join(out)
    except Exception as e:
        return f"Error read {filePath}: {e}"

def tool_write(filePath: str, content: str) -> str:
    p = pathlib.Path(filePath)
    try:
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content)} chars to {filePath}"
    except Exception as e:
        return f"Error write {filePath}: {e}"

def tool_edit(filePath: str, oldString: str, newString: str, replaceAll: bool = False) -> str:
    p = pathlib.Path(filePath)
    if not p.exists():
        return f"Error: file tidak ditemukan: {filePath}"
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if oldString not in text:
            return f"Error: oldString tidak ditemukan di {filePath}"
        if not replaceAll:
            count = text.count(oldString)
            if count > 1:
                return f"Error: oldString ditemukan {count} kali, gunakan replaceAll=true atau perkecil konteks"
            text = text.replace(oldString, newString, 1)
        else:
            text = text.replace(oldString, newString)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return f"OK: edited {filePath} (replaceAll={replaceAll})"
    except Exception as e:
        return f"Error edit {filePath}: {e}"

def tool_glob(pattern: str, path: str = ".") -> str:
    base = pathlib.Path(path) if path else pathlib.Path(".")
    if not base.exists():
        return f"Error: base path tidak ada: {path}"
    try:
        # gunakan glob recursif
        # fnmatch + os.walk untuk tanpa deps
        matches: List[str] = []
        # jika pattern mengandung **, gunakan pathlib.rglob
        if "**" in pattern:
            # pathlib glob
            for m in base.glob(pattern):
                matches.append(str(m).replace("\\", "/"))
        else:
            # glob normal
            full_pat = str(base / pattern)
            for m in globmod.glob(full_pat, recursive=True):
                matches.append(str(pathlib.Path(m)).replace("\\", "/"))
        # fallback: walk + fnmatch untuk pattern sederhana
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

def tool_grep(pattern: str, path: str = ".", include: str = "") -> str:
    base = pathlib.Path(path) if path else pathlib.Path(".")
    if not base.exists():
        return f"Error: path tidak ada: {path}"
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: regex invalid '{pattern}': {e}"
    results: List[str] = []
    max_hits = 200
    # tentukan file list
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

def tool_bash(command: str, workdir: str = ".", timeout: int = 120000) -> str:
    wd = pathlib.Path(workdir) if workdir else pathlib.Path(".")
    if not wd.exists():
        return f"Error: workdir tidak ada: {workdir}"
    try:
        # timeout ms -> s
        timeout_s = max(1, (timeout or 120000) / 1000)
        # Windows: shell=True perlu, gunakan bash jika ada else cmd
        # Kita pakai shell=True agar command string langsung dieksekusi
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

# Skill tools will be injected via skills.py at runtime, tapi sediakan stub
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
    """Eksekusi tool dengan aman, arguments bisa dict atau json string."""
    fn = TOOL_IMPL.get(name)
    if not fn:
        return f"Error: tool tidak dikenal '{name}'. Available: {', '.join(TOOL_IMPL.keys())}"
    # parse arguments jika string JSON
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f"Error: arguments JSON invalid untuk {name}: {e} | raw={arguments[:500]}"
    if not isinstance(arguments, dict):
        return f"Error: arguments harus object untuk {name}, got {type(arguments)}"
    try:
        result = fn(**arguments)
        # pastikan string
        if not isinstance(result, str):
            result = str(result)
        return _truncate(result)
    except TypeError as e:
        return f"Error: argumen tidak cocok untuk {name} {arguments}: {e}"
    except Exception as e:
        return f"Error eksekusi {name}: {e}"

def get_tool_definitions() -> List[Dict[str, Any]]:
    return TOOL_DEFINITIONS
