"""History JSONL per session di .agent/hists/<session_id>.jsonl"""
import json
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import HISTS_DIR

def generate_session_id() -> str:
    return f"ses_{uuid.uuid4().hex}"

def _hist_path(session_id: str, hists_dir: str = None) -> pathlib.Path:
    if hists_dir is None:
        hists_dir = HISTS_DIR
    return pathlib.Path(hists_dir) / f"{session_id}.jsonl"

def append_history(session_id: str, entry: Dict[str, Any], hists_dir: str = None):
    if hists_dir is None:
        hists_dir = HISTS_DIR
    p = _hist_path(session_id, hists_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("ts", time.time())
    entry.setdefault("session_id", session_id)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_history(session_id: str, hists_dir: str = None) -> List[Dict[str, Any]]:
    if hists_dir is None:
        hists_dir = HISTS_DIR
    p = _hist_path(session_id, hists_dir)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except:
                continue
    return out

def load_messages(session_id: str, hists_dir: str = None) -> List[Dict[str, Any]]:
    """Kembalikan list messages (role/content/tool_calls) untuk resume context. History tetap OpenAI format, model/think disimpan terpisah dan tidak dimasukkan ke messages LLM."""
    if hists_dir is None:
        hists_dir = HISTS_DIR
    entries = load_history(session_id, hists_dir)
    msgs: List[Dict[str, Any]] = []
    for e in entries:
        # entry bisa berupa message langsung atau wrapper
        if "role" in e and "content" in e:
            # filter ts etc, tapi pertahankan hanya field LLM (model/think tidak dikirim ke LLM, hanya metadata)
            m = {k: v for k, v in e.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
            msgs.append(m)
        elif "message" in e and isinstance(e["message"], dict):
            msgs.append(e["message"])
    return msgs

def get_last_model_and_think(session_id: str, hists_dir: str = None) -> tuple[Optional[str], Optional[str]]:
    """Ambil model dan thinking variant terakhir dari history. Returns (model, think) atau (None, None) jika tidak ada."""
    if hists_dir is None:
        hists_dir = HISTS_DIR
    entries = load_history(session_id, hists_dir)
    last_model = None
    last_think = None
    for e in entries:
        # cari field model/think_variant atau model/thinking di entry
        if "model" in e:
            last_model = e.get("model")
        if "think_variant" in e:
            last_think = e.get("think_variant")
        elif "thinking" in e:
            last_think = e.get("thinking")
        elif "think" in e:
            last_think = e.get("think")
        # juga cek di dalam message wrapper jika ada
        if "message" in e and isinstance(e["message"], dict):
            msg = e["message"]
            if "model" in msg:
                last_model = msg.get("model")
            if "think_variant" in msg:
                last_think = msg.get("think_variant")
    return last_model, last_think

def list_sessions(hists_dir: str = None) -> List[str]:
    if hists_dir is None:
        hists_dir = HISTS_DIR
    p = pathlib.Path(hists_dir)
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("*.jsonl")])

def save_message(session_id: str, message: Dict[str, Any], hists_dir: str = None):
    if hists_dir is None:
        hists_dir = HISTS_DIR
    append_history(session_id, message, hists_dir)
