"""Provider abstraction untuk multi-SDK fallback - HANYA via opencode.ai/zen/v1.

Mendukung 3 format sesuai SDK-example.md:
- openai_chat: POST {base}/zen/v1/chat/completions (OpenAI SDK chat)
  Request: {model, messages:[{role,content}], stream}
  Response: {choices:[{message:{role,content}, finish_reason}]}
  SSE: data: {"choices":[{"delta":{"content":"..."}}]} + data: [DONE]

- openai_responses: POST {base}/zen/v1/responses (OpenAI Responses, plural)
  Request: {model, instructions: system, input:[{role,content:[{type:input_text/output_text,text}]}], store:false, stream}
  Response: {output:[{type:message, role:assistant, content:[{type:output_text,text}]}], usage}
  SSE: event: response.output_text.delta + data: {"delta":"..."} ; event: response.completed

- anthropic_messages: POST {base}/zen/v1/messages (Anthropic SDK via opencode proxy)
  Request: {model, max_tokens, system, messages:[{role,content}], stream, tools:[{name,description,input_schema}]}
  Response: {content:[{type:text,text}], stop_reason}
  SSE: event: message_start / content_block_start / content_block_delta / message_delta / message_stop

History disimpan tetap format OpenAI (messages[] dengan role user/assistant/tool + tool_calls).
Saat request, payload dikonversi sesuai SDK. State per-model disimpan di .agent/llm_provider_state.json.
Fallback HANYA ke base opencode.ai/zen/v1, rubah state jika fallback terjadi.
"""
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import OPENCODE_BASE_URL, OPENCODE_UA, RESPONSES_MODELS, PROVIDER_STATE_FILE as CFG_PROVIDER_STATE_FILE

import re
import uuid
import base64 as _b64
import mimetypes as _mimes

def base_model_id(model: str) -> str:
    return re.sub(r"\([^()]+\)\s*$", "", str(model or "")).strip()

def is_responses_model(model: str) -> bool:
    base = base_model_id(model)
    return base in RESPONSES_MODELS or "muse-spark" in base

def is_anthropic_model(model: str) -> bool:
    m = model.lower()
    return any(k in m for k in ("claude", "anthropic", "sonnet", "opus", "haiku")) and "muse-spark" not in m

# ---------- Conversion helpers ----------

def openai_tools_to_anthropic(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", {}) if t.get("type") == "function" else t
        name = fn.get("name")
        if not name:
            if "name" in t and "input_schema" in t:
                out.append(t)
                continue
            continue
        out.append({
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        })
    return out if out else None

def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out = []
    for t in tools:
        if "function" in t:
            out.append(t)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}}
            }
        })
    return out

def openai_messages_to_anthropic(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """OpenAI messages -> (system, anthropic_messages)."""
    system_parts: List[str] = []
    anthropic_msgs: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        system_parts.append(part["text"])
                    else:
                        system_parts.append(str(part))
            else:
                system_parts.append(str(content))
            continue
        if role == "user":
            if isinstance(content, str):
                anthropic_msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                anthropic_msgs.append({"role": "user", "content": content})
            elif content is None:
                anthropic_msgs.append({"role": "user", "content": ""})
            else:
                anthropic_msgs.append({"role": "user", "content": str(content)})
            continue
        if role == "assistant":
            tool_calls = m.get("tool_calls")
            if not tool_calls:
                c = content or ""
                if isinstance(c, str):
                    anthropic_msgs.append({"role": "assistant", "content": c})
                else:
                    anthropic_msgs.append({"role": "assistant", "content": str(c)})
            else:
                blocks: List[Dict[str, Any]] = []
                if content and isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except:
                        args = {}
                        if isinstance(args_raw, str):
                            args = {"_raw": args_raw}
                    tid = tc.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
                    blocks.append({"type": "tool_use", "id": tid, "name": name, "input": args if isinstance(args, dict) else {}})
                anthropic_msgs.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            tool_use_id = m.get("tool_call_id") or m.get("id") or "unknown"
            tool_content = m.get("content") or ""
            block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": str(tool_content)}
            if anthropic_msgs and anthropic_msgs[-1].get("role") == "user":
                last = anthropic_msgs[-1]
                if isinstance(last.get("content"), list) and any(b.get("type")=="tool_result" for b in last["content"]):
                    last["content"].append(block)
                else:
                    anthropic_msgs.append({"role": "user", "content": [block]})
            else:
                anthropic_msgs.append({"role": "user", "content": [block]})
            continue
        anthropic_msgs.append({"role": "user", "content": str(content or "")})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_msgs

def anthropic_response_to_openai(content_blocks: List[Dict[str, Any]], stop_reason: Optional[str] = None) -> Dict[str, Any]:
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in content_blocks or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)}
            })
    content = "".join(text_parts)
    finish = "stop"
    if stop_reason == "tool_use":
        finish = "tool_calls"
    elif stop_reason in ("end_turn", "stop"):
        finish = "stop"
    return {"content": content, "tool_calls": tool_calls if tool_calls else None, "finish_reason": finish}

def openai_messages_to_responses(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """OpenAI messages -> (instructions, input) untuk Responses API sesuai SDK-example.md sec 2 & 5.
    - system -> instructions (digabung)
    - user -> {role:user, content:[{type:input_text, text}]}
    - assistant -> {role:assistant, content:[{type:output_text, text}]}
    - tool -> {role:user, content:[{type:input_text, text: tool result}] }  (fallback, karena spec tidak cover tools di responses, tapi kita normalisasi)
    History eksplisit: seluruh conversation dikirim kembali di input, terbaru paling akhir.
    """
    instructions_parts: List[str] = []
    inputs: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        # content bisa string, kita bungkus
        if role == "system":
            if isinstance(content, str):
                instructions_parts.append(content)
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and "text" in p:
                        instructions_parts.append(p["text"])
                    else:
                        instructions_parts.append(str(p))
            else:
                instructions_parts.append(str(content))
            continue
        if role == "user":
            text = content if isinstance(content, str) else str(content)
            # b.md: file input untuk Responses -> input_file / input_image
            has_image = bool(VISION_IMAGE_RE.search(text)) if isinstance(text,str) else False
            has_file = bool(INPUT_FILE_RE.search(text)) if isinstance(text,str) else False
            if has_image or has_file:
                # extract
                clean_img, imgs = _extract_vision_images(text) if has_image else (text, [])
                # setelah extract image, masih ada file marker?
                clean, files = _extract_input_files(clean_img) if has_file else (clean_img, [])
                parts: List[Dict[str, Any]] = []
                if clean.strip():
                    parts.append({"type": "input_text", "text": clean})
                for mime, b64 in imgs:
                    parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "auto"})
                for fpath, mime, b64 in files:
                    # b.md sec2: input_file dengan file_data (fallback upload 404)
                    # pakai file_data data URL sesuai real test yang sukses
                    fname = pathlib.Path(fpath).name if fpath else "file"
                    parts.append({"type": "input_file", "filename": fname, "file_data": f"data:{mime};base64,{b64}"})
                if not parts:
                    parts.append({"type": "input_text", "text": text})
                inputs.append({"role": "user", "content": parts})
            else:
                inputs.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
            continue
        if role == "assistant":
            tool_calls = m.get("tool_calls")
            # Jika tanpa tool, simple output_text
            if not tool_calls:
                text = content if isinstance(content, str) else str(content)
                # jika kosong, tetap kirim array kosong? tapi spec butuh text
                if text.strip():
                    inputs.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
                else:
                    # assistant kosong tanpa tool, tetap skip? tapi kita masukkan kosong untuk jaga turn
                    inputs.append({"role": "assistant", "content": [{"type": "output_text", "text": ""}]})
            else:
                # ada tool_calls: kita representasikan sebagai assistant dengan output_text + function_call
                # Di Responses API, function_call adalah type terpisah, tapi kita sederhanakan jadi output_text + tool info di content
                # Untuk kompatibilitas, kita gabung text + tool_calls sebagai input_text json
                # Namun untuk request, kita tetap kirim text biasa dan biarkan tools di payload terpisah
                text = content if isinstance(content, str) else ""
                blocks = []
                if text.strip():
                    blocks.append({"type": "output_text", "text": text})
                # tool_calls di responses tidak dikirim via input, tapi via tools di payload, jadi kita tidak perlu embed di input
                # Tapi kita tetap perlu merepresentasikan bahwa assistant pernah memanggil tool, jadi kita tambahkan placeholder
                # Sederhananya: kirim assistant dengan text, tool result nanti sebagai user input_text
                inputs.append({"role": "assistant", "content": blocks if blocks else [{"type": "output_text", "text": ""}]})
                # tool results akan dikirim sebagai user input_text terpisah di iterasi berikutnya (role tool)
                # kita tidak embed tool_calls di input, karena Responses API handling tool via output
                continue
        if role == "tool":
            # tool result -> user input_text, support file input b.md
            text = str(content)
            has_image = bool(VISION_IMAGE_RE.search(text))
            has_file = bool(INPUT_FILE_RE.search(text))
            if has_image or has_file:
                clean_img, imgs = _extract_vision_images(text) if has_image else (text, [])
                clean, files = _extract_input_files(clean_img) if has_file else (clean_img, [])
                prefix = f"Tool result ({m.get('name','')}): {clean}" if clean.strip() else f"Tool result ({m.get('name','')}):"
                parts: List[Dict[str, Any]] = []
                if prefix.strip():
                    parts.append({"type": "input_text", "text": prefix})
                for mime, b64 in imgs:
                    parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "auto"})
                for fpath, mime, b64 in files:
                    fname = pathlib.Path(fpath).name if fpath else "file"
                    parts.append({"type": "input_file", "filename": fname, "file_data": f"data:{mime};base64,{b64}"})
                if not parts:
                    parts.append({"type": "input_text", "text": prefix})
                inputs.append({"role": "user", "content": parts})
            else:
                inputs.append({"role": "user", "content": [{"type": "input_text", "text": f"Tool result ({m.get('name','')}): {text}"}]})
            continue
        # fallback
        inputs.append({"role": "user", "content": [{"type": "input_text", "text": str(content)}]})
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, inputs

def responses_output_to_openai(output: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Konversi Responses API output -> OpenAI {content, tool_calls}."""
    # output: [{type:message, role:assistant, content:[{type:output_text,text}]}]
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for item in output or []:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text_parts.append(c.get("text",""))
                elif c.get("type") == "function_call":
                    # function_call di responses
                    tool_calls.append({
                        "id": c.get("call_id") or c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": c.get("name"), "arguments": c.get("arguments") or "{}"}
                    })
        elif item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}
            })
    content = "".join(text_parts)
    return {"content": content, "tool_calls": tool_calls if tool_calls else None, "finish_reason": "tool_calls" if tool_calls else "stop"}

# ---------- File input helpers (b.md - hanya Responses yang didukung untuk binary) ----------
VISION_IMAGE_RE = re.compile(r"\[\[VISION_IMAGE:(.+?)\]\]")
INPUT_FILE_RE = re.compile(r"\[\[INPUT_FILE:(.+?)\]\]")
# Generic binary marker (fallback) - jika ada binary tanpa marker spesifik, tetap deteksi via header
BINARY_HEADER_RE = re.compile(r"\[Binary file:")

def _get_mime_and_b64(path_str: str) -> Optional[Tuple[str, str]]:
    p = pathlib.Path(path_str.strip())
    if not p.exists():
        p2 = pathlib.Path.cwd() / path_str.strip()
        if p2.exists():
            p = p2
        else:
            return None
    try:
        data = p.read_bytes()
        if len(data) > 8*1024*1024:
            data = data[:8*1024*1024]
        mime = None
        if data.startswith(b"\xFF\xD8\xFF"):
            mime = "image/jpeg"
        elif data.startswith(b"\x89PNG"):
            mime = "image/png"
        elif data.startswith(b"GIF8"):
            mime = "image/gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            mime = "image/webp"
        elif data.startswith(b"%PDF"):
            mime = "application/pdf"
        else:
            mime,_ = _mimes.guess_type(str(p))
        mime = mime or "application/octet-stream"
        b64 = _b64.b64encode(data).decode("ascii")
        return mime, b64
    except:
        return None

def _has_binary_marker(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(VISION_IMAGE_RE.search(text) or INPUT_FILE_RE.search(text) or BINARY_HEADER_RE.search(text))

def _extract_vision_images(text: str) -> Tuple[str, List[Tuple[str,str]]]:
    if not text or not isinstance(text, str):
        return text or "", []
    matches = VISION_IMAGE_RE.findall(text)
    imgs: List[Tuple[str,str]] = []
    for m in matches:
        res = _get_mime_and_b64(m)
        if res: imgs.append(res)
    clean = VISION_IMAGE_RE.sub("", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean and imgs:
        clean = "Jelaskan gambar ini."
    return clean, imgs

def _extract_input_files(text: str) -> Tuple[str, List[Tuple[str,str,str]]]:
    """Return (clean_text, [(path,mime,b64),...]) untuk INPUT_FILE"""
    if not text or not isinstance(text, str):
        return text or "", []
    matches = INPUT_FILE_RE.findall(text)
    files: List[Tuple[str,str,str]] = []
    for m in matches:
        res = _get_mime_and_b64(m)
        if res:
            mime,b64 = res
            files.append((m.strip(), mime, b64))
    clean = INPUT_FILE_RE.sub("", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean and files:
        clean = "Ringkas dokumen ini."
    return clean, files

# ---------- Provider definitions ----------

PROVIDER_STATE_FILE = CFG_PROVIDER_STATE_FILE

def load_provider_state() -> Dict[str, Any]:
    p = pathlib.Path(PROVIDER_STATE_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except:
        return {}

def save_provider_state(state: Dict[str, Any]):
    p = pathlib.Path(PROVIDER_STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except:
        pass

def update_provider_state(model: str, provider: str, endpoint: str, base_url: str):
    state = load_provider_state()
    state[model] = {"provider": provider, "endpoint": endpoint, "base_url": base_url, "updated_at": time.time()}
    bmid = base_model_id(model)
    if bmid != model:
        state[bmid] = state[model]
    save_provider_state(state)

def get_saved_provider(model: str) -> Optional[Dict[str, Any]]:
    state = load_provider_state()
    if model in state:
        return state[model]
    bmid = base_model_id(model)
    return state.get(bmid)

def _opencode_headers(session_id: Optional[str] = None) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer public",
        "User-Agent": OPENCODE_UA,
        "x-opencode-client": "desktop",
        "x-opencode-session": session_id or f"ses_{uuid.uuid4().hex}",
        "x-opencode-request": f"msg_{uuid.uuid4().hex}",
        "x-opencode-project": "global",
        "Accept": "text/event-stream",
    }

class ProviderSpec:
    def __init__(self, name: str, base_url: str, endpoint: str, sdk: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.sdk = sdk
    @property
    def url(self) -> str:
        return self.base_url + self.endpoint
    def __repr__(self):
        return f"ProviderSpec({self.name} {self.url} sdk={self.sdk})"

def build_candidate_providers(model: str, base_url: Optional[str] = None) -> List[ProviderSpec]:
    """HANYA fallback ke opencode.ai/zen/v1 (sesuai instruksi)."""
    saved = get_saved_provider(model)
    candidates: List[ProviderSpec] = []
    if saved:
        try:
            candidates.append(ProviderSpec("saved", saved["base_url"], saved["endpoint"], saved["provider"]))
        except:
            pass
    opencode_base = (base_url or OPENCODE_BASE_URL).rstrip("/")
    # endpoint sesuai SDK-example (semua di bawah /zen/v1)
    # OpenAI Chat: /zen/v1/chat/completions, Responses: /zen/v1/responses (plural), Anthropic: /zen/v1/messages
    if is_responses_model(model):
        candidates.append(ProviderSpec("opencode_responses", opencode_base, "/zen/v1/responses", "openai_responses"))
        candidates.append(ProviderSpec("opencode_chat", opencode_base, "/zen/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("opencode_anthropic", opencode_base, "/zen/v1/messages", "anthropic"))
    elif is_anthropic_model(model):
        candidates.append(ProviderSpec("opencode_anthropic", opencode_base, "/zen/v1/messages", "anthropic"))
        candidates.append(ProviderSpec("opencode_chat", opencode_base, "/zen/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("opencode_responses", opencode_base, "/zen/v1/responses", "openai_responses"))
    else:
        candidates.append(ProviderSpec("opencode_chat", opencode_base, "/zen/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("opencode_responses", opencode_base, "/zen/v1/responses", "openai_responses"))
        candidates.append(ProviderSpec("opencode_anthropic", opencode_base, "/zen/v1/messages", "anthropic"))
    # deduplikasi
    seen = set()
    uniq: List[ProviderSpec] = []
    for c in candidates:
        key = (c.url, c.sdk)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def get_headers_for_provider(provider: ProviderSpec, session_id: Optional[str] = None) -> Dict[str, str]:
    # Semua fallback hanya ke opencode, jadi header selalu opencode
    return _opencode_headers(session_id)

def prepare_payload_for_provider(provider: ProviderSpec, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], extra_body: Optional[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
    """Konversi history OpenAI -> payload sesuai SDK-example. Tolak binary jika bukan Responses."""
    sdk = provider.sdk
    # Jika ada binary file diembed dan bukan openai response -> Model tidak didukung (sesuai instruksi)
    # Deteksi marker vision/input_file atau header binary - cek model dulu (hanya muse-spark yang boleh)
    has_binary = False
    for m in messages:
        c = m.get("content")
        if isinstance(c, str) and _has_binary_marker(c):
            has_binary = True
            break
        if isinstance(c, list):
            s = json.dumps(c, ensure_ascii=False)
            if _has_binary_marker(s):
                has_binary = True
                break
    # Cek model: hanya Responses yang boleh handle binary (sesuai b.md)
    if has_binary and not is_responses_model(model):
        raise ValueError("Model tidak didukung")
    if has_binary and sdk != "openai_responses":
        raise ValueError("Model tidak didukung")
    if sdk == "openai_chat":
        # SDK-example sec 1: {model, messages:[{role,content}], stream}
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if extra_body:
            body.update(extra_body)
        return body
    elif sdk == "openai_responses":
        # SDK-example sec 2 & 5: {model, instructions, input:[{role,content:[{type,text}]}], store:false, stream, tools}
        instructions, inputs = openai_messages_to_responses(messages)
        body: Dict[str, Any] = {"model": model, "input": inputs, "stream": stream, "store": False}
        if instructions:
            body["instructions"] = instructions
        if tools:
            # Responses tools: flat {type:function, name, description, parameters} - tidak nested function
            flat_tools = []
            for t in tools:
                if t.get("type") == "function" and "function" in t:
                    fn = t["function"]
                    flat_tools.append({
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}}
                    })
                elif "name" in t:
                    # sudah flat
                    flat_tools.append(t)
                else:
                    flat_tools.append(t)
            body["tools"] = flat_tools
            body["tool_choice"] = "auto"
        if extra_body:
            # responses tidak pakai max_tokens tapi max_output_tokens
            # mapping sudah ada di llm.py? tapi kita handle di sini
            for k, v in extra_body.items():
                if k == "reasoning_effort":
                    body["reasoning"] = {"effort": str(v).lower().strip(), "summary": "auto"}
                elif k in ("max_tokens", "max_completion_tokens"):
                    body["max_output_tokens"] = v
                elif k == "max_output_tokens":
                    body[k] = v
                else:
                    body[k] = v
        return body
    elif sdk == "anthropic":
        # SDK-example sec 3: {model, max_tokens, system, messages, stream, tools}
        system, anth_msgs = openai_messages_to_anthropic(messages)
        anth_tools = openai_tools_to_anthropic(tools)
        body: Dict[str, Any] = {"model": model, "messages": anth_msgs, "stream": stream, "max_tokens": (extra_body or {}).get("max_tokens") or (extra_body or {}).get("max_output_tokens") or 4096}
        if system:
            body["system"] = system
        if anth_tools:
            body["tools"] = anth_tools
        if extra_body:
            for k, v in extra_body.items():
                if k not in ("reasoning_effort", "reasoning", "max_tokens", "max_output_tokens", "max_completion_tokens"):
                    body[k] = v
                # reasoning_effort untuk anthropic tidak ada, ignore
        return body
    else:
        raise ValueError(f"Unknown sdk {sdk}")

def parse_nonstream_response(provider: ProviderSpec, resp_json: Dict[str, Any]) -> Dict[str, Any]:
    sdk = provider.sdk
    if sdk == "openai_chat":
        msg = resp_json.get("choices", [{}])[0].get("message", {})
        return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls"), "finish_reason": resp_json.get("choices", [{}])[0].get("finish_reason", "stop")}
    elif sdk == "openai_responses":
        # SDK-example: {output:[{type:message, content:[{type:output_text,text}]}]}
        if "output" in resp_json:
            try:
                conv = responses_output_to_openai(resp_json.get("output", []))
                # jika ada usage, ignore
                if conv["content"] or conv["tool_calls"]:
                    return conv
            except: pass
            # fallback jika output kosong, coba parse lain
        if "output_text" in resp_json:
            return {"content": resp_json["output_text"], "tool_calls": None, "finish_reason": "stop"}
        # fallback ke choices (jika opencode masih kirim choices untuk responses)
        msg = resp_json.get("choices", [{}])[0].get("message", {})
        if msg:
            return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls"), "finish_reason": resp_json.get("choices", [{}])[0].get("finish_reason", "stop")}
        return {"content": "", "tool_calls": None, "finish_reason": "stop"}
    elif sdk == "anthropic":
        content_blocks = resp_json.get("content", [])
        stop_reason = resp_json.get("stop_reason")
        return anthropic_response_to_openai(content_blocks, stop_reason)
    else:
        raise ValueError(f"Unknown sdk {sdk}")
