"""Provider abstraction untuk multi-SDK fallback.

Mendukung 3 format:
- openai_chat: POST {base}/v1/chat/completions atau {base}/zen/v1/chat/completions (OpenAI SDK chat)
- openai_responses: POST {base}/v1/responses atau {base}/zen/v1/responses (OpenAI responses)
- anthropic_messages: POST {base}/v1/messages (Anthropic SDK)

History disimpan tetap format OpenAI (messages[] dengan role user/assistant/tool + tool_calls).
Saat request ke anthropic, payload dikonversi:
  - system dipisah
  - tool_calls -> content blocks tool_use
  - tool results -> user content blocks tool_result
  - tools OpenAI -> Anthropic input_schema

Streaming SSE juga dinormalisasi ke format OpenAI {content, tool_calls}.
"""
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import OPENCODE_BASE_URL, OPENCODE_UA, RESPONSES_MODELS, OPENAI_BASE_URL, ANTHROPIC_BASE_URL, PROVIDER_STATE_FILE as CFG_PROVIDER_STATE_FILE

# --- Helpers port dari llm.py ---
import re
import uuid

def base_model_id(model: str) -> str:
    return re.sub(r"\([^()]+\)\s*$", "", str(model or "")).strip()

def is_responses_model(model: str) -> bool:
    base = base_model_id(model)
    return base in RESPONSES_MODELS or "muse-spark" in base

def is_anthropic_model(model: str) -> bool:
    m = model.lower()
    return any(k in m for k in ("claude", "anthropic", "sonnet", "opus", "haiku")) and "muse-spark" not in m

# ---------- Conversion OpenAI <-> Anthropic ----------

def openai_tools_to_anthropic(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out = []
    for t in tools:
        # OpenAI: {type:"function", function:{name, description, parameters}}
        fn = t.get("function", {}) if t.get("type") == "function" else t
        # handle both wrapped and unwrapped
        name = fn.get("name")
        if not name:
            # mungkin sudah anthropic? skip
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
    """Konversi OpenAI messages -> (system, anthropic_messages).
    History tetap OpenAI, konversi hanya saat request anthropic.
    """
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
            # content bisa string atau list
            if isinstance(content, str):
                anthropic_msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # sudah anthropic style? pass
                anthropic_msgs.append({"role": "user", "content": content})
            elif content is None:
                anthropic_msgs.append({"role": "user", "content": ""})
            else:
                anthropic_msgs.append({"role": "user", "content": str(content)})
            continue

        if role == "assistant":
            # assistant bisa punya content + tool_calls
            tool_calls = m.get("tool_calls")
            # jika tanpa tool_calls, simple
            if not tool_calls:
                c = content or ""
                if isinstance(c, str):
                    anthropic_msgs.append({"role": "assistant", "content": c})
                else:
                    anthropic_msgs.append({"role": "assistant", "content": str(c)})
            else:
                # ada tool_calls -> konversi ke content blocks
                blocks: List[Dict[str, Any]] = []
                if content and isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    # parse args
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except:
                        args = {}
                        # coba simpan raw
                        if isinstance(args_raw, str):
                            args = {"_raw": args_raw}
                    # anthropic id harus ada
                    tid = tc.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
                    blocks.append({
                        "type": "tool_use",
                        "id": tid,
                        "name": name,
                        "input": args if isinstance(args, dict) else {}
                    })
                anthropic_msgs.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            # OpenAI: {role:"tool", tool_call_id, name, content}
            # Anthropic: user role dengan tool_result
            tool_use_id = m.get("tool_call_id") or m.get("id") or "unknown"
            tool_content = m.get("content") or ""
            # anthropic expects content as string or blocks, kita pakai string
            # need to wrap as tool_result block in user message
            # Jika ada beberapa tool results berturut-turut, anthropic mengizinkan multiple blocks dalam satu user message
            # Tapi di OpenAI mereka terpisah per tool; kita gabung jika sebelumnya juga tool result user?
            # Simpler: tiap tool result jadi satu user message dengan satu block
            block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(tool_content)
            }
            # cek apakah pesan terakhir juga user tool_result -> gabungkan
            if anthropic_msgs and anthropic_msgs[-1].get("role") == "user":
                last = anthropic_msgs[-1]
                # last content mungkin string atau list
                if isinstance(last.get("content"), list) and any(b.get("type")=="tool_result" for b in last["content"]):
                    last["content"].append(block)
                elif isinstance(last.get("content"), str) and last["content"] == "":
                    # shouldn't happen
                    anthropic_msgs.append({"role": "user", "content": [block]})
                else:
                    # jika last adalah user biasa tanpa tool_result, buat baru
                    anthropic_msgs.append({"role": "user", "content": [block]})
            else:
                anthropic_msgs.append({"role": "user", "content": [block]})
            continue

        # fallback: treat as user
        anthropic_msgs.append({"role": "user", "content": str(content or "")})

    system = "\n\n".join(system_parts) if system_parts else None
    # Anthropic requires alternating user/assistant, gabungkan jika ada consecutive same role? Kita biarkan, tapi perbaiki:
    # Jika ada user->user consecutive karena tool_result grouping, sudah diatasi.
    # Jika ada assistant->assistant, gabungkan? jarang.
    # Normalisasi: jika ada dua user berturut-turut tanpa tool_result, gabungkan dengan \n?
    # Biarkan untuk sekarang, provider akan handle.
    return system, anthropic_msgs

def anthropic_response_to_openai(content_blocks: List[Dict[str, Any]], stop_reason: Optional[str] = None) -> Dict[str, Any]:
    """Konversi anthropic content blocks non-stream ke OpenAI {content, tool_calls}."""
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
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                }
            })
    content = "".join(text_parts)
    # map stop_reason
    finish = "stop"
    if stop_reason == "tool_use":
        finish = "tool_calls"
    elif stop_reason in ("end_turn", "stop"):
        finish = "stop"
    return {"content": content, "tool_calls": tool_calls if tool_calls else None, "finish_reason": finish}

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
    state[model] = {
        "provider": provider,
        "endpoint": endpoint,
        "base_url": base_url,
        "updated_at": time.time()
    }
    # juga simpan untuk base model id
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

# Provider configs: list fallback order secara umum
# Jika tidak ada state tersimpan, pilih primary berdasarkan model, lalu fallback semua.

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

def _openai_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = api_key or os.environ.get("OPENAI_API_KEY") or "public"
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "Accept": "text/event-stream",
    }

def _anthropic_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    h = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Accept": "text/event-stream",
    }
    # jika tidak ada key, tetap kirim Bearer public fallback? tapi anthropic butuh x-api-key
    if not key:
        # fallback ke opencode style? biarkan kosong, nanti akan fail dan fallback
        pass
    return h

class ProviderSpec:
    def __init__(self, name: str, base_url: str, endpoint: str, sdk: str):
        self.name = name  # mis. opencode_chat, openai_chat, anthropic
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint  # mis. /zen/v1/chat/completions, /v1/chat/completions, /v1/messages
        self.sdk = sdk  # openai_chat, openai_responses, anthropic
    @property
    def url(self) -> str:
        return self.base_url + self.endpoint
    def __repr__(self):
        return f"ProviderSpec({self.name} {self.url} sdk={self.sdk})"

def build_candidate_providers(model: str, base_url: Optional[str] = None) -> List[ProviderSpec]:
    """Bangun urutan fallback berdasarkan model & saved state."""
    saved = get_saved_provider(model)
    candidates: List[ProviderSpec] = []

    # Jika ada saved dan model sama, jadikan prioritas utama
    if saved:
        try:
            candidates.append(ProviderSpec("saved", saved["base_url"], saved["endpoint"], saved["provider"]))
        except:
            pass

    # Tentukan primary berdasarkan model type
    opencode_base = base_url or OPENCODE_BASE_URL
    openai_base = OPENAI_BASE_URL
    anthropic_base = ANTHROPIC_BASE_URL

    # Susun fallback umum
    if is_responses_model(model):
        # Responses primary
        candidates.append(ProviderSpec("opencode_responses", opencode_base, "/zen/v1/responses", "openai_responses"))
        candidates.append(ProviderSpec("openai_responses", openai_base, "/v1/responses", "openai_responses"))
        candidates.append(ProviderSpec("opencode_chat", opencode_base, "/zen/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("openai_chat", openai_base, "/v1/chat/completions", "openai_chat"))
        # anthropic jarang untuk responses, tapi tambahkan
        candidates.append(ProviderSpec("anthropic", anthropic_base, "/v1/messages", "anthropic"))
    elif is_anthropic_model(model):
        candidates.append(ProviderSpec("anthropic", anthropic_base, "/v1/messages", "anthropic"))
        candidates.append(ProviderSpec("opencode_chat", opencode_base, "/zen/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("openai_chat", openai_base, "/v1/chat/completions", "openai_chat"))
    else:
        # default openai_chat via opencode
        candidates.append(ProviderSpec("opencode_chat", opencode_base, "/zen/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("openai_chat", openai_base, "/v1/chat/completions", "openai_chat"))
        candidates.append(ProviderSpec("opencode_responses", opencode_base, "/zen/v1/responses", "openai_responses"))
        candidates.append(ProviderSpec("anthropic", anthropic_base, "/v1/messages", "anthropic"))

    # deduplikasi berdasarkan url+sdk
    seen = set()
    uniq: List[ProviderSpec] = []
    for c in candidates:
        key = (c.url, c.sdk)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def get_headers_for_provider(provider: ProviderSpec, session_id: Optional[str] = None) -> Dict[str, str]:
    # header dipilih berdasarkan base_url (lebih reliabel daripada name, terutama untuk saved state)
    if "opencode.ai" in provider.base_url:
        return _opencode_headers(session_id)
    if provider.sdk == "anthropic" or "anthropic.com" in provider.base_url:
        return _anthropic_headers()
    # openai
    return _openai_headers()

def prepare_payload_for_provider(provider: ProviderSpec, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], extra_body: Optional[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
    """Konversi payload OpenAI messages/tools ke format provider, history tetap OpenAI."""
    sdk = provider.sdk
    if sdk in ("openai_chat", "openai_responses"):
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if extra_body:
            body.update(extra_body)
        # Responses transformation (dari opencode.js)
        if sdk == "openai_responses":
            if "max_output_tokens" not in body:
                if "max_completion_tokens" in body:
                    body["max_output_tokens"] = body["max_completion_tokens"]
                elif "max_tokens" in body:
                    body["max_output_tokens"] = body["max_tokens"]
            body.pop("max_tokens", None)
            body.pop("max_completion_tokens", None)
            if "reasoning_effort" in body:
                body["reasoning"] = {
                    "effort": str(body.pop("reasoning_effort")).lower().strip(),
                    "summary": "auto"
                }
        return body
    elif sdk == "anthropic":
        system, anth_msgs = openai_messages_to_anthropic(messages)
        anth_tools = openai_tools_to_anthropic(tools)
        body: Dict[str, Any] = {
            "model": model,
            "messages": anth_msgs,
            "stream": stream,
            "max_tokens": (extra_body or {}).get("max_tokens") or (extra_body or {}).get("max_output_tokens") or 4096,
        }
        if system:
            body["system"] = system
        if anth_tools:
            body["tools"] = anth_tools
        # mapping thinking: anthropic thinking beda, tapi kita pass reasoning_effort?
        # untuk sekarang ignore extra_body thinking untuk anthropic, tapi bisa map ke thinking field
        if extra_body:
            # anthropic tidak pakai reasoning_effort, tapi cek jika ada thinking
            if "reasoning_effort" in extra_body:
                # anthropic beta thinking: {"type":"thinking","budget_tokens":...} - skip
                pass
            # pass through other keys like temperature
            for k, v in extra_body.items():
                if k not in ("reasoning_effort", "reasoning", "max_tokens", "max_output_tokens"):
                    body[k] = v
        return body
    else:
        raise ValueError(f"Unknown sdk {sdk}")

def parse_nonstream_response(provider: ProviderSpec, resp_json: Dict[str, Any]) -> Dict[str, Any]:
    sdk = provider.sdk
    if sdk == "openai_chat":
        msg = resp_json.get("choices", [{}])[0].get("message", {})
        return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls"), "finish_reason": resp_json.get("choices", [{}])[0].get("finish_reason", "stop")}
    elif sdk == "openai_responses":
        # coba extract responses text
        # responses format: {output: [{content:[{text}] }]} atau output_text
        try:
            if "output" in resp_json:
                out = resp_json["output"]
                if isinstance(out, list):
                    for item in out:
                        if isinstance(item, dict) and "content" in item:
                            for c in item["content"]:
                                if c.get("text"):
                                    return {"content": c["text"], "tool_calls": None, "finish_reason": "stop"}
            if "output_text" in resp_json:
                return {"content": resp_json["output_text"], "tool_calls": None, "finish_reason": "stop"}
        except: pass
        # fallback ke choices
        msg = resp_json.get("choices", [{}])[0].get("message", {})
        if msg:
            return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls"), "finish_reason": resp_json.get("choices", [{}])[0].get("finish_reason", "stop")}
        # jika masih tidak, coba delta?
        return {"content": "", "tool_calls": None, "finish_reason": "stop"}
    elif sdk == "anthropic":
        # Anthropic non-stream: {content: [{type,text}...], stop_reason}
        content_blocks = resp_json.get("content", [])
        stop_reason = resp_json.get("stop_reason")
        return anthropic_response_to_openai(content_blocks, stop_reason)
    else:
        raise ValueError(f"Unknown sdk {sdk}")

# Streaming helpers
def is_streaming_error_parsable_as_fallback(exc: Exception, body_text: str = "") -> bool:
    """Tentukan apakah error layak fallback ke SDK lain."""
    txt = body_text.lower()
    # 404, 405, 422, 400 sering karena endpoint salah
    if isinstance(exc, Exception) and hasattr(exc, 'code'):
        code = getattr(exc, 'code', 0)
        if code in (404, 405, 422, 415):
            return True
        if code == 400 and any(k in txt for k in ("endpoint", "not found", "unsupported", "invalid", "unknown", "responses", "messages", "chat.completions")):
            return True
    if any(k in txt for k in ("unknown endpoint", "not found", "unsupported model", "invalid request: responses", "anthropic", "messages")):
        return True
    return False
