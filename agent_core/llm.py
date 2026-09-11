"""LLM client untuk OpenCode API - tanpa external deps, streaming SSE, tool calling.
Mengacu pada opencode.js: zen/v1/chat/completions & zen/v1/responses
"""
import json
import re
import uuid
import urllib.request
import urllib.error
from typing import Any, Callable, Dict, List, Optional

from .config import OPENCODE_BASE_URL, OPENCODE_UA, RESPONSES_MODELS

# --- Helpers port dari opencode.js ---

def generate_request_id() -> str:
    return f"msg_{uuid.uuid4().hex}"

def generate_session_id() -> str:
    return f"ses_{uuid.uuid4().hex}"

def base_model_id(model: str) -> str:
    return re.sub(r"\([^()]+\)\s*$", "", str(model or "")).strip()

def is_responses_model(model: str) -> bool:
    base = base_model_id(model)
    return base in RESPONSES_MODELS or "muse-spark" in base

def fetch_models(base_url: str = OPENCODE_BASE_URL) -> List[Dict[str, Any]]:
    url = f"{base_url}/zen/v1/models"
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer public",
        "User-Agent": OPENCODE_UA,
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        raise RuntimeError(f"Failed to fetch models: {e.code} {body}") from e


class LLMClient:
    def __init__(self, model: str, base_url: str = OPENCODE_BASE_URL, session_id: Optional[str] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or generate_session_id()

    def _prepare_body(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
                      extra_body: Optional[Dict[str, Any]] = None, stream: bool = True) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if extra_body:
            body.update(extra_body)

        # Transformasi untuk Responses models (port dari opencode.js:68-79)
        if is_responses_model(self.model):
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

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer public",
            "User-Agent": OPENCODE_UA,
            "x-opencode-client": "desktop",
            "x-opencode-session": self.session_id,
            "x-opencode-request": generate_request_id(),
            "x-opencode-project": "global",
            "Accept": "text/event-stream",
        }

    def chat(self,
             messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             extra_body: Optional[Dict[str, Any]] = None,
             stream: bool = True,
             on_delta: Optional[Callable[[str], None]] = None,
             on_tool_delta: Optional[Callable[[str], None]] = None,
             timeout: int = 120) -> Dict[str, Any]:
        """Kirim chat, dukung streaming + tool calling.
        Returns dict: {content: str, tool_calls: list|None, raw_finish_reason: str}
        Jika stream=True, on_delta dipanggil per token text.
        """
        body = self._prepare_body(messages, tools, extra_body, stream=stream)
        is_resp = is_responses_model(self.model)
        endpoint = "/zen/v1/responses" if is_resp else "/zen/v1/chat/completions"
        url = self.base_url + endpoint

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        # Non-stream: satu request biasa
        if not stream:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    j = json.loads(resp.read().decode("utf-8"))
                    # Normalisasi ke format OpenAI chat
                    if is_resp:
                        # responses API non-stream: coba extract text
                        # format: {output: [{content:[{text:"..."}]}]} atau choices
                        text = _extract_responses_text(j)
                        if text is not None:
                            return {"content": text, "tool_calls": None, "finish_reason": "stop"}
                    msg = j.get("choices", [{}])[0].get("message", {})
                    return {
                        "content": msg.get("content") or "",
                        "tool_calls": msg.get("tool_calls"),
                        "finish_reason": j.get("choices", [{}])[0].get("finish_reason", "stop")
                    }
            except urllib.error.HTTPError as e:
                body_txt = e.read().decode("utf-8", errors="ignore") if e.fp else ""
                raise RuntimeError(f"OpenCode API error {e.code}: {body_txt}") from e

        # Streaming SSE
        content_parts: List[str] = []
        # tool_calls accumulator: {index: {id, name, arguments}}
        tool_accum: Dict[int, Dict[str, str]] = {}
        finish_reason: Optional[str] = None

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                buffer = ""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="ignore")
                    lines = buffer.split("\n")
                    buffer = lines.pop()  # sisa incomplete

                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed or not trimmed.startswith("data:"):
                            continue
                        data_str = trimmed[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # --- Coba parse format responses API ---
                        # responses delta: {"type":"response.output_text.delta","delta":"..."}
                        # atau {"type":"response.function_call_arguments.delta", ...}
                        # Kita fallback ke choices delta jika tidak cocok
                        handled = False
                        ptype = parsed.get("type", "")
                        if isinstance(ptype, str) and ptype.startswith("response."):
                            if "output_text.delta" in ptype:
                                d = parsed.get("delta", "") or parsed.get("text", "")
                                if d:
                                    content_parts.append(d)
                                    if on_delta:
                                        on_delta(d)
                                handled = True
                            elif "function_call" in ptype or "tool" in ptype.lower():
                                # tidak ada spec pasti, lewati
                                pass
                            if ptype in ("response.completed", "response.done"):
                                finish_reason = "stop"
                                handled = True
                            if handled:
                                continue

                        # --- Format OpenAI chat.completions delta ---
                        choices = parsed.get("choices")
                        if not choices:
                            # beberapa responses juga bungkus di output
                            # coba _extract_responses_text delta?
                            # fallback: jika ada 'delta' top-level
                            if "delta" in parsed and isinstance(parsed["delta"], str):
                                content_parts.append(parsed["delta"])
                                if on_delta:
                                    on_delta(parsed["delta"])
                            continue

                        choice = choices[0] if choices else {}
                        # finish_reason
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                        delta = choice.get("delta", {}) or {}
                        # content
                        if delta.get("content"):
                            c = delta["content"]
                            content_parts.append(c)
                            if on_delta:
                                on_delta(c)

                        # tool_calls delta (streaming)
                        tcs = delta.get("tool_calls")
                        if tcs:
                            for tc in tcs:
                                idx = tc.get("index", 0)
                                if idx not in tool_accum:
                                    tool_accum[idx] = {"id": "", "name": "", "arguments": ""}
                                if tc.get("id"):
                                    tool_accum[idx]["id"] = tc["id"]
                                if tc.get("function"):
                                    fn = tc["function"]
                                    if fn.get("name"):
                                        tool_accum[idx]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        arg_delta = fn["arguments"]
                                        tool_accum[idx]["arguments"] += arg_delta
                                        if on_tool_delta:
                                            on_tool_delta(arg_delta)
                        # juga handle message.tool_calls non-stream chunk (kadang tanpa delta)
                        msg_tc = choice.get("message", {}).get("tool_calls") if "message" in choice else None
                        if msg_tc:
                            # non-stream batch
                            for i, tc in enumerate(msg_tc):
                                tool_accum[i] = {
                                    "id": tc.get("id", f"call_{i}"),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", "") if isinstance(tc.get("function", {}).get("arguments"), str) else json.dumps(tc.get("function", {}).get("arguments", {}))
                                }

                    if finish_reason == "stop" and data_str == "[DONE]":
                        break
                # handle leftover buffer line yang belum diproses (tanpa \n)
                if buffer.strip().startswith("data:"):
                    data_str = buffer.strip()[len("data:"):].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            parsed = json.loads(data_str)
                            # sama seperti di atas, coba extract content
                            choices = parsed.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                if delta.get("content"):
                                    content_parts.append(delta["content"])
                                    if on_delta:
                                        on_delta(delta["content"])
                        except:
                            pass

        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", errors="ignore") if e.fp else ""
            raise RuntimeError(f"OpenCode API error {e.code}: {body_txt}") from e
        except Exception as e:
            # Jika streaming gagal mid-way, kembalikan apa yang sudah terkumpul
            if content_parts or tool_accum:
                pass
            else:
                raise

        # Build tool_calls result
        tool_calls: Optional[List[Dict[str, Any]]] = None
        if tool_accum:
            tool_calls = []
            for idx in sorted(tool_accum.keys()):
                acc = tool_accum[idx]
                # skip empty
                if not acc["name"] and not acc["arguments"]:
                    continue
                # validasi arguments JSON, jika gagal biarkan string mentah
                args = acc["arguments"] or "{}"
                tool_calls.append({
                    "id": acc["id"] or f"call_{idx}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": acc["name"],
                        "arguments": args
                    }
                })

        return {
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
            "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop")
        }


def _extract_responses_text(j: Dict[str, Any]) -> Optional[str]:
    try:
        # coba beberapa path
        if "output" in j:
            out = j["output"]
            if isinstance(out, list):
                for item in out:
                    if isinstance(item, dict) and "content" in item:
                        for c in item["content"]:
                            if c.get("text"):
                                return c["text"]
                            if c.get("type") == "output_text" and c.get("text"):
                                return c["text"]
        if "choices" in j:
            return j["choices"][0].get("message", {}).get("content")
        if "output_text" in j:
            return j["output_text"]
    except:
        pass
    return None
