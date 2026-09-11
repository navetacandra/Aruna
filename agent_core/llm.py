"""LLM client untuk multi-provider fallback - tanpa external deps, streaming SSE, tool calling.
Mendukung OpenAI Chat/Responses dan Anthropic Messages dengan konversi otomatis.
History disimpan tetap format OpenAI, request dikonversi sesuai provider.
State provider per model disimpan di .agent/llm_provider_state.json dan akan rubah jika fallback terjadi.
Mengacu pada opencode.js: zen/v1/chat/completions & zen/v1/responses + Anthropic v1/messages
"""
import json
import re
import time
import uuid
import urllib.request
import urllib.error
import sys
from typing import Any, Callable, Dict, List, Optional

from .config import OPENCODE_BASE_URL, OPENCODE_UA
from .providers import (
    build_candidate_providers,
    get_headers_for_provider,
    prepare_payload_for_provider,
    parse_nonstream_response,
    update_provider_state,
    base_model_id,
    is_responses_model,
)

# --- Helpers port dari opencode.js (tetap diekspor untuk kompatibilitas) ---

def generate_request_id() -> str:
    return f"msg_{uuid.uuid4().hex}"

def generate_session_id() -> str:
    return f"ses_{uuid.uuid4().hex}"

# re-ekspor untuk import lama
__all__ = ["LLMClient", "fetch_models", "generate_request_id", "generate_session_id", "base_model_id", "is_responses_model"]

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

    # Backward compat: _prepare_body tetap ada tapi delegasi ke providers
    def _prepare_body(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
                      extra_body: Optional[Dict[str, Any]] = None, stream: bool = True) -> Dict[str, Any]:
        # Untuk kompatibilitas, gunakan opencode_chat logic
        from .providers import prepare_payload_for_provider, ProviderSpec
        # tebak provider dari model
        is_resp = is_responses_model(self.model)
        sdk = "openai_responses" if is_resp else "openai_chat"
        dummy = ProviderSpec("compat", self.base_url, "/zen/v1/chat/completions" if sdk=="openai_chat" else "/zen/v1/responses", sdk)
        return prepare_payload_for_provider(dummy, self.model, messages, tools, extra_body, stream)

    def _headers(self) -> Dict[str, str]:
        # Untuk kompatibilitas, return opencode headers
        from .providers import _opencode_headers
        return _opencode_headers(self.session_id)

    def chat(self,
             messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             extra_body: Optional[Dict[str, Any]] = None,
             stream: bool = True,
             on_delta: Optional[Callable[[str], None]] = None,
             on_tool_delta: Optional[Callable[[str], None]] = None,
             on_reasoning_delta: Optional[Callable[[str], None]] = None,
             timeout: int = 120) -> Dict[str, Any]:
        """Kirim chat dengan fallback multi-provider.
        History tetap OpenAI format, payload otomatis dikonversi per provider.
        State format/url per model disimpan dan akan rubah jika fallback terjadi.
        Returns dict: {content: str, tool_calls: list|None, finish_reason: str}
        """
        candidates = build_candidate_providers(self.model, self.base_url)
        last_exc: Optional[Exception] = None

        for provider in candidates:
            # Untuk 429, retry same provider hingga 5 kali dengan backoff 5 + (n-1)*3
            # Jika 429, jangan fallback ke SDK lain dulu, cek dulu karena bukan kesalahan format
            # Escape (Ctrl-C / ESC) membatalkan response - jangan retry, langsung batal
            for attempt in range(1, 6):
                try:
                    result = self._chat_with_provider(provider, messages, tools, extra_body, stream, on_delta, on_tool_delta, on_reasoning_delta, timeout)
                    # simpan state sukses untuk model yang sama (akan dipakai lagi) - tanpa log mengganggu
                    try:
                        update_provider_state(self.model, provider.sdk, provider.endpoint, provider.base_url)
                    except:
                        pass
                    return result
                except KeyboardInterrupt:
                    print("\n[escape] response dibatalkan (Ctrl-C / ESC)", file=sys.stderr)
                    raise
                except urllib.error.HTTPError as e:
                    body_txt = ""
                    try:
                        if e.fp:
                            body_txt = e.read().decode("utf-8", errors="ignore")
                    except:
                        pass
                    msg = f"HTTP {e.code}: {body_txt[:500]}"
                    if e.code == 429:
                        # 429 bukan kesalahan format, jangan fallback dulu, retry same provider
                        if attempt < 5:
                            wait = 5 + (attempt - 1) * 3  # 5, 8, 11, 14
                            # Format sesuai instruksi: "{error_message} {n} Retry on {x} seconds.."
                            err_msg = body_txt.strip()[:200] if body_txt.strip() else msg
                            print(f"{err_msg} {attempt} Retry on {wait} seconds..", file=sys.stderr)
                            try:
                                time.sleep(wait)
                            except KeyboardInterrupt:
                                raise RuntimeError(f"Interrupted during 429 retry for {provider.name}") from e
                            continue  # retry same provider
                        else:
                            # sudah 5 kali, fallback ke SDK lain
                            _wrapped = RuntimeError(f"Provider {provider.name} {provider.url} failed after 5 retries {msg}")
                            _wrapped.__cause__ = e
                            last_exc = _wrapped
                            print(f"[429] {provider.name} {provider.url} gagal setelah 5 retries {e.code}, fallback ke SDK lain...", file=sys.stderr)
                            break  # break inner retry, lanjut ke provider berikutnya
                    # Semua HTTP error lain langsung fallback ke provider lain
                    _wrapped = RuntimeError(f"Provider {provider.name} {provider.url} failed {msg}")
                    _wrapped.__cause__ = e
                    last_exc = _wrapped
                    print(f"[provider fallback] {provider.name} {provider.url} gagal {e.code}, coba fallback...", file=sys.stderr)
                    break  # break inner retry, lanjut ke provider berikutnya
                except Exception as e:
                    # Jika binary file dan bukan Response -> langsung tolak tanpa fallback
                    if "Model tidak didukung" in str(e):
                        raise RuntimeError("Model tidak didukung") from e
                    last_exc = e
                    print(f"[provider fallback] {provider.name} {provider.url} error: {e}, coba fallback...", file=sys.stderr)
                    break  # non-HTTP error, fallback

        # semua gagal
        if last_exc:
            raise RuntimeError(f"All providers failed for {self.model}: {last_exc}") from last_exc
        raise RuntimeError("All providers failed")

    def _chat_with_provider(self, provider, messages, tools, extra_body, stream, on_delta, on_tool_delta, on_reasoning_delta, timeout):
        body = prepare_payload_for_provider(provider, self.model, messages, tools, extra_body, stream)
        headers = get_headers_for_provider(provider, self.session_id)
        # Request body tetap raw JSON (server opencode tidak support Content-Encoding: gzip untuk request, percobaan menghasilkan 401)
        # Hanya minta compressed response via Accept-Encoding (hemat Down bandwidth 99% - benchmark)
        # Default brotli q4 untuk response - hemat +35% vs gzip
        try:
            import brotli
            headers["Accept-Encoding"] = "br, gzip"
        except ImportError:
            headers["Accept-Encoding"] = "gzip"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(provider.url, data=data, headers=headers, method="POST")

        # Non-stream
        if not stream:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    enc = ""
                    try:
                        enc = resp.headers.get("Content-Encoding", "") if hasattr(resp, "headers") and resp.headers else ""
                    except: pass
                    if enc == "br":
                        try:
                            import brotli
                            raw = brotli.decompress(raw)
                        except: pass
                    elif enc == "gzip":
                        try:
                            import gzip
                            raw = gzip.decompress(raw)
                        except: pass
                    j = json.loads(raw.decode("utf-8"))
                    return parse_nonstream_response(provider, j)
            except urllib.error.HTTPError:
                raise
            except Exception as e:
                raise RuntimeError(f"Non-stream error: {e}") from e

        # Streaming
        if provider.sdk == "anthropic":
            return self._stream_anthropic(req, timeout, on_delta, on_tool_delta, on_reasoning_delta)
        else:
            return self._stream_openai(provider, req, timeout, on_delta, on_tool_delta, on_reasoning_delta)

    def _stream_openai(self, provider, req, timeout, on_delta, on_tool_delta, on_reasoning_delta=None):
        content_parts: List[str] = []
        tool_accum: Dict[int, Dict[str, str]] = {}
        finish_reason: Optional[str] = None

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Handle decompression for streaming (brotli/gzip) - default brotli q4
                enc = ""
                try:
                    enc = resp.headers.get("Content-Encoding", "") if hasattr(resp, "headers") and resp.headers else ""
                except: pass
                decompressor = None
                if enc == "br":
                    try:
                        import brotli
                        decompressor = brotli.Decompressor()
                    except: decompressor = None
                elif enc == "gzip":
                    try:
                        import zlib
                        decompressor = zlib.decompressobj(31)
                    except: decompressor = None
                buffer = ""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    if decompressor:
                        try:
                            if enc == "br":
                                chunk = decompressor.decompress(chunk)
                            else:
                                chunk = decompressor.decompress(chunk)
                            if not chunk:
                                continue
                        except:
                            pass
                    buffer += chunk.decode("utf-8", errors="ignore")
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        trimmed = line.strip()
                        # Anthropic kadang kirim event: line, tapi untuk openai hanya data:
                        if not trimmed or trimmed.startswith("event:"):
                            continue
                        if not trimmed.startswith("data:"):
                            continue
                        data_str = trimmed[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Responses API delta - sesuai SDK-example.md sec 2
                        handled = False
                        ptype = parsed.get("type", "")
                        if isinstance(ptype, str) and ptype.startswith("response."):
                            # Reasoning - saat LLM melakukan reasoning
                            if "reasoning" in ptype:
                                d = parsed.get("delta", "") or parsed.get("text", "") or parsed.get("summary", "") or "reasoning"
                                if on_reasoning_delta:
                                    on_reasoning_delta(d if isinstance(d, str) else json.dumps(d))
                                handled = True
                            elif "output_text.delta" in ptype:
                                d = parsed.get("delta", "") or parsed.get("text", "")
                                if d:
                                    content_parts.append(d)
                                    if on_delta:
                                        on_delta(d)
                                handled = True
                            elif ptype == "response.output_item.added":
                                item = parsed.get("item", {})
                                if item.get("type") == "reasoning":
                                    if on_reasoning_delta:
                                        on_reasoning_delta(item.get("summary", "") or "reasoning")
                                    handled = True
                                elif item.get("type") == "function_call":
                                    idx = parsed.get("output_index", 0)
                                    if idx not in tool_accum:
                                        tool_accum[idx] = {"id": item.get("call_id") or item.get("id") or "", "name": item.get("name") or "", "arguments": item.get("arguments") or ""}
                                    else:
                                        if item.get("call_id"):
                                            tool_accum[idx]["id"] = item["call_id"]
                                        if item.get("name"):
                                            tool_accum[idx]["name"] = item["name"]
                                        if item.get("arguments"):
                                            tool_accum[idx]["arguments"] = item["arguments"]
                                    handled = True
                                else:
                                    handled = True
                            elif ptype == "response.function_call_arguments.delta":
                                idx = parsed.get("output_index", 0)
                                delta = parsed.get("delta", "")
                                if idx not in tool_accum:
                                    tool_accum[idx] = {"id": parsed.get("item_id") or "", "name": "", "arguments": delta}
                                else:
                                    tool_accum[idx]["arguments"] += delta
                                if on_tool_delta:
                                    on_tool_delta(delta)
                                handled = True
                            elif ptype == "response.function_call_arguments.done":
                                idx = parsed.get("output_index", 0)
                                args = parsed.get("arguments", "")
                                if idx in tool_accum:
                                    if args and args != tool_accum[idx]["arguments"]:
                                        tool_accum[idx]["arguments"] = args
                                    if parsed.get("name"):
                                        tool_accum[idx]["name"] = parsed["name"]
                                else:
                                    tool_accum[idx] = {"id": parsed.get("item_id") or "", "name": parsed.get("name") or "", "arguments": args}
                                handled = True
                            elif ptype in ("response.output_item.done", "response.content_part.added", "response.content_part.done", "response.created", "response.in_progress"):
                                handled = True
                            elif "function_call" in ptype:
                                handled = True
                            if ptype in ("response.completed", "response.done"):
                                finish_reason = "stop"
                                handled = True
                            if handled:
                                continue

                        choices = parsed.get("choices")
                        if not choices:
                            if "delta" in parsed and isinstance(parsed["delta"], str):
                                content_parts.append(parsed["delta"])
                                if on_delta:
                                    on_delta(parsed["delta"])
                            continue

                        choice = choices[0] if choices else {}
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta", {}) or {}
                        if delta.get("content"):
                            c = delta["content"]
                            content_parts.append(c)
                            if on_delta:
                                on_delta(c)
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
                        msg_tc = choice.get("message", {}).get("tool_calls") if "message" in choice else None
                        if msg_tc:
                            for i, tc in enumerate(msg_tc):
                                tool_accum[i] = {
                                    "id": tc.get("id", f"call_{i}"),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", "") if isinstance(tc.get("function", {}).get("arguments"), str) else json.dumps(tc.get("function", {}).get("arguments", {}))
                                }

                    if finish_reason == "stop" and data_str == "[DONE]":
                        break
                if buffer.strip().startswith("data:"):
                    data_str = buffer.strip()[len("data:"):].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            parsed = json.loads(data_str)
                            choices = parsed.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                if delta.get("content"):
                                    content_parts.append(delta["content"])
                                    if on_delta:
                                        on_delta(delta["content"])
                        except:
                            pass

        except urllib.error.HTTPError:
            raise
        except Exception as e:
            if content_parts or tool_accum:
                pass
            else:
                raise

        tool_calls = None
        if tool_accum:
            tool_calls = []
            for idx in sorted(tool_accum.keys()):
                acc = tool_accum[idx]
                if not acc["name"] and not acc["arguments"]:
                    continue
                args = acc["arguments"] or "{}"
                tool_calls.append({
                    "id": acc["id"] or f"call_{idx}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": args}
                })

        return {"content": "".join(content_parts), "tool_calls": tool_calls, "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop")}

    def _stream_anthropic(self, req, timeout, on_delta, on_tool_delta, on_reasoning_delta=None):
        """Streaming khusus Anthropic SSE (event: + data:) dinormalisasi ke OpenAI."""
        content_parts: List[str] = []
        # anthropic tool_use: index -> {id, name, input_json}
        tool_accum: Dict[int, Dict[str, Any]] = {}
        # mapping content_block index -> type
        block_types: Dict[int, str] = {}
        # untuk tool delta, simpan partial_json
        finish_reason = None
        current_event = None

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                enc = ""
                try:
                    enc = resp.headers.get("Content-Encoding", "") if hasattr(resp, "headers") and resp.headers else ""
                except: pass
                decompressor = None
                if enc == "br":
                    try:
                        import brotli
                        decompressor = brotli.Decompressor()
                    except: decompressor = None
                elif enc == "gzip":
                    try:
                        import zlib
                        decompressor = zlib.decompressobj(31)
                    except: decompressor = None
                buffer = ""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    if decompressor:
                        try:
                            if enc == "br":
                                chunk = decompressor.decompress(chunk)
                            else:
                                chunk = decompressor.decompress(chunk)
                            if not chunk:
                                continue
                        except:
                            pass
                    buffer += chunk.decode("utf-8", errors="ignore")
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed:
                            continue
                        if trimmed.startswith("event:"):
                            current_event = trimmed[len("event:"):].strip()
                            continue
                        if not trimmed.startswith("data:"):
                            continue
                        data_str = trimmed[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                        except:
                            continue

                        ptype = parsed.get("type", "")

                        # Anthropic events - thinking = reasoning
                        if ptype == "content_block_start":
                            idx = parsed.get("index", 0)
                            block = parsed.get("content_block", {})
                            btype = block.get("type", "")
                            block_types[idx] = btype
                            if btype == "tool_use":
                                # init tool
                                tool_accum[idx] = {"id": block.get("id",""), "name": block.get("name",""), "input_json": ""}
                            elif btype == "thinking":
                                if on_reasoning_delta:
                                    on_reasoning_delta(block.get("thinking", "") or "thinking")
                        elif ptype == "content_block_delta":
                            idx = parsed.get("index", 0)
                            delta = parsed.get("delta", {})
                            dtype = delta.get("type", "")
                            btype = block_types.get(idx, "")
                            if dtype == "thinking_delta" or btype == "thinking":
                                txt = delta.get("thinking", "") or delta.get("text", "")
                                if on_reasoning_delta:
                                    on_reasoning_delta(txt or "thinking")
                            elif dtype == "text_delta" or (btype == "text" and "text" in delta):
                                txt = delta.get("text", "")
                                if txt:
                                    content_parts.append(txt)
                                    if on_delta:
                                        on_delta(txt)
                            elif dtype == "input_json_delta":
                                pj = delta.get("partial_json", "")
                                if idx in tool_accum:
                                    tool_accum[idx]["input_json"] += pj
                                    if on_tool_delta:
                                        on_tool_delta(pj)
                                else:
                                    # fallback: jika belum ada, buat entry
                                    if idx not in tool_accum:
                                        tool_accum[idx] = {"id": f"toolu_{idx}", "name": "", "input_json": pj}
                                    else:
                                        tool_accum[idx]["input_json"] += pj
                        elif ptype == "content_block_stop":
                            pass
                        elif ptype == "message_delta":
                            delta = parsed.get("delta", {})
                            if delta.get("stop_reason"):
                                sr = delta["stop_reason"]
                                if sr == "tool_use":
                                    finish_reason = "tool_calls"
                                elif sr in ("end_turn", "stop"):
                                    finish_reason = "stop"
                                else:
                                    finish_reason = sr
                        elif ptype == "message_start":
                            pass
                        elif ptype == "message_stop":
                            finish_reason = finish_reason or "stop"
                            break
                        else:
                            # Fallback: jika ada text delta langsung
                            if "delta" in parsed and isinstance(parsed["delta"], dict) and "text" in parsed["delta"]:
                                txt = parsed["delta"]["text"]
                                content_parts.append(txt)
                                if on_delta:
                                    on_delta(txt)

                    # anthropic biasanya tidak pakai [DONE], tapi event message_stop
                    if finish_reason and current_event == "message_stop":
                        break

                # handle leftover
                if buffer.strip().startswith("data:"):
                    try:
                        data_str = buffer.strip()[len("data:"):].strip()
                        if data_str and data_str != "[DONE]":
                            parsed = json.loads(data_str)
                            # cek text
                            if parsed.get("type") == "content_block_delta":
                                delta = parsed.get("delta", {})
                                if delta.get("text"):
                                    content_parts.append(delta["text"])
                                    if on_delta:
                                        on_delta(delta["text"])
                    except:
                        pass

        except urllib.error.HTTPError:
            raise
        except Exception as e:
            if content_parts or tool_accum:
                pass
            else:
                raise

        # build tool_calls
        tool_calls = None
        if tool_accum:
            tool_calls = []
            for idx in sorted(tool_accum.keys()):
                acc = tool_accum[idx]
                # skip jika bukan tool_use (text blocks tidak di tool_accum)
                if not acc.get("name"):
                    continue
                # input_json mungkin tidak lengkap? coba parse
                raw = acc.get("input_json", "{}")
                if not raw.strip():
                    raw = "{}"
                # validasi JSON, jika gagal biarkan raw
                try:
                    json.loads(raw)
                    args = raw
                except:
                    args = raw  # biarkan, nanti execute_tool akan error jika invalid
                tool_calls.append({
                    "id": acc.get("id") or f"call_{idx}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": args}
                })

        return {"content": "".join(content_parts), "tool_calls": tool_calls, "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop")}


def _extract_responses_text(j: Dict[str, Any]) -> Optional[str]:
    try:
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
