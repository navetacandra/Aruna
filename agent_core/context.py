"""Context Manager - krusial untuk hasil bagus.
- Estimasi token tanpa external deps (chars/4 heuristic)
- Kompaksi: keep system + recent, older di-summarize/truncate
- Truncate tool output besar
"""
import json
from typing import Any, Dict, List

from .config import COMPACTION_THRESHOLD, KEEP_RECENT_MESSAGES, MAX_CONTEXT_TOKENS, MAX_TOOL_OUTPUT_CHARS

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # heuristic: ~4 chars per token, fallback ke word count
    return max(len(text) // 4, len(text.split()) // 2 + 1) + 2

def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        # content bisa string atau list/pilihan
        c = m.get("content")
        if isinstance(c, str):
            total += estimate_tokens(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(part["text"])
                else:
                    total += estimate_tokens(str(part))
        elif c is not None:
            total += estimate_tokens(str(c))
        # tool_calls juga hitung
        tcs = m.get("tool_calls")
        if tcs:
            total += estimate_tokens(json.dumps(tcs, ensure_ascii=False))
        # overhead per message
        total += 8
    return total


class ContextManager:
    """Kelola percakapan + kompaksi otomatis."""
    def __init__(self, system_prompt: str, max_tokens: int = MAX_CONTEXT_TOKENS,
                 keep_recent: int = KEEP_RECENT_MESSAGES):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._compaction_count = 0

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: List[Dict[str, Any]] = None):
        msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, name: str, content: str):
        # truncate tool output besar
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = content[:MAX_TOOL_OUTPUT_CHARS // 2] + f"\n...[TRUNCATED {len(content)-MAX_TOOL_OUTPUT_CHARS} chars]...\n" + content[-MAX_TOOL_OUTPUT_CHARS // 2:]
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        })

    # untuk kompatibilitas format lama: add_tool dengan (id, content)
    def add_tool(self, tool_call_id: str, content: str, name: str = "tool"):
        self.add_tool_result(tool_call_id, name, content)

    def get_messages(self, force_compact: bool = False) -> List[Dict[str, Any]]:
        tokens = estimate_messages_tokens(self.messages)
        threshold = int(self.max_tokens * COMPACTION_THRESHOLD)
        # iterative compaction sampai di bawah threshold
        attempts = 0
        while (force_compact or tokens > threshold) and attempts < 5:
            before = len(self.messages)
            did = self._compact(tokens, threshold)
            if not did:
                break
            force_compact = False  # hanya paksa sekali
            tokens = estimate_messages_tokens(self.messages)
            # jika masih di atas threshold dan masih banyak pesan, kurangi keep_recent dan coba lagi
            if tokens > threshold and len(self.messages) > self.keep_recent + 2:
                # perkecil keep_recent untuk percobaan berikutnya (simulasi adaptive)
                # _compact berikutnya akan keep lebih sedikit karena kita tidak ubah keep_recent permanen,
                # tapi kita bisa langsung potong pesan recent tertua
                if tokens > self.max_tokens * 0.95:
                    # emergency: potong pesan tengah yang masih besar (keep hanya 2 terbaru)
                    # rebuild dengan 2 terbaru saja
                    system = self.messages[0]
                    # ambil 2 terbaru selain system
                    recent = self.messages[-2:]
                    # ringkas sisanya yang sekarang (di antara system dan recent)
                    middle = self.messages[1:-2]
                    if middle:
                        self._compaction_count += 1
                        summary = f"[Emergency compaction #{self._compaction_count}: {len(middle)} pesan dipotong karena masih >95% limit]"
                        self.messages = [system, {"role": "user", "content": summary}] + recent
                        tokens = estimate_messages_tokens(self.messages)
            attempts += 1
            if before == len(self.messages):
                break
        return self.messages

    def _compact(self, current_tokens: int, threshold: int) -> bool:
        """Strategi kompaksi: keep system + last N, middle diringkas jadi 1 message.
        Return True jika melakukan kompaksi.
        """
        if len(self.messages) <= self.keep_recent + 1:
            return False  # tidak ada yang bisa dikompaksi
        # selalu keep system
        system = self.messages[0]
        # keep recent N
        recent = self.messages[-self.keep_recent:]
        # middle yang akan diringkas
        middle = self.messages[1:-self.keep_recent]
        if not middle:
            return

        # Buat summary manual tanpa LLM (agar tanpa deps & cepat)
        # Hitung stats
        middle_tokens = estimate_messages_tokens(middle)
        summary_lines = [f"[Context compaction #{self._compaction_count+1}: {len(middle)} pesan lama (~{middle_tokens} tokens) diringkas]"]
        for m in middle:
            role = m.get("role", "?")
            # ringkas tiap message jadi 1 baris
            c = m.get("content", "")
            if isinstance(c, str):
                snippet = c[:400].replace("\n", " ").strip()
                if len(c) > 400:
                    snippet += f" ...(+{len(c)-400} chars)"
            else:
                snippet = str(c)[:400]
            # tool_calls
            if m.get("tool_calls"):
                tcs = m["tool_calls"]
                try:
                    tc_str = ", ".join([f"{tc['function']['name']}({tc['function']['arguments'][:80]})" for tc in tcs])
                except:
                    tc_str = f"{len(tcs)} tool_calls"
                snippet += f" | tool_calls: {tc_str}"
            if role == "tool":
                snippet = f"tool:{m.get('name','')} -> {snippet[:300]}"
            summary_lines.append(f"- {role}: {snippet}")
            if len("\n".join(summary_lines)) > 4000:
                summary_lines.append(f"...(+{len(middle)-len(summary_lines)+1} pesan lain dipotong)")
                break

        summary_text = "\n".join(summary_lines)
        self._compaction_count += 1

        # Rebuild messages: system + summary (sebagai user note) + recent
        summary_msg = {
            "role": "user",
            "content": f"[SYSTEM NOTE - CONTEXT COMPACTED]\n{summary_text}\n[End compaction - lanjutkan dengan konteks recent di bawah]"
        }
        self.messages = [system, summary_msg] + recent
        return True

    def token_usage(self) -> Dict[str, int]:
        t = estimate_messages_tokens(self.messages)
        return {"tokens": t, "max": self.max_tokens, "percent": int(t / self.max_tokens * 100) if self.max_tokens else 0, "compactions": self._compaction_count}

    def dump(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def load(self, messages: List[Dict[str, Any]]):
        """Load dari history (mis. resume session)."""
        if messages and messages[0].get("role") == "system":
            self.messages = messages
        else:
            # ensure system tetap di depan
            self.messages = [{"role": "system", "content": self.system_prompt}] + messages
