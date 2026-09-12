"""Context Manager - crucial for good results.
- Token estimation without external deps (chars/4 heuristic)
- Compaction: keep system + recent, older summarized/truncated
- Truncate large tool output
"""
import json
from typing import Any, Dict, List

from .config import COMPACTION_THRESHOLD, KEEP_RECENT_MESSAGES, MAX_CONTEXT_TOKENS, MAX_TOOL_OUTPUT_CHARS

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # heuristic: ~4 chars per token, fallback to word count
    return max(len(text) // 4, len(text.split()) // 2 + 1) + 2

def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        # content can be string or list/options
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
        # also count tool_calls
        tcs = m.get("tool_calls")
        if tcs:
            total += estimate_tokens(json.dumps(tcs, ensure_ascii=False))
        # overhead per message
        total += 8
    return total


class ContextManager:
    """Manage conversation + automatic compaction."""
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
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = content[:MAX_TOOL_OUTPUT_CHARS] + f"\n...[TRUNCATED {len(content)-MAX_TOOL_OUTPUT_CHARS} chars]...\n"
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        })

    # for backward compatibility with old format: add_tool with (id, content)
    def add_tool(self, tool_call_id: str, content: str, name: str = "tool"):
        self.add_tool_result(tool_call_id, name, content)

    def get_messages(self, force_compact: bool = False) -> List[Dict[str, Any]]:
        tokens = estimate_messages_tokens(self.messages)
        threshold = int(self.max_tokens * COMPACTION_THRESHOLD)
        # iterative compaction until below threshold
        attempts = 0
        while (force_compact or tokens > threshold) and attempts < 5:
            before = len(self.messages)
            did = self._compact(tokens, threshold)
            if not did:
                break
            force_compact = False  # only force once
            tokens = estimate_messages_tokens(self.messages)
            # if still above threshold and many messages remain, try more aggressive compaction
            if tokens > threshold and len(self.messages) > self.keep_recent + 2:
                if tokens > self.max_tokens * 0.95:
                    # emergency: keep fewer recent messages but still summarize middle with snippets
                    # keep at least 4 or keep_recent//2, not just 2
                    keep = max(4, self.keep_recent // 2)
                    if len(self.messages) > keep + 1:
                        system = self.messages[0]
                        recent = self.messages[-keep:]
                        middle = self.messages[1:-keep]
                        if middle:
                            # Create a summarized emergency compaction with snippets
                            middle_tokens = estimate_messages_tokens(middle)
                            self._compaction_count += 1
                            summary_lines = [f"[Emergency compaction #{self._compaction_count}: {len(middle)} messages (~{middle_tokens} tokens) summarized]"]
                            for m in middle[:8]:  # keep up to 8 snippets even in emergency
                                role = m.get("role", "?")
                                c = m.get("content", "")
                                snippet = c[:300].replace("\n", " ").strip() if isinstance(c, str) else str(c)[:300]
                                if isinstance(c, str) and len(c) > 300:
                                    snippet += f" ...(+{len(c)-300})"
                                if m.get("tool_calls"):
                                    try:
                                        tc_str = ", ".join([f"{tc['function']['name']}" for tc in m["tool_calls"]])
                                    except:
                                        tc_str = f"{len(m['tool_calls'])} tool_calls"
                                    snippet += f" | {tc_str}"
                                summary_lines.append(f"- {role}: {snippet}")
                            if len(middle) > 8:
                                summary_lines.append(f"...(+{len(middle)-8} more truncated)")
                            summary = "\n".join(summary_lines)
                            self.messages = [system, {"role": "user", "content": f"[EMERGENCY COMPACTED]\n{summary}"}] + recent
                            tokens = estimate_messages_tokens(self.messages)
            attempts += 1
            if before == len(self.messages):
                break
        return self.messages

    def _compact(self, current_tokens: int, threshold: int) -> bool:
        """Compaction strategy: keep system + last N, middle summarized into 1 message.
        Return True if compaction was performed.
        """
        if len(self.messages) <= self.keep_recent + 1:
            return False  # nothing to compact
        # always keep system
        system = self.messages[0]
        # keep recent N, but ensure we don't orphan tool call pairs
        # Check if the cut point splits a tool call sequence
        recent_start = len(self.messages) - self.keep_recent
        # If the message before recent is an assistant with tool_calls and recent starts with tool, keep the pair together
        if recent_start > 1:
            prev = self.messages[recent_start - 1]
            first_recent = self.messages[recent_start]
            # If previous is assistant with tool_calls and first recent is tool result, include previous in recent
            if prev.get("role") == "assistant" and prev.get("tool_calls") and first_recent.get("role") == "tool":
                recent = self.messages[recent_start - 1:]
                middle = self.messages[1:recent_start - 1]
            else:
                recent = self.messages[-self.keep_recent:]
                middle = self.messages[1:-self.keep_recent]
        else:
            recent = self.messages[-self.keep_recent:]
            middle = self.messages[1:-self.keep_recent]
        if not middle:
            return False

        # Create manual summary without LLM (to avoid deps & be fast)
        # Calculate stats
        middle_tokens = estimate_messages_tokens(middle)
        summary_lines = [f"[Context compaction #{self._compaction_count+1}: {len(middle)} old messages (~{middle_tokens} tokens) summarized]"]
        for m in middle:
            role = m.get("role", "?")
            # summarize each message into 1 line
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
                summary_lines.append(f"...(+{len(middle)-len(summary_lines)+1} other messages truncated)")
                break

        summary_text = "\n".join(summary_lines)
        self._compaction_count += 1

        # Rebuild messages: system + summary (as user note) + recent
        summary_msg = {
            "role": "user",
            "content": f"[SYSTEM NOTE - CONTEXT COMPACTED]\n{summary_text}\n[End compaction - continue with recent context below]"
        }
        self.messages = [system, summary_msg] + recent
        return True

    def token_usage(self) -> Dict[str, int]:
        t = estimate_messages_tokens(self.messages)
        return {"tokens": t, "max": self.max_tokens, "percent": int(t / self.max_tokens * 100) if self.max_tokens else 0, "compactions": self._compaction_count}

    def dump(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def load(self, messages: List[Dict[str, Any]]):
        """Load from history (e.g. resume session)."""
        if messages and messages[0].get("role") == "system":
            self.messages = messages
        else:
            # ensure system stays at front
            self.messages = [{"role": "system", "content": self.system_prompt}] + messages
