"""Agent Loop - ReAct iterative dengan tool calling kuat."""
import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from .config import MAX_ITERATIONS
from .context import ContextManager
from .history import save_message
from .llm import LLMClient
from .tools import execute_tool, get_tool_definitions

try:
    from .permissions import PermissionManager
except ImportError:
    PermissionManager = None


class AgentLoop:
    def __init__(self,
                 llm: LLMClient,
                 context: ContextManager,
                 session_id: str,
                 max_iterations: int = MAX_ITERATIONS,
                 verbose: bool = True,
                 permission_manager=None,
                 extra_body: dict = None):
        self.llm = llm
        self.context = context
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tool_defs = get_tool_definitions()
        self.permission_manager = permission_manager
        self.extra_body = extra_body or {}

    def _log(self, s: str):
        if self.verbose:
            print(s, file=sys.stderr, flush=True)

    def run(self, user_input: str, stream: bool = True) -> str:
        """Satu turn: user_input -> loop hingga selesai -> return final answer."""
        self.context.add_user(user_input)
        save_message(self.session_id, {"role": "user", "content": user_input})
        # helper untuk ambil think saat ini
        def _current_think():
            return self.extra_body.get("reasoning_effort") if self.extra_body else "none"

        final_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            messages = self.context.get_messages()
            usage = self.context.token_usage()
            self._log(f"\n[iter {iteration}/{self.max_iterations} | tokens {usage['tokens']}/{usage['max']} ({usage['percent']}%)]")

            # Streaming callbacks: print ke stdout langsung (tanpa TUI)
            def on_delta(tok: str):
                if stream:
                    sys.stdout.write(tok)
                    sys.stdout.flush()

            # Panggil LLM dengan extra_body (mis. reasoning_effort untuk thinking) - escape (Ctrl-C/ESC) membatalkan
            try:
                result = self.llm.chat(
                    messages,
                    tools=self.tool_defs,
                    stream=stream,
                    on_delta=on_delta if stream else None,
                    extra_body=self.extra_body if self.extra_body else None,
                    timeout=120
                )
            except KeyboardInterrupt:
                # Escape membatalkan response - jangan simpan partial ke context/history
                print("\n[escape] response dibatalkan", file=sys.stderr)
                try:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                except:
                    pass
                return "[cancelled - escape]"
            except Exception as e:
                err = f"[LLM error iter {iteration}: {e}]"
                print(f"\n{err}", file=sys.stderr)
                self.context.add_assistant(err)
                save_message(self.session_id, {"role": "assistant", "content": err, "model": self.llm.model, "think_variant": _current_think()})
                return err

            content = result.get("content") or ""
            tool_calls = result.get("tool_calls")

            if stream and content:
                # sudah di-print via on_delta, beri newline jika ada tool_calls selanjutnya
                if tool_calls:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            if not tool_calls:
                # Selesai - tidak ada tool
                # Jika stream=False, print content sekarang
                if not stream and content:
                    print(content)
                self.context.add_assistant(content)
                save_message(self.session_id, {"role": "assistant", "content": content, "model": self.llm.model, "think_variant": _current_think()})
                final_answer = content
                break
            else:
                # Ada tool calls: simpan assistant message + eksekusi tools (sertakan model & think)
                self.context.add_assistant(content, tool_calls)
                save_message(self.session_id, {"role": "assistant", "content": content, "tool_calls": tool_calls, "model": self.llm.model, "think_variant": _current_think()})

                # Tampilkan tool calls ke stderr agar user tahu
                for tc in tool_calls:
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    # pretty print args
                    try:
                        parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                        fargs_str = json.dumps(parsed, ensure_ascii=False, indent=2)[:800]
                    except:
                        fargs_str = str(fargs)[:800]
                    self._log(f"  -> tool: {fname} {fargs_str}")

                # Eksekusi tiap tool sequential dengan permission check
                for tc in tool_calls:
                    tid = tc.get("id", f"call_{iteration}")
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    # Permission gate untuk hardware tools
                    if self.permission_manager is not None:
                        # pretty args untuk prompt
                        try:
                            parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                            prompt_args = json.dumps(parsed, ensure_ascii=False)[:300]
                        except:
                            prompt_args = str(fargs)[:300]
                        allowed = self.permission_manager.check_or_prompt(fname, prompt_args)
                        if not allowed:
                            output = f"[DENIED] User menolak eksekusi tool '{fname}' dengan args {prompt_args}. Sampaikan ke user bahwa izin ditolak dan tawarkan alternatif."
                            self._log(f"[tool {fname} DENIED by user]")
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                            continue
                    self._log(f"[exec {fname}...]")
                    t0 = time.time()
                    output = execute_tool(fname, fargs)
                    dt = time.time() - t0
                    self._log(f"[tool {fname} done {dt:.1f}s, {len(output)} chars]")
                    # Masukkan ke context + history
                    self.context.add_tool_result(tid, fname, output)
                    save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                    # Juga print summary ke stdout jika verbose? Tidak, biar LLM saja yang ringkas.
                    if self.verbose:
                        # tampilkan preview 300 chars ke stderr
                        preview = output[:600].replace("\n", " ")
                        if len(output) > 600:
                            preview += "..."
                        self._log(f"     result preview: {preview}")

                # lanjut loop - LLM akan lihat tool results
                if stream:
                    # pemisah visual
                    print(f"\n[tool results fed back, continuing...]\n", file=sys.stderr)
                continue
        else:
            # max iterations tercapai
            final_answer = content if 'content' in locals() else ""
            self._log(f"[max iterations {self.max_iterations} reached]")
            if not final_answer:
                final_answer = "[Agent stopped: max iterations reached without final answer]"
                self.context.add_assistant(final_answer)
                save_message(self.session_id, {"role": "assistant", "content": final_answer, "model": self.llm.model, "think_variant": _current_think()})

        return final_answer

    def run_once(self, user_input: str, stream: bool = True) -> str:
        return self.run(user_input, stream=stream)
