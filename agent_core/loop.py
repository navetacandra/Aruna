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
            self._log(f"[iter {iteration}/{self.max_iterations}]")

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

                # Tampilkan hanya apa yang dikerjakan (tanpa hasil)
                for tc in tool_calls:
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    try:
                        parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                    except:
                        parsed = {}
                    # Format ringkas: Read blabla.txt, Write blabla.txt, Exec echo 'blabla'
                    display = fname
                    if fname == "read":
                        display = f"Read {parsed.get('filePath','')}"
                    elif fname == "write":
                        display = f"Write {parsed.get('filePath','')}"
                    elif fname == "edit":
                        display = f"Edit {parsed.get('filePath','')}"
                    elif fname == "glob":
                        display = f"Glob {parsed.get('pattern','')}"
                    elif fname == "grep":
                        display = f"Grep {parsed.get('pattern','')}"
                    elif fname == "bash":
                        cmd = parsed.get('command','')[:60].replace('\n',' ')
                        display = f"Exec {cmd}"
                    elif fname == "skill_list":
                        display = "Skill list"
                    elif fname == "skill_load":
                        display = f"Skill load {parsed.get('name','')}"
                    else:
                        display = fname
                    self._log(display)

                # Eksekusi tiap tool sequential dengan permission check
                for tc in tool_calls:
                    tid = tc.get("id", f"call_{iteration}")
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    # Permission gate untuk hardware tools
                    if self.permission_manager is not None:
                        try:
                            parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                            prompt_args = json.dumps(parsed, ensure_ascii=False)[:300]
                        except:
                            prompt_args = str(fargs)[:300]
                        allowed = self.permission_manager.check_or_prompt(fname, prompt_args)
                        if not allowed:
                            output = f"[DENIED] User menolak eksekusi tool '{fname}' dengan args {prompt_args}. Sampaikan ke user bahwa izin ditolak dan tawarkan alternatif."
                            # Truncate untuk history ringan
                            truncated = output[:4000] + ("\n...[truncated]" if len(output) > 4000 else "")
                            self.context.add_tool_result(tid, fname, truncated)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": truncated})
                            continue
                    output = execute_tool(fname, fargs)
                    # Truncate hasil panjang sebelum simpan - trade-off konteks perlu diulang tapi history tidak berat
                    # Simpan max 4000 chars ke history & context, bukan full 20000
                    if len(output) > 4000:
                        truncated = output[:3500] + f"\n...[truncated {len(output)-3500} chars, total {len(output)}]...\n" + output[-500:]
                        to_store = truncated[:4000]
                    else:
                        to_store = output
                    self.context.add_tool_result(tid, fname, to_store)
                    save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": to_store})

                continue
        else:
            # max iterations tercapai
            final_answer = content if 'content' in locals() else ""
            self._log(f"[max iterations {self.max_iterations} reached]")
            if not final_answer:
                final_answer = "[Agent stopped: max iterations reached without final answer]"
                self.context.add_assistant(final_answer)
                save_message(self.session_id, {"role": "assistant", "content": final_answer, "model": self.llm.model, "think_variant": _current_think()})

        # Token usage hanya tampil ketika iter selesai (bukan saat berjalan)
        try:
            usage = self.context.token_usage()
            self._log(f"[done] tokens {usage['tokens']}/{usage['max']} ({usage['percent']}%)")
        except:
            pass

        return final_answer

    def run_once(self, user_input: str, stream: bool = True) -> str:
        return self.run(user_input, stream=stream)
