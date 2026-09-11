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
        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            messages = self.context.get_messages()
            self._log(f"[iter {iteration}/{self.max_iterations}]")

            # Streaming callbacks: print ke stdout langsung (tanpa TUI)
            def on_delta(tok: str):
                if stream:
                    sys.stdout.write(tok)
                    sys.stdout.flush()

            # Thinking callback - tampilkan saat LLM benar-benar reasoning
            _think_started = False
            def on_reasoning_delta(tok: str):
                nonlocal _think_started
                if not _think_started:
                    _think = _current_think()
                    if _think != "none":
                        self._log(f">>>>>> thinking [{_think}]")
                    else:
                        self._log(f">>>>>> thinking")
                    _think_started = True
                # Jika reasoning text tersedia (Anthropic thinking), bisa tampilkan sebagai dimmed? Untuk sekarang hanya indikator
                # Jika ingin tampilkan reasoning, uncomment di bawah:
                # if stream and tok:
                #     sys.stdout.write(tok)
                #     sys.stdout.flush()

            # Panggil LLM dengan extra_body (mis. reasoning_effort untuk thinking) - escape (Ctrl-C/ESC) membatalkan
            # Jika non-stream dan thinking aktif, tampilkan thinking sebelum call (karena tidak ada streaming reasoning)
            if not stream and _current_think() != "none" and not _think_started:
                if _current_think() != "none":
                    self._log(f">>>>>> thinking [{_current_think()}]")
                else:
                    self._log(f">>>>>> thinking")
                _think_started = True
            try:
                result = self.llm.chat(
                    messages,
                    tools=self.tool_defs,
                    stream=stream,
                    on_delta=on_delta if stream else None,
                    on_reasoning_delta=on_reasoning_delta if stream else None,
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

                # Tampilkan hanya apa yang dikerjakan sesuai spec (tanpa hasil)
                for tc in tool_calls:
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    try:
                        parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                        if not isinstance(parsed, dict):
                            parsed = {}
                    except:
                        parsed = {}
                    # Format per spec
                    if fname in ("read", "write", "edit"):
                        # fs: ">>>> {type} {filepath}"
                        fp = parsed.get("filePath") or parsed.get("filepath") or ""
                        self._log(f">>>> {fname} {fp}".strip())
                    elif fname == "bash":
                        # bash/command: ">>>> exec {command}"
                        cmd = parsed.get("command", "")
                        self._log(f">>>> exec {cmd}".strip())
                    elif fname == "skill_list":
                        self._log(">>>> fetch-skills")
                    elif fname == "skill_load":
                        name = parsed.get("name", "")
                        self._log(f">>>> load-skill [{name}]" if name else ">>>> load-skill")
                    elif fname == "grep":
                        pat = parsed.get("pattern", "")
                        p = parsed.get("path", "") or parsed.get("include", "")
                        # grep: ">>>> grep {pattern} {path}"
                        self._log(f">>>> grep {pat} {p}".strip())
                    elif fname == "glob":
                        pat = parsed.get("pattern", "")
                        p = parsed.get("path", "")
                        self._log(f">>>> glob {pat} {p}".strip() if p else f">>>> glob {pat}".strip())
                    else:
                        # fallback generic
                        if isinstance(fargs, str):
                            args_str = fargs
                        else:
                            args_str = json.dumps(fargs, ensure_ascii=False)
                        self._log(f">>>> {fname} {args_str}")

                # Eksekusi tiap tool sequential dengan permission check - Batalkan truncate, simpan full
                for tc in tool_calls:
                    tid = tc.get("id", f"call_{iteration}")
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    # Tampilkan preparing sebelum eksekusi
                    self._log(f">>>>> preparing {fname}")
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
                            # Batalkan truncate - simpan full history
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                            continue
                    output = execute_tool(fname, fargs)
                    # Batalkan truncate - simpan full output ke context & history
                    self.context.add_tool_result(tid, fname, output)
                    save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})

                continue
        else:
            # max iterations tercapai
            final_answer = content if 'content' in locals() else ""
            self._log(f"[max iterations {self.max_iterations} reached]")
            if not final_answer:
                final_answer = "[Agent stopped: max iterations reached without final answer]"
                self.context.add_assistant(final_answer)
                save_message(self.session_id, {"role": "assistant", "content": final_answer, "model": self.llm.model, "think_variant": _current_think()})

        # Token usage & iter & time hanya tampil ketika iter selesai (bukan saat berjalan)
        try:
            usage = self.context.token_usage()
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            # n = total iter yang dijalankan (iteration terakhir)
            n = iteration if 'iteration' in locals() else self.max_iterations
            self._log(f"[Total Iter: {n} | Token: {usage['tokens']}/{usage['max']} {usage['percent']}% | Time: {h}h {m}m {s}s]")
        except:
            pass

        return final_answer

    def run_once(self, user_input: str, stream: bool = True) -> str:
        return self.run(user_input, stream=stream)
