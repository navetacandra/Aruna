"""Agent Loop - Iterative ReAct with powerful tool calling."""
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
        self._all_tool_defs = {t["function"]["name"]: t for t in get_tool_definitions()}
        # Lazy: only load required tools, initially empty, will be populated per iteration based on need
        self.tool_defs: List[Dict[str, Any]] = []
        self.permission_manager = permission_manager
        self.extra_body = extra_body or {}
        self._tool_cache: Dict[str, str] = {}
        self._tool_history: List[str] = []  # list of "fname:args_json" for dedup
        self._loaded_tools: set = set()

    def _select_tools_for_input(self, user_input: str) -> List[Dict[str, Any]]:
        """Select only the required tools based on user prompt (lazy loading)."""
        text = user_input.lower()
        needed = set()
        # Always provide skill discovery as lazy entry point
        # But don't load all skill content, only via tool
        # Heuristic based on keywords
        if any(k in text for k in ["baca", "read", "lihat", "tampilkan", "file", "cat", "ls"]):
            needed.update(["read", "glob"])
        if any(k in text for k in ["tulis", "buat", "write", "simpan", "edit", "ubah", "ganti"]):
            needed.update(["write", "edit", "read"])
        if any(k in text for k in ["cari", "search", "grep", "temukan"]):
            needed.update(["grep", "glob"])
        if any(k in text for k in ["jalan", "exec", "bash", "command", "shell", "run", "eksekusi", "pwd", "ls", "python", "pip", "npm"]):
            needed.update(["bash"])
        if any(k in text for k in ["skill", "kemampuan", "lakukan"]):
            needed.update(["skill_list", "skill_load"])
        # If nothing matches, provide minimal discovery tools + read/bash as fallback
        if not needed:
            needed.update(["read", "bash", "skill_list"])
        # Always include skill discovery if not already present
        if "skill_list" not in needed and "skill" in text:
            needed.add("skill_list")
        # Build list of defs that are only required
        defs = [self._all_tool_defs[n] for n in needed if n in self._all_tool_defs]
        # Track loaded
        for n in needed:
            self._loaded_tools.add(n)
        return defs

    def _ensure_tool_loaded(self, fname: str):
        """If LLM requests a tool that hasn't been loaded, load on-demand for next iteration."""
        if fname not in self._loaded_tools and fname in self._all_tool_defs:
            self._loaded_tools.add(fname)
            # Add to tool_defs for next iteration
            if self._all_tool_defs[fname] not in self.tool_defs:
                self.tool_defs.append(self._all_tool_defs[fname])
                self._log(f"[lazy] tool '{fname}' loaded on-demand")

    def _log(self, s: str):
        if self.verbose:
            print(s, file=sys.stderr, flush=True)

    def run(self, user_input: str, stream: bool = True) -> str:
        """One turn: user_input -> loop until done -> return final answer."""
        self.context.add_user(user_input)
        save_message(self.session_id, {"role": "user", "content": user_input})
        # Lazy load: only tools required for this prompt
        self.tool_defs = self._select_tools_for_input(user_input)
        self._log(f"[tools] loaded {len(self.tool_defs)}: {', '.join(t['function']['name'] for t in self.tool_defs)} (lazy)")
        # helper to get current think
        def _current_think():
            return self.extra_body.get("reasoning_effort") if self.extra_body else "none"

        final_answer = ""
        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            messages = self.context.get_messages()
            self._log(f"[iter {iteration}/{self.max_iterations}]")

            # Stuck loop detection: if many iterations have passed and still repeating same tool, nudge LLM
            if iteration in (15, 20, 25):
                self._log(f"[warning] iter {iteration} still running, LLM may be stuck. Pushing to provide final answer soon.")
                self.context.messages.append({"role": "user", "content": f"[SYSTEM REMINDER] You are at iteration {iteration}/{self.max_iterations}. If you have enough information, please provide the final answer soon. Do not keep calling the same tool repeatedly (especially fetch-skills/glob/grep). If stuck, make the best conclusion from available information."})
                messages = self.context.get_messages()
            # Detect excessive tool frequency in history
            if len(self._tool_history) >= 8:
                from collections import Counter
                recent = self._tool_history[-8:]
                cnt = Counter([h.split(":")[0] for h in recent])
                for tname, c in cnt.items():
                    if c >= 4:
                        self._log(f"[warning] tool {tname} called {c}x in last 8 calls, likely stuck.")
                        self.context.messages.append({"role": "user", "content": f"[SYSTEM REMINDER] Tool '{tname}' has been called {c} times in the last 8 calls with similar arguments. Result is already in history. DO NOT call '{tname}' again with the same arguments. Use the existing results or provide the final answer."})
                        break

            # Streaming callbacks: print directly to stdout (without TUI)
            def on_delta(tok: str):
                if stream:
                    sys.stdout.write(tok)
                    sys.stdout.flush()

            # Thinking callback - show when LLM is actually reasoning
            _think_started = False
            def on_reasoning_delta(tok: str):
                nonlocal _think_started
                if not _think_started:
                    _think = _current_think()
                    if _think != "none":
                        self._log(f"<<< thinking [{_think}]")
                    else:
                        self._log(f"<<< thinking")
                    _think_started = True
                # If reasoning text is available (Anthropic thinking), could display as dimmed? For now just an indicator
                # If you want to display reasoning, uncomment below:
                # if stream and tok:
                #     sys.stdout.write(tok)
                #     sys.stdout.flush()

            # Call LLM with extra_body (e.g. reasoning_effort for thinking) - escape (Ctrl-C/ESC) cancels
            # If non-stream and thinking is active, show thinking before call (since there is no streaming reasoning)
            if not stream and _current_think() != "none" and not _think_started:
                if _current_think() != "none":
                    self._log(f"<<< thinking [{_current_think()}]")
                else:
                    self._log(f"<<< thinking")
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
                # Escape cancels response - don't save partial to context/history
                print("\n[escape] response cancelled", file=sys.stderr)
                try:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                except:
                    pass
                return "[cancelled - escape]"
            except Exception as e:
                # Special binary file - only Response is supported
                if "Model tidak didukung" in str(e) or "Model not supported" in str(e):
                    err = "Model not supported"
                    print(f"\n{err}", file=sys.stderr)
                    # also print to stdout so user can see
                    try:
                        print(err)
                    except:
                        pass
                    self.context.add_assistant(err)
                    save_message(self.session_id, {"role": "assistant", "content": err, "model": self.llm.model, "think_variant": _current_think()})
                    return err
                err = f"[LLM error iter {iteration}: {e}]"
                print(f"\n{err}", file=sys.stderr)
                self.context.add_assistant(err)
                save_message(self.session_id, {"role": "assistant", "content": err, "model": self.llm.model, "think_variant": _current_think()})
                return err

            content = result.get("content") or ""
            tool_calls = result.get("tool_calls")

            if stream and content:
                # already printed via on_delta, add newline if there are subsequent tool_calls
                if tool_calls:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            if not tool_calls:
                # Check if LLM mentioned a tool not yet loaded but needed (lazy fallback) - only from content, not from input
                lower_content = content.lower() if content else ""
                needs_tool = False
                for tname in self._all_tool_defs:
                    if tname not in [t["function"]["name"] for t in self.tool_defs] and tname in lower_content:
                        # LLM mentioned a tool not yet loaded, load on-demand for next iteration
                        self._ensure_tool_loaded(tname)
                        needs_tool = True
                if needs_tool and iteration < self.max_iterations:
                    # Inject reminder to use newly loaded tool
                    self._log(f"[lazy] additional tool loaded, asking LLM to try again")
                    self.context.messages.append({"role": "user", "content": f"[SYSTEM] The tool you need is now available: {', '.join([t['function']['name'] for t in self.tool_defs])}. Use that tool to complete the task, don't just answer with text."})
                    save_message(self.session_id, {"role": "user", "content": f"[SYSTEM] Available tools: {', '.join([t['function']['name'] for t in self.tool_defs])}"})
                    continue
                # Done - no tools
                # If stream=False, print content now (handle UTF-8 on Windows)
                if not stream and content:
                    try:
                        print(content)
                    except UnicodeEncodeError:
                        try:
                            sys.stdout.reconfigure(encoding='utf-8')
                            print(content)
                        except:
                            print(content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'))
                self.context.add_assistant(content)
                save_message(self.session_id, {"role": "assistant", "content": content, "model": self.llm.model, "think_variant": _current_think()})
                final_answer = content
                break
            else:
                # Tool calls present: save assistant message + execute tools (include model & think)
                self.context.add_assistant(content, tool_calls)
                save_message(self.session_id, {"role": "assistant", "content": content, "tool_calls": tool_calls, "model": self.llm.model, "think_variant": _current_think()})

                # Display only what is being done per spec (without results)
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
                        # fs: "<<< {type} {filepath}"
                        fp = parsed.get("filePath") or parsed.get("filepath") or ""
                        self._log(f"<<< preparing {fname}")
                    elif fname == "bash":
                        # bash/command: "<<< exec {command}"
                        cmd = parsed.get("command", "")
                        self._log(f"<<< exec {cmd}".strip())
                    elif fname == "skill_list":
                        self._log("<<< fetch-skills")
                    elif fname == "skill_load":
                        name = parsed.get("name", "")
                        self._log(f"<<< load-skill [{name}]" if name else "<<< load-skill")
                    elif fname == "grep":
                        pat = parsed.get("pattern", "")
                        p = parsed.get("path", "") or parsed.get("include", "")
                        # grep: "<<< grep {pattern} {path}"
                        self._log(f"<<< grep {pat} {p}".strip())
                    elif fname == "glob":
                        pat = parsed.get("pattern", "")
                        p = parsed.get("path", "")
                        self._log(f"<<< glob {pat} {p}".strip() if p else f"<<< glob {pat}".strip())
                    else:
                        # fallback generic
                        if isinstance(fargs, str):
                            args_str = fargs
                        else:
                            args_str = json.dumps(fargs, ensure_ascii=False)
                        self._log(f"<<< {fname} {args_str}")

                # Execute each tool sequentially with permission check - Cancel truncate, save full
                for tc in tool_calls:
                    tid = tc.get("id", f"call_{iteration}")
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]

                    # Show preparing before execution
                    if fname in ("read", "write", "edit"):
                        parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                        fp = parsed.get("filePath") or parsed.get("filepath") or ""
                        self._log(f"<<< {fname} {fp}".strip())

                    # Deduplication: check if same tool with same args was called recently
                    try:
                        parsed_args = json.loads(fargs) if isinstance(fargs, str) else fargs
                        cache_key = f"{fname}:{json.dumps(parsed_args, sort_keys=True, ensure_ascii=False)}"
                    except:
                        cache_key = f"{fname}:{fargs}"
                    # Track history for frequency detection
                    self._tool_history.append(cache_key)
                    if len(self._tool_history) > 20:
                        self._tool_history = self._tool_history[-20:]
                    # If already called in last 5 calls with same result, skip and warn
                    if cache_key in self._tool_cache:
                        # Check if this is a repeated call within short window (last 5)
                        recent_calls = self._tool_history[-6:-1]  # 5 before current
                        if cache_key in recent_calls:
                            self._log(f"[skip] {fname} with same args was called recently, using cache")
                            output = self._tool_cache[cache_key] + "\n\n[NOTE: This tool was already called previously with the same arguments. Result is cached to avoid loops. Do not call again with the same arguments.]"
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                            continue
                    # Permission gate for hardware tools
                    if self.permission_manager is not None:
                        try:
                            parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                            prompt_args = json.dumps(parsed, ensure_ascii=False)[:300]
                        except:
                            prompt_args = str(fargs)[:300]
                        allowed = self.permission_manager.check_or_prompt(fname, prompt_args)
                        if not allowed:
                            output = f"[DENIED] User denied execution of tool '{fname}' with args {prompt_args}. Inform the user that permission was denied and offer alternatives."
                            # Smart truncate for lightweight history
                            from agent_core.config import MAX_TOOL_OUTPUT_CHARS
                            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                                head = int(MAX_TOOL_OUTPUT_CHARS * 0.6)
                                tail = MAX_TOOL_OUTPUT_CHARS - head - 100
                                output = output[:head] + f"\n...[SMART TRUNCATED {len(output)-MAX_TOOL_OUTPUT_CHARS} chars]...\n" + output[-tail:] if tail>0 else output[:MAX_TOOL_OUTPUT_CHARS]
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                            continue
                    output = execute_tool(fname, fargs)
                    # Save to cache for deduplication (full)
                    self._tool_cache[cache_key] = output
                    # Smart truncate before saving - saves 50-70% tokens but keeps head+tail
                    from agent_core.config import MAX_TOOL_OUTPUT_CHARS
                    to_store = output
                    if len(output) > MAX_TOOL_OUTPUT_CHARS:
                        head = int(MAX_TOOL_OUTPUT_CHARS * 0.6)
                        tail = MAX_TOOL_OUTPUT_CHARS - head - 100
                        to_store = output[:head] + f"\n...[SMART TRUNCATED {len(output)-MAX_TOOL_OUTPUT_CHARS} chars, saved {(1-MAX_TOOL_OUTPUT_CHARS/len(output))*100:.0f}%]...\n" + output[-tail:] if tail>0 else output[:MAX_TOOL_OUTPUT_CHARS]
                    self.context.add_tool_result(tid, fname, to_store)
                    save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": to_store})

                continue
        else:
            # max iterations reached
            final_answer = content if 'content' in locals() else ""
            self._log(f"[max iterations {self.max_iterations} reached]")
            if not final_answer:
                final_answer = "[Agent stopped: max iterations reached without final answer]"
                self.context.add_assistant(final_answer)
                save_message(self.session_id, {"role": "assistant", "content": final_answer, "model": self.llm.model, "think_variant": _current_think()})

        # Token usage & iter & time only shown when iteration finishes (not while running)
        try:
            usage = self.context.token_usage()
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            # n = total iterations run (last iteration)
            n = iteration if 'iteration' in locals() else self.max_iterations
            self._log(f"[Total Iter: {n} | Token: {usage['tokens']}/{usage['max']} {usage['percent']}% | Time: {h}h {m}m {s}s]")
        except:
            pass

        return final_answer

    def run_once(self, user_input: str, stream: bool = True) -> str:
        return self.run(user_input, stream=stream)
