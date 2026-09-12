"""Agent Loop - Iterative ReAct with powerful tool calling and concurrent sub-agents."""
import concurrent.futures
import json
import sys
import time
import uuid
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
        self._tool_history: List[str] = []  # list of "fname:args_json" for stuck detection
        self._loaded_tools: set = set()

    def _select_tools_for_input(self, user_input: str) -> List[Dict[str, Any]]:
        """Select tools - now loads all core tools by default to avoid hiding capabilities."""
        # Always load core tools to avoid LLM blindness - lazy loading was hiding critical tools
        needed = {"read", "write", "edit", "glob", "grep", "bash", "spawn_agents", "skill_list", "skill_load"}
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

    def _execute_spawn_agents(self, tasks: List[Dict[str, str]], parent_iteration: int) -> str:
        """Execute multiple sub-agents concurrently. Each sub-agent gets its own context and loop."""
        from .context import ContextManager
        from .prompts import build_system_prompt
        import uuid as _uuid

        def run_single(task_info: Dict[str, str]) -> str:
            label = task_info.get("label", "subagent")
            task_prompt = task_info.get("task", "")
            sub_session_id = f"{self.session_id}-sub-{_uuid.uuid4().hex[:8]}"
            # Sub-agent gets a fresh context but shares system prompt
            sub_ctx = ContextManager(system_prompt=self.context.system_prompt, max_tokens=self.context.max_tokens, keep_recent=self.context.keep_recent)
            # Load recent history snippet for context (last 2 messages) to give sub-agent some background without full history
            # Keep it minimal to avoid token bloat
            try:
                recent = self.context.messages[-2:] if len(self.context.messages) > 2 else []
                for m in recent:
                    if m.get("role") == "user":
                        sub_ctx.add_user(m.get("content", "")[:2000])
                    elif m.get("role") == "assistant":
                        sub_ctx.add_assistant(m.get("content", "")[:2000])
            except:
                pass
            # Sub-agent uses same LLM model but with reduced max_iterations and no spawn_agents to avoid recursion
            sub_max_iter = max(5, min(12, self.max_iterations // 2 + 3))
            # Filter out spawn_agents from sub-agent tools to prevent infinite recursion
            sub_loop = AgentLoop(
                llm=self.llm,
                context=sub_ctx,
                session_id=sub_session_id,
                max_iterations=sub_max_iter,
                verbose=False,  # quiet sub-agents, parent will log summary
                permission_manager=self.permission_manager,
                extra_body=dict(self.extra_body) if self.extra_body else {}
            )
            # Remove spawn_agents from sub-agent's available tools
            if "spawn_agents" in sub_loop._all_tool_defs:
                # Create filtered defs without spawn_agents for sub-agent
                sub_loop._all_tool_defs = {k: v for k, v in sub_loop._all_tool_defs.items() if k != "spawn_agents"}
            try:
                result = sub_loop.run(task_prompt, stream=False)
                return f"[{label}] Task: {task_prompt}\nResult: {result}\n[Sub-agent {label} finished in {sub_loop._tool_history.__len__()} tool calls]"
            except Exception as e:
                return f"[{label}] Task: {task_prompt}\nError: {e}"

        # Run all sub-agents concurrently
        start = time.time()
        results: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {executor.submit(run_single, t): t for t in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    res = future.result(timeout=120)
                    results.append(res)
                except Exception as e:
                    label = future_to_task[future].get("label", "unknown")
                    results.append(f"[{label}] Error: {e}")

        elapsed = time.time() - start
        header = f"[spawn_agents] Completed {len(tasks)} sub-agents concurrently in {elapsed:.1f}s (parent iter {parent_iteration} still counts as 1)\n"
        header += f"Sub-agents executed in parallel, saving ~{len(tasks)-1} iterations for main agent.\n"
        header += "="*60 + "\n"
        return header + "\n\n".join(results)

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
            # Detect excessive tool frequency in history - only if same tool with same args repeated
            if len(self._tool_history) >= 8:
                from collections import Counter
                recent = self._tool_history[-8:]
                cnt = Counter(recent)
                for full_key, c in cnt.items():
                    if c >= 4:
                        tname = full_key.split(":", 1)[0]
                        self._log(f"[warning] tool {tname} with same args called {c}x in last 8 calls, likely stuck.")
                        self.context.messages.append({"role": "user", "content": f"[SYSTEM REMINDER] Tool '{tname}' with the same arguments has been called {c} times in the last 8 calls. Result is already in history. DO NOT call '{tname}' again with the same arguments. Use the existing results or provide the final answer."})
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
                    self._log(f"[Iter {iteration}/{self.max_iterations}]\n")
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
                self._log(f"[Iter {iteration}/{self.max_iterations}]\n")
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
                        # fs: "<<< preparing {type} {filepath}" - show target path
                        fp = parsed.get("filePath") or parsed.get("filepath") or parsed.get("file_path") or parsed.get("path") or ""
                        self._log(f"<<< preparing {fname} {fp}".strip())
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
                    elif fname == "spawn_agents":
                        tasks = parsed.get("tasks", [])
                        labels = [t.get("label", f"agent-{i+1}") for i, t in enumerate(tasks) if isinstance(t, dict)]
                        if not labels and isinstance(tasks, list):
                            labels = [f"agent-{i+1}" for i in range(len(tasks))]
                        self._log(f"<<< spawn_agents [{', '.join(labels)}] ({len(tasks)} concurrent)")
                    else:
                        # fallback generic
                        if isinstance(fargs, str):
                            args_str = fargs
                        else:
                            args_str = json.dumps(fargs, ensure_ascii=False)
                        self._log(f"<<< {fname} {args_str}")

                # Execute each tool sequentially, but handle spawn_agents concurrently
                for tc in tool_calls:
                    tid = tc.get("id", f"call_{iteration}")
                    fname = tc["function"]["name"]
                    fargs = tc["function"]["arguments"]
                    # Special handling for spawn_agents - concurrent sub-agents
                    if fname == "spawn_agents":
                        try:
                            parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                            tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
                            # Normalize tasks - handle both string and object formats
                            normalized = []
                            for t in tasks:
                                if isinstance(t, str):
                                    normalized.append({"task": t, "label": f"agent-{len(normalized)+1}"})
                                elif isinstance(t, dict) and "task" in t:
                                    normalized.append({"task": t["task"], "label": t.get("label", f"agent-{len(normalized)+1}")})
                            if not normalized:
                                output = "Error: spawn_agents requires 'tasks' array with at least one {task: string}"
                                self.context.add_tool_result(tid, fname, output)
                                save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                                self._log(f"[Iter {iteration}/{self.max_iterations}]\n")
                                continue
                            # Limit concurrent sub-agents to avoid explosion
                            if len(normalized) > 5:
                                normalized = normalized[:5]
                                self._log(f"[spawn_agents] limited to 5 concurrent sub-agents (was {len(tasks)})")
                            self._log(f"[spawn_agents] spawning {len(normalized)} sub-agents concurrently...")
                            output = self._execute_spawn_agents(normalized, iteration)
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                            # Show preview
                            preview = output[:2000] + ("...[truncated]" if len(output) > 2000 else "")
                            self._log(f"[spawn_agents output]\n{preview}")
                        except Exception as e:
                            output = f"Error in spawn_agents: {e}"
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                        # Track history
                        try:
                            parsed_args = json.loads(fargs) if isinstance(fargs, str) else fargs
                            cache_key = f"{fname}:{json.dumps(parsed_args, sort_keys=True, ensure_ascii=False)}"
                        except:
                            cache_key = f"{fname}:{fargs}"
                        self._tool_history.append(cache_key)
                        if len(self._tool_history) > 20:
                            self._tool_history = self._tool_history[-20:]
                        self._log(f"[Iter {iteration}/{self.max_iterations}]\n")
                        continue

                    # Show preparing before execution
                    if fname in ("read", "write", "edit"):
                        try:
                            parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
                        except:
                            parsed = {}
                        fp = parsed.get("filePath") or parsed.get("filepath") or parsed.get("file_path") or parsed.get("path") or ""
                        self._log(f"<<< {fname} {fp}".strip())

                    # Track history for stuck detection (no cache)
                    try:
                        parsed_args = json.loads(fargs) if isinstance(fargs, str) else fargs
                        cache_key = f"{fname}:{json.dumps(parsed_args, sort_keys=True, ensure_ascii=False)}"
                    except:
                        cache_key = f"{fname}:{fargs}"
                    self._tool_history.append(cache_key)
                    if len(self._tool_history) > 20:
                        self._tool_history = self._tool_history[-20:]
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
                            self.context.add_tool_result(tid, fname, output)
                            save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})
                            continue
                    output = execute_tool(fname, fargs)
                    # Show bash output directly
                    if fname == "bash":
                        self._log(f"[bash output]\n{output}")
                    self.context.add_tool_result(tid, fname, output)
                    save_message(self.session_id, {"role": "tool", "tool_call_id": tid, "name": fname, "content": output})

                self._log(f"[Iter {iteration}/{self.max_iterations}]\n")
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
