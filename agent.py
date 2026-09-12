#!/usr/bin/env python3
"""Agent CLI - simple without TUI, user input always at the bottom.

Usage:
  python agent.py                   # interactive REPL
  python agent.py --once "query"    # single turn
  python agent.py --model mimo-v2.5-free --session ses_xxx
  python agent.py --list-models
  python agent.py --continue        # continue last session

Commands REPL:
  /compact                 manual context compaction
  /model(s) [name]         view model list & select (interactive if no arg)
  /usage(s)                view token usage
  /skill(s) [name]         view skill list / load skill
  /reload                  reload state without losing context
  /think [variant]         view/select thinking variant (none/low/medium/high/xhigh)
  /tool-call [mode]        tool permission: accept-all, accept-fs, ask
  /max-iter [n]            view/set max iterations (1-100, saved per session)
  /info                    show model, token, chat length, max-iter
  @file_name               embed file (text/binary pdf/photo) into prompt, e.g.: @README.md @"my file.pdf"
  !<command>               run shell directly (terminatable via Ctrl-C)
  /help, /clear, /session, /exit
"""
import argparse
import pathlib
import sys
import os
import subprocess
import json

# Ensure package import works when run as script
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agent_core.config import DEFAULT_MODEL, HISTS_DIR
from agent_core.context import ContextManager
from agent_core.history import generate_session_id, list_sessions, load_messages, get_last_model_and_think, get_last_max_iterations, save_message
from agent_core.llm import LLMClient, fetch_models, is_responses_model
from agent_core.loop import AgentLoop
from agent_core.prompts import build_system_prompt
from agent_core.skills import build_skills_catalog, list_skills as skl_list, load_skill
from agent_core.permissions import PermissionManager

THINK_VARIANTS = ["none", "low", "medium", "high", "xhigh"]  # none = non-thinking, xhigh = extra high
DEFAULT_THINK = "none"

# For @file embedding
import re
import base64
import mimetypes

def _is_binary_file(path: pathlib.Path) -> bool:
    # Detect binary via extension or try reading
    bin_exts = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".zip", ".tar", ".gz", ".exe", ".bin", ".docx", ".xlsx", ".pptx"}
    if path.suffix.lower() in bin_exts:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(8000)
            if b"\x00" in chunk:
                return True
            # try to decode
            chunk.decode("utf-8")
            return False
    except:
        return True

def _handle_binary_file(path: pathlib.Path) -> str:
    size = path.stat().st_size if path.exists() else 0
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    # detect mime via magic so beboo.png (JPEG) is correct
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if head.startswith(b"\xFF\xD8\xFF"):
                mime = "image/jpeg"
            elif head.startswith(b"\x89PNG"):
                mime = "image/png"
            elif head.startswith(b"GIF8"):
                mime = "image/gif"
            elif head.startswith(b"RIFF") and b"WEBP" in head:
                mime = "image/webp"
            elif head.startswith(b"%PDF"):
                mime = "application/pdf"
    except:
        pass
    ext = path.suffix.lower()
    header = f"[Binary file: {path} | type: {mime} | size: {size} bytes]"
    abs_path = str(path.resolve()) if path.exists() else str(path)
    # For images & PDFs, create marker for Response API (b.md)
    # If model is not Response, will be rejected in providers.prepare_payload -> Model not supported
    if mime.startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        return f"{header}\n[[VISION_IMAGE:{abs_path}]]\n[Image - will be sent via OpenAI Response input_image, only muse-spark models supported]"
    if mime == "application/pdf" or ext == ".pdf":
        return f"{header}\n[[INPUT_FILE:{abs_path}]]\n[PDF - will be sent via OpenAI Response input_file file_data, only muse-spark models supported]"
    # For other binaries, try small base64 preview but also generic marker
    if mime.startswith("image/") or ext in (".zip",".bin",".exe",".docx",".xlsx"):
        return f"{header}\n[[INPUT_FILE:{abs_path}]]\n[Binary - only supported via Response]"
    try:
        with open(path, "rb") as f:
            data = f.read(3000)
        b64 = base64.b64encode(data).decode("ascii")
        preview = b64[:500] + ("..." if len(data) == 3000 else "")
        return f"{header}\n[Binary preview base64 (first 3000 bytes): {preview}]"
    except Exception as e:
        return f"{header}\n[Binary file - error: {e}]"

def _has_binary_embed(text: str) -> bool:
    return bool(text and ("[[VISION_IMAGE:" in text or "[[INPUT_FILE:" in text))

def expand_at_mentions(user_input: str) -> str:
    """Detect @filepath in input, embed file content. Supports @"path with spaces" and @path."""
    # Pattern: @ followed by path without spaces or with quotes
    # Example: @README.md, @"my file.txt", @agent_core/tools.py, @/tmp/test.pdf
    pattern = r'@(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))'
    def repl(match):
        raw = match.group(1) or match.group(2) or match.group(3)
        if not raw:
            return match.group(0)
        p = pathlib.Path(raw)
        # If path is relative and does not exist, check if it looks like a real file path
        # Avoid poisoning context for @decorator or user@example.com
        if not p.exists():
            # If it doesn't look like a file path (no / or . or too short), leave untouched
            if "/" not in raw and "\\" not in raw and "." not in raw:
                return match.group(0)
            # For email-like or decorator, don't inject [file not found] into LLM context
            # Just return original and log to stderr
            print(f"[embed] file not found (ignored): {raw}", file=sys.stderr)
            return match.group(0)
        try:
            if p.is_dir():
                try:
                    entries = sorted(os.listdir(p))
                    # Filter out noisy dirs for cleaner preview
                    filtered = [e for e in entries if e not in (".git", "node_modules", ".venv", "venv", "__pycache__", ".agent")]
                    # Show filtered, but count total
                    preview_entries = filtered[:20] if filtered else entries[:20]
                    preview = ", ".join(preview_entries)
                    if len(entries) > 20:
                        preview += f" ... (+{len(entries)-20} more)"
                    if len(filtered) != len(entries):
                        preview += f" [filtered {len(entries)-len(filtered)} hidden]"
                    return f"[Dir: {p} | {len(entries)} entries]\n{preview}"
                except Exception as e:
                    return f"[Dir: {p} | error: {e}]"
            if _is_binary_file(p):
                return _handle_binary_file(p)
            # Text file - check size to avoid blowing context
            try:
                size = p.stat().st_size
                # Prevent huge files from blowing context (limit 500KB per read, suggest using read tool for large files)
                from agent_core.config import MAX_READ_BYTES
                if size > MAX_READ_BYTES * 5:
                    return f"[File: {p} | {size} bytes too large ({size//1024}KB), use 'read' tool with offset/limit or grep]"
                text = p.read_text(encoding="utf-8", errors="ignore")
                return f"[File: {p} | {size} bytes]\n{text}"
            except Exception as e:
                return f"[File: {p} | read error: {e}]"
        except Exception as e:
            return f"[File: {raw} | error: {e}]"
    # Replace all @mentions
    expanded = re.sub(pattern, repl, user_input)
    if expanded != user_input:
        print(f"[embed] @file detected and embedded", file=sys.stderr)
    return expanded

def parse_args():
    p = argparse.ArgumentParser(description="Simple AI Agent - tool calling + filesystem + LLM")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model id (default {DEFAULT_MODEL})")
    p.add_argument("--session", "-s", default=None, help="Session id (ses_xxx). If not provided, create new")
    p.add_argument("--once", default=None, help="Run single prompt then exit (non-interactive)")
    p.add_argument("--continue", dest="cont", action="store_true", help="Continue last session")
    p.add_argument("--list-models", action="store_true", help="List models from opencode.ai then exit")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming (debug)")
    p.add_argument("--max-iterations", type=int, default=None, help="Override max loop iterations")
    p.add_argument("--base-url", default=None, help="Override OpenCode base URL")
    p.add_argument("--tool-call", default="ask", choices=["accept-all", "accept-fs", "ask"], help="Tool permission mode (default ask)")
    p.add_argument("--think", default=None, help="Thinking variant: none/low/medium/high/xhigh")
    return p.parse_args()

def run_shell_direct(command: str, workdir: str = "."):
    """Run shell via ! - terminatable, streaming output."""
    if not command.strip():
        print("[shell] empty command", file=sys.stderr)
        return
    print(f"[shell] $ {command}", file=sys.stderr)
    try:
        # Use Popen so it can be terminated via Ctrl-C
        proc = subprocess.Popen(command, shell=True, cwd=workdir or ".")
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\n[shell] terminating...", file=sys.stderr)
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except:
                try:
                    proc.kill()
                except:
                    pass
            print(f"[shell] terminated (exit {proc.returncode})", file=sys.stderr)
            return
        print(f"[shell] exit {proc.returncode}", file=sys.stderr)
    except Exception as e:
        print(f"[shell error] {e}", file=sys.stderr)

def handle_compact(ctx: ContextManager):
    before = ctx.token_usage()
    before_tokens = before["tokens"]
    before_msgs = len(ctx.messages)
    ctx.get_messages(force_compact=True)
    after = ctx.token_usage()
    print(f"[compact] {before_msgs} msgs, {before_tokens} tokens -> {len(ctx.messages)} msgs, {after['tokens']} tokens", file=sys.stderr)
    print(f"  before: {before_tokens}/{before['max']} ({before['percent']}%)", file=sys.stderr)
    print(f"  after : {after['tokens']}/{after['max']} ({after['percent']}%) compactions={after['compactions']}", file=sys.stderr)
    if len(ctx.messages) > 1:
        preview = ctx.messages[1].get("content","")[:500].replace("\n"," ")
        print(f"  summary preview: {preview}...", file=sys.stderr)

def handle_model(arg: str, llm: LLMClient, base_url=None, loop=None, think_variant: str = "none"):
    """arg can be empty (interactive), or direct model name. Return new think_variant (may be disabled if model not supported)."""
    arg = arg.strip() if arg else ""
    if arg:
        llm.model = arg
        print(f"[model] switched to {arg}", file=sys.stderr)
        if is_responses_model(arg):
            print(f"  -> model supports thinking (reasoning_effort)", file=sys.stderr)
        else:
            print(f"  -> model does not support thinking (will show None)", file=sys.stderr)
            if loop is not None and think_variant != "none":
                print(f"  -> thinking {think_variant} disabled because model not supported", file=sys.stderr)
                loop.extra_body.pop("reasoning_effort", None)
                loop.extra_body.pop("reasoning", None)
                think_variant = "none"
        return think_variant
    # interactive: fetch & select
    print("Fetching models...", file=sys.stderr)
    try:
        models = fetch_models(base_url=base_url) if base_url else fetch_models()
    except Exception as e:
        print(f"Error fetch models: {e}", file=sys.stderr)
        return think_variant
    # show ALL models so user can choose
    display = models
    print(f"Found {len(models)} models (showing all):", file=sys.stderr)
    for i, m in enumerate(display, 1):
        cur = " * current" if m.get("id")==llm.model else ""
        print(f" {i:3}. {m.get('id')} ({m.get('owned_by','')}){cur}", file=sys.stderr)
    print("Select model [1-{}, or type full name, Enter to cancel]: ".format(len(display)), end="", file=sys.stderr, flush=True)
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[model] cancelled", file=sys.stderr)
        return think_variant
    if not choice:
        print("[model] cancelled", file=sys.stderr)
        return think_variant
    # try to parse number
    sel = None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(display):
            sel = display[idx-1].get("id")
        else:
            print(f"[model] number out of range 1-{len(display)}", file=sys.stderr)
            return think_variant
    else:
        # check if name exists in models
        sel = choice
        # validate exists in list? still allow custom
        found = any(m.get("id")==choice for m in models)
        if not found:
            print(f"[model] warning: '{choice}' not in list, still using (custom)", file=sys.stderr)
    llm.model = sel
    print(f"[model] switched to {sel}", file=sys.stderr)
    if is_responses_model(sel):
        print(f"  -> model supports thinking (reasoning_effort)", file=sys.stderr)
    else:
        print(f"  -> model does not support thinking (will show None)", file=sys.stderr)
        if loop is not None and think_variant != "none":
            print(f"  -> thinking {think_variant} disabled because model not supported", file=sys.stderr)
            loop.extra_body.pop("reasoning_effort", None)
            loop.extra_body.pop("reasoning", None)
            think_variant = "none"
    return think_variant

def handle_usage(ctx: ContextManager):
    u = ctx.token_usage()
    print(f"[usage] tokens {u['tokens']}/{u['max']} ({u['percent']}%)", file=sys.stderr)
    print(f"  messages: {len(ctx.messages)} (system + {len(ctx.messages)-1} turns)", file=sys.stderr)
    print(f"  compactions: {u['compactions']}", file=sys.stderr)
    # detail per message estimate
    from agent_core.context import estimate_messages_tokens
    # breakdown system vs recent
    if len(ctx.messages) > 0:
        sys_tokens = estimate_messages_tokens([ctx.messages[0]])
        recent_tokens = u['tokens'] - sys_tokens
        print(f"  system: ~{sys_tokens} tokens, rest: ~{recent_tokens} tokens", file=sys.stderr)
    # threshold info
    from agent_core.config import COMPACTION_THRESHOLD
    thr = int(u['max'] * COMPACTION_THRESHOLD)
    print(f"  compaction threshold: {thr} tokens ({int(COMPACTION_THRESHOLD*100)}%)", file=sys.stderr)

def handle_skill(arg: str):
    arg = arg.strip() if arg else ""
    if not arg or arg.lower() in ("list", "ls"):
        print(skl_list(), file=sys.stderr)
        return
    # load specific skill
    print(load_skill(arg), file=sys.stderr)

def handle_reload(ctx: ContextManager, llm, permission_manager, think_variant):
    """Reload state without losing context: rebuild prompt lazy, keep messages."""
    print("[reload] reloading state...", file=sys.stderr)
    # Lazy: don't load all skills, just note
    catalog = ""  # lazy
    new_prompt = build_system_prompt(catalog)
    ctx.system_prompt = new_prompt
    ctx.messages[0] = {"role": "system", "content": new_prompt}
    u = ctx.token_usage()
    print(f"[reload] system prompt updated ({len(new_prompt)} chars) (lazy, skills via skill_list)", file=sys.stderr)
    print(f"  model: {llm.model} ({'supports thinking' if is_responses_model(llm.model) else 'no thinking'})", file=sys.stderr)
    print(f"  think: {think_variant}", file=sys.stderr)
    print(f"  permission: {permission_manager.status()}", file=sys.stderr)
    print(f"  tokens: {u['tokens']}/{u['max']} ({u['percent']}%) msgs={len(ctx.messages)}", file=sys.stderr)
    print("[reload] done - conversation context preserved", file=sys.stderr)

def handle_think(arg: str, llm: LLMClient, loop: AgentLoop, current_variant: str):
    arg = arg.strip().lower() if arg else ""
    supports = is_responses_model(llm.model)
    if not arg:
        # show list & current
        print(f"[think] model: {llm.model}", file=sys.stderr)
        if not supports:
            print("  thinking capability: None (model does not support reasoning_effort)", file=sys.stderr)
            print(f"  available variants: None", file=sys.stderr)
            print(f"  current: {current_variant} (no effect for this model)", file=sys.stderr)
        else:
            print(f"  thinking capability: Supported (via reasoning_effort)", file=sys.stderr)
            print(f"  available variants: {', '.join(THINK_VARIANTS)}", file=sys.stderr)
            print(f"    - none   : without thinking (default)", file=sys.stderr)
            print(f"    - low    : fast reasoning, token-efficient", file=sys.stderr)
            print(f"    - medium : balanced", file=sys.stderr)
            print(f"    - high   : deep, slow & expensive", file=sys.stderr)
            print(f"    - xhigh  : extra deep, slowest & most expensive", file=sys.stderr)
            print(f"  current: {current_variant}", file=sys.stderr)
            print(f"  usage: /think medium  or  /think xhigh", file=sys.stderr)
        return current_variant
    # set variant
    if arg not in THINK_VARIANTS:
        print(f"[think] invalid variant: {arg}. Choices: {', '.join(THINK_VARIANTS)}", file=sys.stderr)
        return current_variant
    if not supports and arg != "none":
        print(f"[think] warning: model {llm.model} does not support thinking, variant '{arg}' will be ignored (will show None)", file=sys.stderr)
        # still save but will not be used
    new_variant = arg
    # update loop extra_body
    if new_variant == "none":
        loop.extra_body.pop("reasoning_effort", None)
        loop.extra_body.pop("reasoning", None)
        print(f"[think] set to none (thinking disabled)", file=sys.stderr)
    else:
        loop.extra_body["reasoning_effort"] = new_variant
        # ensure no old reasoning conflicts
        loop.extra_body.pop("reasoning", None)
        print(f"[think] set to {new_variant} (reasoning_effort={new_variant})", file=sys.stderr)
    return new_variant

def handle_tool_call(arg: str, pm: PermissionManager):
    arg = arg.strip().lower() if arg else ""
    if not arg:
        print(f"[tool-call] current: {pm.status()}", file=sys.stderr)
        print("  choices: accept-all, accept-fs, ask", file=sys.stderr)
        print("    accept-all : all tools automatically without prompt", file=sys.stderr)
        print("    accept-fs  : filesystem (read/write/glob/grep) auto, bash still prompts", file=sys.stderr)
        print("    ask        : all require confirmation (default)", file=sys.stderr)
        return
    if arg not in ("accept-all", "accept-fs", "ask"):
        print(f"[tool-call] invalid mode: {arg}. Choices: accept-all, accept-fs, ask", file=sys.stderr)
        return
    pm.set_mode(arg)
    print(f"[tool-call] mode -> {pm.status()}", file=sys.stderr)

def handle_max_iter(arg: str, loop: AgentLoop, session_id: str = ""):
    arg = arg.strip() if arg else ""
    if not arg:
        print(f"[max-iter] current: {loop.max_iterations}", file=sys.stderr)
        print(f"  usage: /max-iter <n>  (e.g. /max-iter 50)", file=sys.stderr)
        print(f"  range: 1-100", file=sys.stderr)
        return
    try:
        n = int(arg)
        if n < 1 or n > 100:
            print(f"[max-iter] invalid value: {arg}. Must be 1-100", file=sys.stderr)
            return
        loop.max_iterations = n
        # persist for session as metadata (not as conversation message)
        if session_id:
            try:
                save_message(session_id, {"type": "config", "max_iterations": n, "content": f"[max-iter set to {n}]"})
            except Exception as e:
                print(f"[max-iter] failed to save: {e}", file=sys.stderr)
        print(f"[max-iter] set to {n} (saved for session {session_id})", file=sys.stderr)
    except ValueError:
        print(f"[max-iter] invalid number: {arg}. Usage: /max-iter <n>", file=sys.stderr)

def handle_info(ctx: ContextManager, llm: LLMClient, think_variant: str, session_id: str = "", permission_manager=None, loop=None):
    u = ctx.token_usage()
    total_msgs = len(ctx.messages)
    user_msgs = sum(1 for m in ctx.messages if m.get("role") == "user")
    if session_id:
        print(f"Session Id: {session_id}", file=sys.stderr)
    print(f"Model: {llm.model} [{think_variant}]", file=sys.stderr)
    print(f"Token: {u['tokens']}/{u['max']} {u['percent']}%", file=sys.stderr)
    print(f"Message: {user_msgs} (total {total_msgs}, system 1)", file=sys.stderr)
    if permission_manager:
        print(f"Permission: {permission_manager.status()}", file=sys.stderr)
    if loop is not None:
        print(f"Max-iter: {loop.max_iterations}", file=sys.stderr)

def main():
    args = parse_args()

    if args.list_models:
        print("Fetching models from opencode.ai...", file=sys.stderr)
        try:
            models = fetch_models(base_url=args.base_url) if args.base_url else fetch_models()
            print(f"Found {len(models)} models (all):")
            for m in models:
                print(f" - {m.get('id')}  ({m.get('owned_by','')})")
        except Exception as e:
            print(f"Error fetch models: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Session handling
    session_id = args.session
    if args.cont and not session_id:
        sessions = list_sessions()
        if sessions:
            session_id = sorted(sessions)[-1]
            print(f"[resume session {session_id}]", file=sys.stderr)
        else:
            print("[no previous session, creating new]", file=sys.stderr)

    if not session_id:
        session_id = generate_session_id()
        print(f"[new session {session_id}]", file=sys.stderr)
    else:
        if not session_id.startswith("ses_"):
            session_id = f"ses_{session_id}"

    # Build context - lazy skill/tool loading: don't load all skills at startup
    # Only tell LLM that skills are available via skill_list/skill_load, don't inject full catalog
    catalog = ""  # lazy, don't load at startup
    system_prompt = build_system_prompt(catalog)

    ctx = ContextManager(system_prompt=system_prompt)

    if args.session or args.cont:
        msgs = load_messages(session_id)
        if msgs:
            filtered = [m for m in msgs if m.get("role") != "system"]
            # Cap history to avoid blowing context on resume (keep last 80)
            if len(filtered) > 80:
                print(f"[history] capping {len(filtered)} messages to last 80 for resume", file=sys.stderr)
                filtered = filtered[-80:]
            if filtered:
                ctx.load(filtered)
                print(f"[loaded {len(filtered)} messages from history]", file=sys.stderr)

    llm_kwargs = {}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = LLMClient(model=args.model, session_id=session_id, **llm_kwargs)

    loop_kwargs = {}
    if args.max_iterations:
        loop_kwargs["max_iterations"] = args.max_iterations

    # Permission manager
    pm = PermissionManager(mode=args.tool_call)

    # Thinking variant - initial from CLI
    think_variant = args.think.lower().strip() if args.think else DEFAULT_THINK
    # If resuming session, try to use last model & think from history
    # CLI explicitly set values take precedence over history
    if args.session or args.cont:
        try:
            last_model, last_think = get_last_model_and_think(session_id)
            # Model: only use history if CLI didn't explicitly change from default
            if last_model:
                cli_model_explicit = args.model != DEFAULT_MODEL
                if cli_model_explicit:
                    print(f"[history] saved model {last_model} ignored (CLI --model {args.model} takes precedence)", file=sys.stderr)
                elif last_model != llm.model:
                    print(f"[history] using last model from session: {last_model} (CLI: {llm.model})", file=sys.stderr)
                    llm.model = last_model
            if last_think:
                cli_think_explicit = args.think is not None
                if cli_think_explicit:
                    print(f"[history] saved thinking {last_think} ignored (CLI --think {args.think} takes precedence)", file=sys.stderr)
                elif last_think != think_variant:
                    print(f"[history] using last thinking: {last_think} (CLI: {think_variant})", file=sys.stderr)
                    think_variant = last_think
            # Also load saved max-iter for session
            last_max = get_last_max_iterations(session_id)
            if last_max is not None:
                if args.max_iterations is None:
                    # Only use history if CLI didn't explicitly set --max-iterations
                    loop_kwargs["max_iterations"] = last_max
                    print(f"[history] using last max-iter from session: {last_max}", file=sys.stderr)
                else:
                    print(f"[history] saved max-iter {last_max} ignored (CLI --max-iterations {args.max_iterations} takes precedence)", file=sys.stderr)
        except Exception as e:
            print(f"[history] failed to load last model/think/max-iter: {e}", file=sys.stderr)

    extra_body = {}
    if think_variant and think_variant != "none":
        if is_responses_model(llm.model):
            extra_body["reasoning_effort"] = think_variant
        else:
            if think_variant != "none":
                print(f"[warning] model {llm.model} does not support thinking, variant '{think_variant}' ignored", file=sys.stderr)
                think_variant = "none"
    loop = AgentLoop(llm=llm, context=ctx, session_id=session_id, permission_manager=pm, extra_body=extra_body, **loop_kwargs)

    stream = not args.no_stream

    print(f"Agent ready | model={llm.model} | session={session_id} | stream={stream}", file=sys.stderr)
    print(f"History: {HISTS_DIR}/{session_id}.jsonl", file=sys.stderr)
    if catalog:
        print(f"Skills loaded: {catalog.count(chr(10))+1} skill(s)", file=sys.stderr)
    print(f"Permission: {pm.status()} | Think: {think_variant}", file=sys.stderr)
    print("Type message, Enter to send. /help for help, /exit to quit.", file=sys.stderr)
    print("Tip: !<command> for shell, /tool-call for permissions, /think for reasoning", file=sys.stderr)
    print("", file=sys.stderr)

    if args.once:
        # Expand @file before sending to loop
        expanded_once = expand_at_mentions(args.once)
        print(f"> {args.once}", file=sys.stderr)
        if expanded_once != args.once:
            print(f"[expanded] {expanded_once[:500]}...", file=sys.stderr)
        # If binary file is embedded and not Response model -> reject
        if _has_binary_embed(expanded_once) and not is_responses_model(llm.model):
            print("Model not supported - binary files only supported via OpenAI Response API (use model muse-spark-1.2/1.3)", file=sys.stderr)
            print("Model not supported", file=sys.stdout)
            return
        try:
            answer = loop.run(expanded_once, stream=stream)
        except KeyboardInterrupt:
            print("\n[interrupted]", file=sys.stderr)
            sys.exit(0)
        return

    # Interactive REPL
    while True:
        try:
            user_input = input("\n> ")
        except EOFError:
            print("\n[exit EOF]", file=sys.stderr)
            break
        except KeyboardInterrupt:
            print("\n[interrupted, type /exit to quit]", file=sys.stderr)
            continue

        if not user_input.strip():
            continue

        stripped = user_input.strip()

        # Shell escape ! (must be terminatable)
        if stripped.startswith("!"):
            shell_cmd = stripped[1:].strip()
            if not shell_cmd:
                print("[shell] usage: !<command>  e.g. !ls -la", file=sys.stderr)
                continue
            run_shell_direct(shell_cmd)
            continue

        # Command parsing - order matters, those with args first
        low = stripped.lower()

        # /compact
        if low == "/compact" or low == "/compress":
            handle_compact(ctx)
            continue

        # /model /models
        if low.startswith("/model"):
            # /model, /models, /model xxx, /models xxx
            # get arg after space
            if low.startswith("/models"):
                arg = stripped[7:].strip()  # len /models =7
            else:  # /model
                arg = stripped[6:].strip()  # len /model =6
            think_variant = handle_model(arg, llm, base_url=args.base_url, loop=loop, think_variant=think_variant)
            continue

        # /usage(s) /tokens
        if low in ("/usage", "/usages", "/tokens", "/token"):
            handle_usage(ctx)
            continue

        # /skill(s)
        if low.startswith("/skill"):
            # /skill, /skills, /skill xxx
            if low.startswith("/skills"):
                arg = stripped[7:].strip()
            else:  # /skill
                # handle /skill vs /skills done, but also check spaces
                # len /skill =6
                arg = stripped[6:].strip()
                # if low == "/skill" without arg -> list, if "/skill xxx" -> load
            handle_skill(arg)
            continue

        # /reload
        if low == "/reload":
            handle_reload(ctx, llm, pm, think_variant)
            continue

        # /think
        if low.startswith("/think"):
            arg = stripped[6:].strip()  # len /think =6
            think_variant = handle_think(arg, llm, loop, think_variant)
            continue

                # /tool-call
        if low.startswith("/tool-call") or low.startswith("/toolcall"):
            # support dash or not
            if low.startswith("/tool-call"):
                arg = stripped[10:].strip()  # len /tool-call =10
            else:
                arg = stripped[9:].strip()  # /toolcall =9
            handle_tool_call(arg, pm)
            continue

        # /max-iter /max-iterations
        if low.startswith("/max-iter"):
            # handle /max-iter and /max-iterations with optional hyphen
            # extract arg after command
            # len "/max-iter" = 9, "/max-iterations" = 15
            if low.startswith("/max-iterations"):
                arg = stripped[15:].strip()
            else:
                arg = stripped[9:].strip()
            handle_max_iter(arg, loop, session_id)
            continue

        # /info
        if low == "/info" or low.startswith("/info "):
            handle_info(ctx, llm, think_variant, session_id, pm, loop)
            continue

        # Legacy & helpers
        if stripped in ("/exit", "/quit", ":q"):
            print("[bye]", file=sys.stderr)
            break
        if low == "/help":
            print("""
Commands:
  /help                  show help
  /compact               force context compaction
  /model [name]          view model list & select (alias /models)
  /usage                 view token usage (alias /usages, /tokens)
  /skill [name]          view skill list / load skill (alias /skills)
  /reload                reload state without losing context
  /think [variant]       view/select thinking: none/low/medium/high/xhigh (None if model not supported)
  /tool-call [mode]      tool permission: accept-all, accept-fs, ask
  /max-iter [n]          view/set max iterations 1-100 (saved per session)
  /info                  show model, token, chat length, max-iter
  @file_name             embed file into prompt (text/pdf/photo), e.g.: @README.md @"photo.jpg" @doc.pdf
  !<command>             run shell directly (Ctrl-C to terminate)
  /clear                 clear context (reset)
  /session               show session id
  /exit, /quit           quit
""", file=sys.stderr)
            continue
        if low == "/clear":
            ctx = ContextManager(system_prompt=system_prompt)
            # need to update loop context reference
            loop.context = ctx
            print("[context cleared]", file=sys.stderr)
            continue
        if low == "/session":
            print(f"session: {session_id}", file=sys.stderr)
            print(f"model: {llm.model} | think: {think_variant} | permission: {pm.status()}", file=sys.stderr)
            continue
        # fallback old /models etc already handled above

        # Expand @file before sending (embed file + handle binary)
        expanded = expand_at_mentions(user_input)
        if expanded != user_input:
            print(f"[embed] file embedded into prompt ({len(expanded)} chars)", file=sys.stderr)
        # If binary file is embedded and not Response model -> reject
        if _has_binary_embed(expanded) and not is_responses_model(llm.model):
            print("Model not supported - binary files only supported via OpenAI Response API (use model muse-spark-1.2/1.3)", file=sys.stderr)
            print("Model not supported")
            continue
        # Send to agent loop
        try:
            loop.run(expanded, stream=stream)
        except KeyboardInterrupt:
            print("\n[loop interrupted]", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\n[error: {e}]", file=sys.stderr)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
