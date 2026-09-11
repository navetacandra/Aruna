#!/usr/bin/env python3
"""Agent CLI - sederhana tanpa TUI, input user selalu di bawah.

Usage:
  python agent.py                   # interactive REPL
  python agent.py --once "tanya"    # single turn
  python agent.py --model mimo-v2.5-free --session ses_xxx
  python agent.py --list-models
  python agent.py --continue        # lanjut sesi terakhir
"""
import argparse
import pathlib
import sys
import os

# Ensure package import works when run as script
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agent_core.config import DEFAULT_MODEL, HISTS_DIR
from agent_core.context import ContextManager
from agent_core.history import generate_session_id, list_sessions, load_messages
from agent_core.llm import LLMClient, fetch_models
from agent_core.loop import AgentLoop
from agent_core.prompts import build_system_prompt
from agent_core.skills import build_skills_catalog

def parse_args():
    p = argparse.ArgumentParser(description="AI Agent sederhana - tool calling + filesystem + LLM")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model id (default {DEFAULT_MODEL})")
    p.add_argument("--session", "-s", default=None, help="Session id (ses_xxx). Jika tidak ada, buat baru")
    p.add_argument("--once", default=None, help="Jalankan satu prompt lalu exit (non-interaktif)")
    p.add_argument("--continue", dest="cont", action="store_true", help="Lanjutkan sesi terakhir")
    p.add_argument("--list-models", action="store_true", help="List model dari opencode.ai lalu exit")
    p.add_argument("--no-stream", action="store_true", help="Matikan streaming (debug)")
    p.add_argument("--max-iterations", type=int, default=None, help="Override max loop iterations")
    p.add_argument("--base-url", default=None, help="Override OpenCode base URL")
    return p.parse_args()

def main():
    args = parse_args()

    if args.list_models:
        print("Fetching models from opencode.ai...", file=sys.stderr)
        try:
            models = fetch_models(base_url=args.base_url) if args.base_url else fetch_models()
            print(f"Found {len(models)} models:")
            for m in models[:30]:
                print(f" - {m.get('id')}  ({m.get('owned_by','')})")
            if len(models) > 30:
                print(f"...(+{len(models)-30} more)")
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
        # ensure format
        if not session_id.startswith("ses_"):
            # allow bare hex
            session_id = f"ses_{session_id}"

    # Build context: system prompt + skills catalog
    catalog = build_skills_catalog()
    system_prompt = build_system_prompt(catalog)

    ctx = ContextManager(system_prompt=system_prompt)

    # Jika resume, load history messages
    if args.session or args.cont:
        msgs = load_messages(session_id)
        if msgs:
            # filter system from history (kita sudah punya system baru)
            filtered = [m for m in msgs if m.get("role") != "system"]
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
    agent = AgentLoop(llm=llm, context=ctx, session_id=session_id, **loop_kwargs)

    stream = not args.no_stream

    print(f"Agent ready | model={args.model} | session={session_id} | stream={stream}", file=sys.stderr)
    print(f"History: {HISTS_DIR}/{session_id}.jsonl", file=sys.stderr)
    if catalog:
        print(f"Skills loaded: {catalog.count(chr(10))+1} skill(s)", file=sys.stderr)
    print("Ketik pesan, Enter untuk kirim. /help untuk bantuan, /exit untuk keluar.\n", file=sys.stderr)

    if args.once:
        # single turn mode: user input di bawah? tetap print prompt
        print(f"> {args.once}", file=sys.stderr)
        try:
            answer = agent.run(args.once, stream=stream)
        except KeyboardInterrupt:
            print("\n[interrupted]", file=sys.stderr)
            sys.exit(0)
        # answer sudah diprint via streaming, jika no-stream print sudah dilakukan
        # buat pemisah
        if stream:
            pass  # already printed
        return

    # Interactive REPL - sederhana tanpa TUI, input selalu di bawah
    while True:
        try:
            # prompt minimal di baris bawah
            user_input = input("\n> ")
        except EOFError:
            print("\n[exit EOF]", file=sys.stderr)
            break
        except KeyboardInterrupt:
            print("\n[interrupted, ketik /exit untuk keluar]", file=sys.stderr)
            continue

        if not user_input.strip():
            continue
        cmd = user_input.strip()
        if cmd in ("/exit", "/quit", ":q", "exit", "quit") and len(cmd) < 10:
            # hanya treat sebagai command jika memang pendek & tanpa konteks
            # tapi user mungkin mau tanya "quit?" - jadi cek exact
            if cmd in ("/exit", "/quit", ":q"):
                print("[bye]", file=sys.stderr)
                break
        if cmd == "/help":
            print("""
Commands:
  /help          tampilkan bantuan
  /exit, /quit   keluar
  /clear         bersihkan context (reset)
  /session       tampilkan session id
  /tokens        tampilkan token usage
  /models        list models
  /skills        list skills
""", file=sys.stderr)
            continue
        if cmd == "/clear":
            ctx = ContextManager(system_prompt=system_prompt)
            print("[context cleared]", file=sys.stderr)
            continue
        if cmd == "/session":
            print(f"session: {session_id}", file=sys.stderr)
            continue
        if cmd == "/tokens":
            u = ctx.token_usage()
            print(f"tokens {u['tokens']}/{u['max']} ({u['percent']}%) compactions={u['compactions']} msgs={len(ctx.messages)}", file=sys.stderr)
            continue
        if cmd == "/models":
            try:
                models = fetch_models(base_url=args.base_url) if args.base_url else fetch_models()
                print(f"{len(models)} models:", file=sys.stderr)
                for m in models[:20]:
                    print(f" - {m.get('id')}", file=sys.stderr)
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
            continue
        if cmd == "/skills":
            from agent_core.skills import list_skills
            print(list_skills(), file=sys.stderr)
            continue

        # Kirim ke agent loop - output agent langsung ke stdout, input berikutnya akan di bawah
        try:
            agent.run(user_input, stream=stream)
        except KeyboardInterrupt:
            print("\n[loop interrupted]", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\n[error: {e}]", file=sys.stderr)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
