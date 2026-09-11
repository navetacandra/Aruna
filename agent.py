#!/usr/bin/env python3
"""Agent CLI - sederhana tanpa TUI, input user selalu di bawah.

Usage:
  python agent.py                   # interactive REPL
  python agent.py --once "tanya"    # single turn
  python agent.py --model mimo-v2.5-free --session ses_xxx
  python agent.py --list-models
  python agent.py --continue        # lanjut sesi terakhir

Commands REPL:
  /compact                 kompaksi context manual
  /model(s) [name]         lihat daftar model & pilih (interaktif jika tanpa arg)
  /usage(s)                lihat penggunaan token
  /skill(s) [name]         lihat daftar skill / load skill
  /reload                  reload state tanpa kehilangan konteks
  /think [variant]         lihat/pilih thinking variant (none/low/medium/high/xhigh)
  /tool-call [mode]        izin tool: accept-all, accept-fs, ask
  /info                    tampilkan model, token, panjang chat
  !<command>               jalankan shell langsung (terminatable via Ctrl-C)
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
from agent_core.history import generate_session_id, list_sessions, load_messages, get_last_model_and_think
from agent_core.llm import LLMClient, fetch_models, is_responses_model
from agent_core.loop import AgentLoop
from agent_core.prompts import build_system_prompt
from agent_core.skills import build_skills_catalog, list_skills as skl_list, load_skill
from agent_core.permissions import PermissionManager

THINK_VARIANTS = ["none", "low", "medium", "high", "xhigh"]  # none = non-thinking, xhigh = extra high
DEFAULT_THINK = "none"

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
    p.add_argument("--tool-call", default="ask", choices=["accept-all", "accept-fs", "ask"], help="Mode izin tool (default ask)")
    p.add_argument("--think", default=None, help="Thinking variant: none/low/medium/high/xhigh")
    return p.parse_args()

def run_shell_direct(command: str, workdir: str = "."):
    """Jalankan shell via ! - terminatable, streaming output."""
    if not command.strip():
        print("[shell] empty command", file=sys.stderr)
        return
    print(f"[shell] $ {command}", file=sys.stderr)
    try:
        # Gunakan Popen agar bisa di-terminate via Ctrl-C
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
    """arg bisa kosong (interactive), atau nama model langsung. Return think_variant baru (mungkin dimatikan jika model tidak support)."""
    arg = arg.strip() if arg else ""
    if arg:
        llm.model = arg
        print(f"[model] switched to {arg}", file=sys.stderr)
        if is_responses_model(arg):
            print(f"  -> model mendukung thinking (reasoning_effort)", file=sys.stderr)
        else:
            print(f"  -> model tidak mendukung thinking (akan tampil None)", file=sys.stderr)
            if loop is not None and think_variant != "none":
                print(f"  -> thinking {think_variant} dimatikan karena model tidak support", file=sys.stderr)
                loop.extra_body.pop("reasoning_effort", None)
                loop.extra_body.pop("reasoning", None)
                think_variant = "none"
        return think_variant
    # interactive: fetch & pilih
    print("Fetching models...", file=sys.stderr)
    try:
        models = fetch_models(base_url=base_url) if base_url else fetch_models()
    except Exception as e:
        print(f"Error fetch models: {e}", file=sys.stderr)
        return think_variant
    # tampilkan SEMUA models agar pengguna dapat memilih
    display = models
    print(f"Found {len(models)} models (menampilkan semua):", file=sys.stderr)
    for i, m in enumerate(display, 1):
        cur = " * current" if m.get("id")==llm.model else ""
        print(f" {i:3}. {m.get('id')} ({m.get('owned_by','')}){cur}", file=sys.stderr)
    print("Pilih model [1-{}, atau ketik nama lengkap, Enter untuk batal]: ".format(len(display)), end="", file=sys.stderr, flush=True)
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[model] batal", file=sys.stderr)
        return think_variant
    if not choice:
        print("[model] batal", file=sys.stderr)
        return think_variant
    # coba parse angka
    sel = None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(display):
            sel = display[idx-1].get("id")
        else:
            print(f"[model] nomor di luar range 1-{len(display)}", file=sys.stderr)
            return think_variant
    else:
        # cek apakah nama ada di models
        sel = choice
        # validasi ada di list? tetap izinkan custom
        found = any(m.get("id")==choice for m in models)
        if not found:
            print(f"[model] warning: '{choice}' tidak ada di daftar, tetap gunakan (custom)", file=sys.stderr)
    llm.model = sel
    print(f"[model] switched to {sel}", file=sys.stderr)
    if is_responses_model(sel):
        print(f"  -> model mendukung thinking (reasoning_effort)", file=sys.stderr)
    else:
        print(f"  -> model tidak mendukung thinking (akan tampil None)", file=sys.stderr)
        if loop is not None and think_variant != "none":
            print(f"  -> thinking {think_variant} dimatikan karena model tidak support", file=sys.stderr)
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
    # breakdown sistem vs recent
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
    """Reload state tanpa kehilangan konteks: rebuild catalog, system prompt, keep messages."""
    print("[reload] reloading state...", file=sys.stderr)
    catalog = build_skills_catalog()
    new_prompt = build_system_prompt(catalog)
    # keep current messages selain system, ganti system
    old_messages = ctx.messages[1:] if len(ctx.messages) > 1 else []
    ctx.system_prompt = new_prompt
    ctx.messages[0] = {"role": "system", "content": new_prompt}
    # tidak hilangkan history messages, cuma update system
    # estimasi ulang token
    u = ctx.token_usage()
    print(f"[reload] system prompt updated ({len(new_prompt)} chars)", file=sys.stderr)
    print(f"  skills: {catalog.count(chr(10))+1 if catalog else 0} skill(s)", file=sys.stderr)
    print(f"  model: {llm.model} ({'supports thinking' if is_responses_model(llm.model) else 'no thinking'})", file=sys.stderr)
    print(f"  think: {think_variant}", file=sys.stderr)
    print(f"  permission: {permission_manager.status()}", file=sys.stderr)
    print(f"  tokens: {u['tokens']}/{u['max']} ({u['percent']}%) msgs={len(ctx.messages)}", file=sys.stderr)
    # juga re-set extra_body jika thinking berubah? tidak perlu, sudah di loop
    print("[reload] done - konteks percakapan tetap dipertahankan", file=sys.stderr)

def handle_think(arg: str, llm: LLMClient, loop: AgentLoop, current_variant: str):
    arg = arg.strip().lower() if arg else ""
    supports = is_responses_model(llm.model)
    if not arg:
        # tampilkan daftar & current
        print(f"[think] model: {llm.model}", file=sys.stderr)
        if not supports:
            print("  kemampuan thinking: None (model tidak mendukung reasoning_effort)", file=sys.stderr)
            print(f"  varian tersedia: None", file=sys.stderr)
            print(f"  current: {current_variant} (tidak berpengaruh untuk model ini)", file=sys.stderr)
        else:
            print(f"  kemampuan thinking: Supported (via reasoning_effort)", file=sys.stderr)
            print(f"  varian tersedia: {', '.join(THINK_VARIANTS)}", file=sys.stderr)
            print(f"    - none   : tanpa thinking (default)")
            print(f"    - low    : reasoning cepat, hemat token")
            print(f"    - medium : seimbang")
            print(f"    - high   : mendalam, lambat & mahal")
            print(f"    - xhigh  : ekstra mendalam, paling lambat & mahal")
            print(f"  current: {current_variant}", file=sys.stderr)
            print(f"  cara pakai: /think medium  atau  /think xhigh", file=sys.stderr)
        return current_variant
    # set variant
    if arg not in THINK_VARIANTS:
        print(f"[think] varian tidak valid: {arg}. Pilihan: {', '.join(THINK_VARIANTS)}", file=sys.stderr)
        return current_variant
    if not supports and arg != "none":
        print(f"[think] warning: model {llm.model} tidak mendukung thinking, varian '{arg}' akan diabaikan (akan tampil None)", file=sys.stderr)
        # tetap simpan tapi tidak akan dipakai
    new_variant = arg
    # update loop extra_body
    if new_variant == "none":
        loop.extra_body.pop("reasoning_effort", None)
        loop.extra_body.pop("reasoning", None)
        print(f"[think] set to none (thinking dimatikan)", file=sys.stderr)
    else:
        loop.extra_body["reasoning_effort"] = new_variant
        # pastikan tidak ada reasoning lama yang conflict
        loop.extra_body.pop("reasoning", None)
        print(f"[think] set to {new_variant} (reasoning_effort={new_variant})", file=sys.stderr)
    return new_variant

def handle_tool_call(arg: str, pm: PermissionManager):
    arg = arg.strip().lower() if arg else ""
    if not arg:
        print(f"[tool-call] current: {pm.status()}", file=sys.stderr)
        print("  pilihan: accept-all, accept-fs, ask", file=sys.stderr)
        print("    accept-all : semua tool otomatis tanpa tanya")
        print("    accept-fs  : filesystem (read/write/glob/grep) auto, bash tetap tanya")
        print("    ask        : semua hardware konfirmasi (default)", file=sys.stderr)
        return
    if arg not in ("accept-all", "accept-fs", "ask"):
        print(f"[tool-call] mode tidak valid: {arg}. Pilihan: accept-all, accept-fs, ask", file=sys.stderr)
        return
    pm.set_mode(arg)
    print(f"[tool-call] mode -> {pm.status()}", file=sys.stderr)

def handle_info(ctx: ContextManager, llm: LLMClient, think_variant: str, session_id: str = "", permission_manager=None):
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

def main():
    args = parse_args()

    if args.list_models:
        print("Fetching models from opencode.ai...", file=sys.stderr)
        try:
            models = fetch_models(base_url=args.base_url) if args.base_url else fetch_models()
            print(f"Found {len(models)} models (semua):")
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

    # Build context
    catalog = build_skills_catalog()
    system_prompt = build_system_prompt(catalog)

    ctx = ContextManager(system_prompt=system_prompt)

    if args.session or args.cont:
        msgs = load_messages(session_id)
        if msgs:
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

    # Permission manager
    pm = PermissionManager(mode=args.tool_call)

    # Thinking variant - awal dari CLI
    think_variant = args.think.lower().strip() if args.think else DEFAULT_THINK
    # Jika resume session, coba pakai model & think terakhir dari history (sesuai instruksi)
    if args.session or args.cont:
        try:
            last_model, last_think = get_last_model_and_think(session_id)
            if last_model:
                if last_model != llm.model:
                    print(f"[history] menggunakan model terakhir dari session: {last_model} (CLI: {llm.model})", file=sys.stderr)
                    llm.model = last_model
            if last_think:
                # last_think dari history bisa "none" atau varian
                if last_think != think_variant:
                    # Jika CLI tidak eksplisit (args.think is None) atau history berbeda, pakai history sebagai sumber kebenaran terakhir
                    # Karena history menyimpan yang benar-benar dipakai terakhir, kita prioritaskan history
                    print(f"[history] menggunakan thinking terakhir: {last_think} (CLI: {think_variant})", file=sys.stderr)
                    think_variant = last_think
        except Exception as e:
            print(f"[history] gagal load last model/think: {e}", file=sys.stderr)

    extra_body = {}
    if think_variant and think_variant != "none":
        if is_responses_model(llm.model):
            extra_body["reasoning_effort"] = think_variant
        else:
            if think_variant != "none":
                print(f"[warning] model {llm.model} tidak support thinking, variant '{think_variant}' diabaikan", file=sys.stderr)
                think_variant = "none"
    loop = AgentLoop(llm=llm, context=ctx, session_id=session_id, permission_manager=pm, extra_body=extra_body, **loop_kwargs)

    stream = not args.no_stream

    print(f"Agent ready | model={llm.model} | session={session_id} | stream={stream}", file=sys.stderr)
    print(f"History: {HISTS_DIR}/{session_id}.jsonl", file=sys.stderr)
    if catalog:
        print(f"Skills loaded: {catalog.count(chr(10))+1} skill(s)", file=sys.stderr)
    print(f"Permission: {pm.status()} | Think: {think_variant}", file=sys.stderr)
    print("Ketik pesan, Enter untuk kirim. /help untuk bantuan, /exit untuk keluar.", file=sys.stderr)
    print("Tip: !<command> untuk shell, /tool-call untuk izin, /think untuk reasoning", file=sys.stderr)
    print("", file=sys.stderr)

    if args.once:
        print(f"> {args.once}", file=sys.stderr)
        try:
            answer = loop.run(args.once, stream=stream)
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
            print("\n[interrupted, ketik /exit untuk keluar]", file=sys.stderr)
            continue

        if not user_input.strip():
            continue

        stripped = user_input.strip()

        # Shell escape ! (harus bisa diterminate)
        if stripped.startswith("!"):
            shell_cmd = stripped[1:].strip()
            if not shell_cmd:
                print("[shell] usage: !<command>  e.g. !ls -la", file=sys.stderr)
                continue
            run_shell_direct(shell_cmd)
            continue

        # Command parsing - urutan penting, yang pakai arg dulu
        low = stripped.lower()

        # /compact
        if low == "/compact" or low == "/compress":
            handle_compact(ctx)
            continue

        # /model /models
        if low.startswith("/model"):
            # /model, /models, /model xxx, /models xxx
            # ambil arg setelah spasi
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
                # handle /skill vs /skills sudah, tapi cek juga spasi
                # len /skill =6
                arg = stripped[6:].strip()
                # jika low == "/skill" tanpa arg -> list, jika "/skill xxx" -> load
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
            # support dash atau tidak
            if low.startswith("/tool-call"):
                arg = stripped[10:].strip()  # len /tool-call =10
            else:
                arg = stripped[9:].strip()  # /toolcall =9
            handle_tool_call(arg, pm)
            continue

        # /info
        if low == "/info" or low.startswith("/info "):
            handle_info(ctx, llm, think_variant, session_id, pm)
            continue

        # Legacy & helpers
        if stripped in ("/exit", "/quit", ":q"):
            print("[bye]", file=sys.stderr)
            break
        if low == "/help":
            print("""
Commands:
  /help                  tampilkan bantuan
  /compact               paksa kompaksi context
  /model [name]          lihat daftar model & pilih (alias /models)
  /usage                 lihat penggunaan token (alias /usages, /tokens)
  /skill [name]          lihat daftar skill / load skill (alias /skills)
  /reload                reload state tanpa kehilangan konteks
  /think [variant]       lihat/pilih thinking: none/low/medium/high/xhigh (None jika model tidak support)
  /tool-call [mode]      izin tool: accept-all, accept-fs, ask
  /info                  tampilkan model, token, panjang chat
  !<command>             jalankan shell langsung (Ctrl-C untuk terminate)
  /clear                 bersihkan context (reset)
  /session               tampilkan session id
  /exit, /quit           keluar
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
        # fallback /models lama etc sudah di-handle di atas

        # Kirim ke agent loop
        try:
            loop.run(user_input, stream=stream)
        except KeyboardInterrupt:
            print("\n[loop interrupted]", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\n[error: {e}]", file=sys.stderr)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
