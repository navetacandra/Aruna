# AI Agent Sederhana - Tanpa TUI

Agent Python minimal tanpa external dependencies yang fokus pada **tool calling, filesystem, LLM API, Agent Loop & Context Manager**. Mengacu pada `opencode.js` sebagai provider tunggal (`https://opencode.ai`).

## Fitur Utama

- **Tanpa TUI merepotkan**: REPL sederhana `input()` - input user selalu di bawah, output streaming ke stdout, log ke stderr.
- **LLM API**: Port dari `opencode.js` - dukung `zen/v1/chat/completions` & `zen/v1/responses`, deteksi `muse-spark`, SSE streaming, akumulasi `tool_calls` delta, header `x-opencode-*`.
- **Tool Calling Kuat**: 8 tools OpenAI-compatible:
  - `read` - baca file / list dir dengan `offset/limit`, truncate line >2000
  - `write` - tulis file (mkdir -p otomatis)
  - `edit` - exact replace, cegah multi-match tanpa `replaceAll`
  - `glob` - `**/*.py` via `pathlib` + fallback `fnmatch`
  - `grep` - regex search, skip `.git/node_modules/.venv`, cap 200 hits
  - `bash` - `subprocess.run` dengan `timeout`, `workdir`
  - `skill_list` / `skill_load` - skill system
  - Truncate output >20k chars, error handling ramah.
- **Agent Loop**: ReAct iterative `max 25` iter - panggil LLM, eksekusi tools sequential, feed back, lanjut sampai `finish_reason != tool_calls`.
- **Context Manager** (krusial): estimasi token `chars/4`, kompaksi otomatis saat >85% `MAX_CONTEXT_TOKENS` (default 120k):
  - Keep `system` + `KEEP_RECENT_MESSAGES=8` terbaru
  - Middle diringkas jadi 1 `user` note (prune >4000 chars)
  - Iterative + emergency compaction jika masih >95%
  - Truncate tool output besar
- **Skills**: `.agent/skills/<nama>/SKILL.md` (frontmatter `name/description`). Katalog disuntik ke system prompt, tool `skill_list`/`skill_load` lazy-load.
- **History**: `.agent/hists/<session_id>.jsonl` JSONL per turn, `load_messages` untuk resume.

## Struktur

```
agent.py                 # CLI entry (interactive + --once)
agent_core/
  llm.py                 # LLM client + SSE + is_responses_model
  tools.py               # 8 tool schemas & impl
  context.py             # ContextManager
  loop.py                # AgentLoop
  skills.py              # discover/load .agent/skills
  history.py             # JSONL session
  config.py              # MAX_TOKENS, DEFAULT_MODEL, dll
  prompts.py             # BASE_SYSTEM_PROMPT
.agent/
  skills/example/SKILL.md
  hists/<session_id>.jsonl
tests/test_agent.py      # 20 tests (mock LLM)
opencode.js              # acuan provider (asli)
```

## Instalasi

Tanpa external deps, Python 3.10+ cukup:

```bash
git clone <repo>
cd agent
python -m unittest tests.test_agent -v  # 20 passed
```

## Penggunaan

```bash
# Interactive REPL (input selalu di bawah)
python agent.py
# > halo baca README.md
# /help, /clear, /tokens, /session, /skills, /models, /exit

# Single turn (non-interaktif)
python agent.py --once "baca agent_core/llm.py dan jelaskan SSE" --model mimo-v2.5-free

# Session resume
python agent.py --session ses_xxx
python agent.py --continue   # lanjut sesi terakhir

# List model
python agent.py --list-models

# Custom model / limits
python agent.py -m muse-spark-1.2-contributor-free --max-iterations 15
OPENCODE_BASE_URL=https://opencode.ai AGENT_MAX_TOKENS=50000 python agent.py
```

## Contoh Skill

Buat folder `.agent/skills/<nama>/SKILL.md`:

```markdown
---
name: my-skill
description: Panduan deploy
---
# My Skill
1. `read` file config
2. `bash` npm run build
...
```

Agent akan otomatis deteksi via `skill_list`. Cek ` .agent/skills/example/SKILL.md` sebagai contoh.

## Context Manager Detail

`agent_core/context.py:12` `estimate_tokens()` pakai `len(text)//4`. `get_messages()` cek `tokens > max*0.85` lalu `_compact()`:

- Hitung `middle_tokens`, buat ringkasan per message 1 baris (max 400 chars + tool_calls)
- Rebuild ` [system, summary, *recent]`
- Loop iterative hingga `tokens < threshold` atau 5 attempts, plus emergency jika >95%.

Statistik: `ctx.token_usage()` -> `{tokens, max, percent, compactions}`. Lihat `/tokens` di REPL.

## Agent Loop Detail

`agent_core/loop.py:34` `AgentLoop.run(user_input, stream)`:

1. `context.add_user()` + `save_message()`
2. Loop `max_iterations`:
   - `llm.chat(messages, tools, stream, on_delta)` - streaming cetak langsung
   - Jika `tool_calls`: simpan assistant + eksekusi `execute_tool(name, args)` sequential, log preview, `add_tool_result`, lanjut loop
   - Jika tidak: simpan assistant, return `content`
3. History JSONL tiap turn.

Header LLM: `Authorization: Bearer public`, `User-Agent: opencode`, `x-opencode-session`, `x-opencode-request` (uuid), `x-opencode-project: global` - mirror `opencode.js:85`.

## Pengujian

```bash
python -m unittest tests.test_agent -v
# 20 tests: LLM helpers, Context compact, Tools read/write/edit/glob/grep/bash, Skills, History, Loop mock

# E2E real LLM (butuh network)
python agent.py --once "Baca agent.py dan ringkas"          # tool calling + history
python -c "from agent_core.llm import fetch_models; print(fetch_models()[:3])"
```

## Tanpa External Deps

Semua pakai stdlib: `urllib.request`, `http.client`, `subprocess`, `pathlib`, `json`, `re`, `uuid`, `argparse`. Tidak ada `openai`, `requests`, `tiktoken`, `rich`/`textual`.

## Lisensi

MIT - Bebas dipakai. Provider opencode.ai gratis tier.
