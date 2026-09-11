# Aruna — Agent Ringan Untuk Nalar & Aksi

> **Aruna** (Sanskerta: अरुण) berarti *fajar kemerahan / cahaya awal matahari* — metafora agent yang menyingkap file dan konteks sebelum menalar & bertindak. Di sini **Aruna = Agent Ringan Untuk Nalar & Aksi**: `Ar` = Agent Ringan, `una` = Untuk Nalar & Aksi (ReAct loop + tool calling).

Agent Python minimal tanpa external dependencies yang fokus pada **tool calling, filesystem, LLM API, Agent Loop & Context Manager**. Mengacu pada `opencode.js` sebagai provider utama (`https://opencode.ai`) dengan **fallback multi-SDK** (OpenAI `v1/chat/completions`, `v1/responses`, Anthropic `v1/messages`) dan state per-model.

## Fitur Utama

- **Tanpa TUI merepotkan**: REPL sederhana `input()` - input user selalu di bawah, output streaming ke stdout, log ke stderr.
- **LLM API**: Port dari `opencode.js` - dukung `zen/v1/chat/completions` & `zen/v1/responses`, deteksi `muse-spark`, SSE streaming, akumulasi `tool_calls` delta, header `x-opencode-*`.
- **Fallback Multi-SDK**: `agent_core/providers.py:1` - fallback otomatis `openai_chat` ↔ `openai_responses` ↔ `anthropic_messages` dengan konversi payload/tool. Simpan state per-model di `.agent/llm_provider_state.json` (rubah saat fallback, pakai lagi jika model sama). History tetap format OpenAI, konversi hanya saat request Anthropic (`openai_messages_to_anthropic()`).
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
- **History**: `.agent/hists/<session_id>.jsonl` JSONL per turn `{role,content,tool_calls,model,think_variant,ts}`, `load_messages` untuk resume (filter hanya `role/content` untuk LLM), `get_last_model_and_think()` untuk restore model & thinking terakhir saat `--session`/`--continue` (override CLI).
- **Permission**: `agent_core/permissions.py:17` `PermissionManager` gate semua hardware tools (read/write/edit/glob/grep/bash) - mode `ask` prompt y/n/a/f, `accept-fs` fs auto, `accept-all` semua auto, `skill_list/load` selalu auto.
- **Shell Escape**: `!` jalankan `subprocess.Popen` streaming & terminatable (Ctrl-C → `terminate()` → `kill()`).
- **Thinking**: `/think` cek `is_responses_model()` - set `loop.extra_body["reasoning_effort"]` untuk muse-spark, `None` untuk model lain.

## Struktur

```
agent.py                 # CLI entry (interactive + --once, 8+ commands, permission, thinking)
agent_core/
  llm.py                 # LLM client + fallback chain + SSE
  providers.py           # Provider abstraction & conversion OpenAI↔Anthropic + state .agent/llm_provider_state.json
  tools.py               # 8 tool schemas & impl
  context.py             # ContextManager
  loop.py                # AgentLoop + permission gate + extra_body
  permissions.py         # PermissionManager (ask/accept-fs/accept-all)
  skills.py              # discover/load .agent/skills
  history.py             # JSONL session (tetap OpenAI format)
  config.py              # MAX_TOKENS, DEFAULT_MODEL, PROVIDER URL, dll
  prompts.py             # BASE_SYSTEM_PROMPT
.agent/
  skills/example/SKILL.md
  hists/<session_id>.jsonl
  llm_provider_state.json # state per-model: {model:{provider,endpoint,base_url}}
tests/test_agent.py      # 42 tests (mock LLM, permissions, commands, providers fallback)
opencode.js              # acuan provider (asli)
```

## Instalasi

Tanpa external deps, Python 3.10+ cukup:

```bash
git clone <repo>
cd agent
python -m unittest tests.test_agent -v  # 42 passed
```

## Penggunaan

```bash
# Interactive REPL (input selalu di bawah)
python agent.py
# > halo baca README.md
# /help untuk semua commands

# Single turn (non-interaktif, auto izin fs)
python agent.py --once "baca agent_core/llm.py dan jelaskan SSE" --model mimo-v2.5-free --tool-call accept-fs

# Session resume & permission
python agent.py --session ses_xxx --tool-call ask
python agent.py --continue --tool-call accept-all --think medium

# List model
python agent.py --list-models

# Custom model / limits
python agent.py -m muse-spark-1.2-contributor-free --max-iterations 15 --think high
OPENCODE_BASE_URL=https://opencode.ai AGENT_MAX_TOKENS=50000 python agent.py
```

### Commands REPL (semua tanpa TUI, input selalu di bawah)

| Command | Fungsi |
|---------|--------|
| `/compact` | Paksa kompaksi context (`ctx.get_messages(force_compact=True)`) |
| `/model [name]` / `/models` | Lihat daftar model `fetch_models()` dan pilih interaktif atau ` /model mimo-v2.5-free` |
| `/usage` / `/usages` / `/tokens` | Lihat `ctx.token_usage()` (tokens, max, %, compactions, msgs) |
| `/skill [name]` / `/skills` | `skill_list` semua atau `skill_load` detail `SKILL.md` |
| `/reload` | Reload state tanpa kehilangan konteks (rebuild catalog, system prompt, keep messages) |
| `/think [none/low/medium/high/xhigh]` | Lihat/pilih thinking variant; tampil `None` jika model tidak support `is_responses_model()` |
| `/tool-call [accept-all/accept-fs/ask]` | Izin hardware: `ask` semua konfirmasi, `accept-fs` fs auto, `accept-all` semua auto |
| `!echo hi` | Shell langsung `subprocess.Popen` terminatable via Ctrl-C |
| `/help` `/clear` `/session` `/exit` | Bantuan, reset context, tampil session, keluar |
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

Header LLM: `Authorization: Bearer public`, `User-Agent: opencode`, `x-opencode-session`, `x-opencode-request` (uuid), `x-opencode-project: global` - mirror `opencode.js:85`. Semua fallback hanya ke `OPENCODE_BASE_URL/zen/v1` (sesuai instruksi).

## Provider Fallback Detail

`agent_core/providers.py:320` `build_candidate_providers(model)` hanya fallback ke `opencode.ai/zen/v1`:
- `muse-spark` → `/zen/v1/responses` → `/zen/v1/chat/completions` → `/zen/v1/messages`
- `claude` → `/zen/v1/messages` → `/zen/v1/chat/completions` → `/zen/v1/responses`
- lain → `/zen/v1/chat/completions` → `/zen/v1/responses` → `/zen/v1/messages`
Saved state di `.agent/llm_provider_state.json` dipakai pertama jika model sama; jika request 404/401/422/400 → fallback ke berikutnya dan `update_provider_state()` simpan yang sukses. History tetap OpenAI (`history.py:60` `save_message` tidak diubah), konversi hanya di `prepare_payload_for_provider()` untuk Responses (`instructions`+`input:[{type:input_text/output_text}]`+`store:false`) dan Anthropic (`system` pisah, `tool_calls`→`tool_use`, `tool`→`tool_result`, `tools` parameter→`input_schema`).

SSE parsing: Chat `data: {"choices":[{"delta":{"content":...}}]}` + `[DONE]`, Responses `event: response.output_text.delta` `data:{"delta":"..."}`, Anthropic `event: content_block_delta` `data:{"delta":{"text":...}}`/`input_json_delta`.

**429 handling** `agent_core/llm.py:90`: Jika `429 Too Many Requests` jangan fallback ke SDK lain dulu (bukan kesalahan format). Retry same provider max 5 kali dengan backoff `5 + (n-1)*3` detik (5, 8, 11, 14), tampil `"{error_message} {n} Retry on {x} seconds.."` per attempt, baru fallback ke SDK lain setelah 5 gagal. `time.sleep` interruptible via Ctrl-C/ESC.

Env: `OPENCODE_BASE_URL` (default `https://opencode.ai`), `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` jika diperlukan (tetap via opencode proxy).

## Pengujian

```bash
python -m unittest tests.test_agent -v
# 45 tests: LLM helpers, Context, Tools, Skills, History, Loop, Permissions, Commands, Providers (conversion, fallback hanya opencode, state, history, streaming, 429 retry)

# E2E real LLM (butuh network)
python agent.py --once "Baca agent.py dan ringkas" --tool-call accept-fs --model mimo-v2.5-free  # fallback hanya opencode/zen/v1
echo -e "baca demo.txt\ny\ny" | python agent.py --tool-call ask  # test permission prompt
!echo hi  # di REPL jalankan shell terminatable
python -c "from agent_core.llm import fetch_models; print(fetch_models()[:3])"
cat .agent/llm_provider_state.json  # lihat state per-model: {model:{provider,endpoint,base_url}}
# Test 429 retry: mock 429 2x lalu sukses -> retry 5s,8s tanpa fallback dulu
```

## Tanpa External Deps

Semua pakai stdlib: `urllib.request`, `http.client`, `subprocess`, `pathlib`, `json`, `re`, `uuid`, `argparse`. Tidak ada `openai`, `requests`, `tiktoken`, `rich`/`textual`.

## Lisensi

MIT - Bebas dipakai. Provider opencode.ai gratis tier.
