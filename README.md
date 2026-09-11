# Aruna — Lightweight Agent for Reasoning & Action

> **Aruna** (Sanskrit: अरुण) means *reddish dawn / early sunlight* — a metaphor for an agent that uncovers files and context before reasoning and acting. Here **Aruna = Lightweight Agent for Reasoning & Action**: `Ar` = Agent Lightweight, `una` = for Reasoning & Action (ReAct loop + tool calling).

Minimal Python agent with no external dependencies, focused on **tool calling, filesystem, LLM API, Agent Loop & Context Manager**. Based on `opencode.js` as the main provider (`https://opencode.ai`) with **multi-SDK fallback** (OpenAI `v1/chat/completions`, `v1/responses`, Anthropic `v1/messages`) and per-model state.

## Key Features

- **No cumbersome TUI**: Simple `input()` REPL - user input always at the bottom, streaming output to stdout, logs to stderr.
- **LLM API**: Ported from `opencode.js` - supports `zen/v1/chat/completions` & `zen/v1/responses`, `muse-spark` detection, SSE streaming, `tool_calls` delta accumulation, `x-opencode-*` headers.
- **Multi-SDK Fallback**: `agent_core/providers.py:1` - automatic fallback `openai_chat` ↔ `openai_responses` ↔ `anthropic_messages` with payload/tool conversion. Stores per-model state in `.agent/llm_provider_state.json` (updates on fallback, reused if same model). History stays OpenAI format, conversion only on Anthropic requests (`openai_messages_to_anthropic()`).
- **Powerful Tool Calling**: 8 OpenAI-compatible tools:
  - `read` - read file / list directory with `offset/limit`, truncate lines >2000
  - `write` - write file (auto mkdir -p)
  - `edit` - exact replace, prevents multi-match without `replaceAll`
  - `glob` - `**/*.py` via `pathlib` + `fnmatch` fallback
  - `grep` - regex search, skips `.git/node_modules/.venv`, cap 200 hits
  - `bash` - `subprocess.run` with `timeout`, `workdir`
  - `skill_list` / `skill_load` - skill system
  - Truncate output >20k chars, friendly error handling.
- **Agent Loop**: Iterative ReAct `max 25` iters - calls LLM, executes tools sequentially, feeds back, continues until `finish_reason != tool_calls`.
- **Context Manager** (crucial): token estimation `chars/4`, auto-compaction when >85% `MAX_CONTEXT_TOKENS` (default 120k):
  - Keeps `system` + `KEEP_RECENT_MESSAGES=8` latest
  - Middle compressed into 1 `user` note (pruned >4000 chars)
  - Iterative + emergency compaction if still >95%
  - Truncates large tool outputs
- **Skills**: `.agent/skills/<name>/SKILL.md` (frontmatter `name/description`). Catalog injected into system prompt, `skill_list`/`skill_load` lazy-load.
- **History**: `.agent/hists/<session_id>.jsonl` JSONL per turn `{role,content,tool_calls,model,think_variant,ts}`, `load_messages` for resume (filters only `role/content` for LLM), `get_last_model_and_think()` to restore last model & thinking on `--session`/`--continue` (overrides CLI).
- **Permissions**: `agent_core/permissions.py:17` `PermissionManager` gates all hardware tools (read/write/edit/glob/grep/bash) - mode `ask` prompts y/n/a/f, `accept-fs` auto for fs, `accept-all` auto for all, `skill_list/load` always auto.
- **Shell Escape**: `!` runs `subprocess.Popen` streaming & terminatable (Ctrl-C → `terminate()` → `kill()`).
- **Thinking**: `/think` checks `is_responses_model()` - sets `loop.extra_body["reasoning_effort"]` for muse-spark, `None` for other models.
- **File Input**: `@file` embeds text/binary (image/PDF) via `[[VISION_IMAGE:]]`/`[[INPUT_FILE:]]` markers. Binary files only supported via OpenAI Responses API (`muse-spark-1.2/1.3`); other models return `Model not supported`.

## Structure

```
agent.py                 # CLI entry (interactive + --once, 8+ commands, permissions, thinking)
agent_core/
  llm.py                 # LLM client + fallback chain + SSE
  providers.py           # Provider abstraction & OpenAI↔Anthropic conversion + state .agent/llm_provider_state.json
  tools.py               # 8 tool schemas & implementations
  context.py             # ContextManager
  loop.py                # AgentLoop + permission gate + extra_body
  permissions.py         # PermissionManager (ask/accept-fs/accept-all)
  skills.py              # discover/load .agent/skills
  history.py             # JSONL session (always OpenAI format)
  config.py              # MAX_TOKENS, DEFAULT_MODEL, PROVIDER URL, etc
  prompts.py             # BASE_SYSTEM_PROMPT
.agent/
  skills/example/SKILL.md
  hists/<session_id>.jsonl
  llm_provider_state.json # per-model state: {model:{provider,endpoint,base_url}}
tests/test_agent.py      # 42 tests (mock LLM, permissions, commands, providers fallback)
opencode.js              # provider reference (original)
```

## Installation

No external deps, Python 3.10+ is enough:

```bash
git clone <repo>
cd agent
python -m unittest tests.test_agent -v  # 42 passed
```

## Usage

```bash
# Interactive REPL (input always at bottom)
python agent.py
# > hello read README.md
# /help for all commands

# Single turn (non-interactive, auto fs permission)
python agent.py --once "read agent_core/llm.py and explain SSE" --model mimo-v2.5-free --tool-call accept-fs

# Session resume & permissions
python agent.py --session ses_xxx --tool-call ask
python agent.py --continue --tool-call accept-all --think medium

# List models
python agent.py --list-models

# Custom model / limits
python agent.py -m muse-spark-1.2-contributor-free --max-iterations 15 --think high
OPENCODE_BASE_URL=https://opencode.ai AGENT_MAX_TOKENS=50000 python agent.py

# File input (binary only via Responses)
python agent.py --once '@"beboo.png" explain this image' --model muse-spark-1.2-contributor-free
python agent.py --once '@"report.pdf" summarize this document' --model muse-spark-1.2-contributor-free
```

### REPL Commands (all without TUI, input always at bottom)

| Command | Function |
|---------|--------|
| `/compact` | Force context compaction (`ctx.get_messages(force_compact=True)`) |
| `/model [name]` / `/models` | List models via `fetch_models()` and select interactively or `/model mimo-v2.5-free` |
| `/usage` / `/usages` / `/tokens` | Show `ctx.token_usage()` (tokens, max, %, compactions, msgs) |
| `/skill [name]` / `/skills` | `skill_list` all or `skill_load` detail `SKILL.md` |
| `/reload` | Reload state without losing context (rebuild catalog, system prompt, keep messages) |
| `/think [none/low/medium/high/xhigh]` | View/select thinking variant; shows `None` if model doesn't support `is_responses_model()` |
| `/tool-call [accept-all/accept-fs/ask]` | Hardware permissions: `ask` confirm all, `accept-fs` auto fs, `accept-all` auto all |
| `!echo hi` | Direct shell `subprocess.Popen` terminatable via Ctrl-C |
| `/help` `/clear` `/session` `/exit` | Help, reset context, show session, exit |
| `@file` | Embed file to prompt (text/pdf/photo), e.g. `@README.md @"photo.jpg" @doc.pdf` — binary only via Responses |
```

## Skill Example

Create folder `.agent/skills/<name>/SKILL.md`:

```markdown
---
name: my-skill
description: Deploy guide
---
# My Skill
1. `read` config file
2. `bash` npm run build
...
```

Agent auto-discovers via `skill_list`. See `.agent/skills/example/SKILL.md` as example.

## Context Manager Details

`agent_core/context.py:12` `estimate_tokens()` uses `len(text)//4`. `get_messages()` checks `tokens > max*0.85` then `_compact()`:

- Calculates `middle_tokens`, creates 1-line summary per message (max 400 chars + tool_calls)
- Rebuilds `[system, summary, *recent]`
- Loops iteratively until `tokens < threshold` or 5 attempts, plus emergency if >95%.

Stats: `ctx.token_usage()` -> `{tokens, max, percent, compactions}`. See `/tokens` in REPL.

## Agent Loop Details

`agent_core/loop.py:34` `AgentLoop.run(user_input, stream)`:

1. `context.add_user()` + `save_message()`
2. Loop `max_iterations`:
   - `llm.chat(messages, tools, stream, on_delta)` - streaming direct print
   - If `tool_calls`: save assistant + execute `execute_tool(name, args)` sequentially, log preview, `add_tool_result`, continue loop
   - If not: save assistant, return `content`
3. JSONL history per turn.

LLM Headers: `Authorization: Bearer public`, `User-Agent: opencode`, `x-opencode-session`, `x-opencode-request` (uuid), `x-opencode-project: global` - mirrored from `opencode.js:85`. All fallbacks only to `OPENCODE_BASE_URL/zen/v1` (as instructed).

## Provider Fallback Details

`agent_core/providers.py:320` `build_candidate_providers(model)` only fallbacks to `opencode.ai/zen/v1`:
- `muse-spark` → `/zen/v1/responses` → `/zen/v1/chat/completions` → `/zen/v1/messages`
- `claude` → `/zen/v1/messages` → `/zen/v1/chat/completions` → `/zen/v1/responses`
- others → `/zen/v1/chat/completions` → `/zen/v1/responses` → `/zen/v1/messages`
Saved state in `.agent/llm_provider_state.json` is used first if same model; on 404/401/422/400 → fallback to next and `update_provider_state()` stores the successful one. History stays OpenAI (`history.py:60` `save_message` unchanged), conversion only in `prepare_payload_for_provider()` for Responses (`instructions`+`input:[{type:input_text/output_text}]`+`store:false`) and Anthropic (`system` split, `tool_calls`→`tool_use`, `tool`→`tool_result`, `tools` param→`input_schema`).

SSE parsing: Chat `data: {"choices":[{"delta":{"content":...}}]}` + `[DONE]`, Responses `event: response.output_text.delta` `data:{"delta":"..."}`, Anthropic `event: content_block_delta` `data:{"delta":{"text":...}}`/`input_json_delta`.

**429 handling** `agent_core/llm.py:90`: On `429 Too Many Requests` do not fallback to other SDK yet (not a format error). Retry same provider max 5 times with backoff `5 + (n-1)*3` seconds (5, 8, 11, 14), show `"{error_message} {n} Retry on {x} seconds.."` per attempt, only fallback after 5 failures. `time.sleep` interruptible via Ctrl-C/ESC.

Env: `OPENCODE_BASE_URL` (default `https://opencode.ai`), `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` if needed (still via opencode proxy).

**File Input** `b.md`: Binary files (image/PDF) via `[[VISION_IMAGE:]]`/`[[INPUT_FILE:]]` markers → Responses `input_image`/`input_file file_data` (`data:mime;base64`). Only `muse-spark-1.2/1.3` (Responses) supported; others return `Model not supported`.

## Testing

```bash
python -m unittest tests.test_agent -v
# 45 tests: LLM helpers, Context, Tools, Skills, History, Loop, Permissions, Commands, Providers (conversion, fallback only opencode, state, history, streaming, 429 retry)

# E2E real LLM (requires network)
python agent.py --once "Read agent.py and summarize" --tool-call accept-fs --model mimo-v2.5-free  # fallback only opencode/zen/v1
echo -e "read demo.txt\ny\ny" | python agent.py --tool-call ask  # test permission prompt
!echo hi  # in REPL run terminatable shell
python -c "from agent_core.llm import fetch_models; print(fetch_models()[:3])"
cat .agent/llm_provider_state.json  # view per-model state: {model:{provider,endpoint,base_url}}
# Test 429 retry: mock 429 2x then success -> retry 5s,8s without fallback first
```

## No External Dependencies

All stdlib: `urllib.request`, `http.client`, `subprocess`, `pathlib`, `json`, `re`, `uuid`, `argparse`. No `openai`, `requests`, `tiktoken`, `rich`/`textual`.

## License

MIT - Free to use. Provider opencode.ai free tier.
