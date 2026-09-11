import json
import os
import pathlib
import tempfile
import unittest
import sys

# pastikan import agent_core
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from agent_core.llm import base_model_id, is_responses_model, generate_session_id, generate_request_id
from agent_core.context import ContextManager, estimate_tokens, estimate_messages_tokens
from agent_core.tools import execute_tool, tool_read, tool_write, tool_edit, tool_glob, tool_grep, tool_bash
from agent_core.skills import discover_skills, list_skills, load_skill, build_skills_catalog
from agent_core.history import generate_session_id as hist_sid, append_history, load_history, load_messages, save_message
from agent_core.loop import AgentLoop
from agent_core.config import MAX_TOOL_OUTPUT_CHARS

class TestLLMHelpers(unittest.TestCase):
    def test_base_model_id(self):
        self.assertEqual(base_model_id("muse-spark-1.2-contributor-free"), "muse-spark-1.2-contributor-free")
        self.assertEqual(base_model_id("muse-spark-1.2-contributor-free (free)"), "muse-spark-1.2-contributor-free")
        self.assertEqual(base_model_id("  mimo-v2.5-free (free) "), "mimo-v2.5-free")

    def test_is_responses(self):
        self.assertTrue(is_responses_model("muse-spark-1.2-contributor-free"))
        self.assertTrue(is_responses_model("muse-spark-1.3-contributor-free"))
        self.assertTrue(is_responses_model("muse-spark-custom"))
        self.assertFalse(is_responses_model("mimo-v2.5-free"))
        self.assertFalse(is_responses_model("gpt-4"))

    def test_ids(self):
        sid = generate_session_id()
        rid = generate_request_id()
        self.assertTrue(sid.startswith("ses_"))
        self.assertTrue(rid.startswith("msg_"))
        self.assertNotEqual(generate_session_id(), generate_session_id())

class TestContextManager(unittest.TestCase):
    def test_estimate(self):
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertGreater(estimate_tokens("a"*100), 10)

    def test_add_and_compact(self):
        ctx = ContextManager(system_prompt="system", max_tokens=200, keep_recent=2)
        # tokens awal kecil
        self.assertEqual(len(ctx.messages), 1)
        # tambah banyak pesan hingga trigger compaction
        for i in range(10):
            ctx.add_user("hello " + "x"*100)
            ctx.add_assistant("reply " + "y"*100)
        # paksa compact
        msgs = ctx.get_messages(force_compact=True)
        # harus tetap ada system di depan
        self.assertEqual(msgs[0]["role"], "system")
        # harus ada compaction note
        self.assertIn("COMPACTED", msgs[1]["content"])
        # recent keep 2
        self.assertEqual(len(msgs), 1 + 1 + 2)  # system + summary + 2 recent

    def test_truncate_tool_output(self):
        ctx = ContextManager(system_prompt="sys", max_tokens=100000)
        big = "a"* (MAX_TOOL_OUTPUT_CHARS + 5000)
        ctx.add_tool_result("call_1", "bash", big)
        last = ctx.messages[-1]
        self.assertLess(len(last["content"]), len(big))
        self.assertIn("TRUNCATED", last["content"])

    def test_token_usage(self):
        ctx = ContextManager(system_prompt="sys")
        ctx.add_user("hello")
        u = ctx.token_usage()
        self.assertIn("tokens", u)
        self.assertIn("percent", u)

class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_write_edit(self):
        f = self.base / "a.txt"
        # write
        res = tool_write(str(f), "hello\nworld\n")
        self.assertIn("OK", res)
        # read
        out = tool_read(str(f))
        self.assertIn("hello", out)
        self.assertIn("1: hello", out)
        # edit
        res2 = tool_edit(str(f), "world", "python")
        self.assertIn("OK", res2)
        out2 = tool_read(str(f))
        self.assertIn("python", out2)
        # edit replaceAll
        f2 = self.base / "b.txt"
        tool_write(str(f2), "a a a")
        tool_edit(str(f2), "a", "b", replaceAll=True)
        self.assertIn("b b b", tool_read(str(f2)))
        # read dir
        d_out = tool_read(str(self.base))
        self.assertIn("a.txt", d_out)

    def test_glob(self):
        (self.base / "sub").mkdir()
        tool_write(str(self.base / "sub" / "x.py"), "x")
        tool_write(str(self.base / "y.py"), "y")
        out = tool_glob("**/*.py", str(self.base))
        self.assertIn("x.py", out)
        self.assertIn("y.py", out)

    def test_grep(self):
        tool_write(str(self.base / "a.py"), "def hello():\n  pass\n# hello world")
        out = tool_grep("hello", str(self.base), "*.py")
        self.assertIn("hello", out)
        self.assertIn("a.py", out)

    def test_bash(self):
        out = tool_bash("echo hello", workdir=str(self.base))
        self.assertIn("hello", out)
        out2 = tool_bash("invalid_command_zzz_123", workdir=str(self.base))
        # harus tetap ada exit code
        self.assertIn("exit", out2.lower())

    def test_execute_tool_json_string(self):
        f = self.base / "c.txt"
        res = execute_tool("write", json.dumps({"filePath": str(f), "content": "hi"}))
        self.assertIn("OK", res)
        res2 = execute_tool("read", json.dumps({"filePath": str(f)}))
        self.assertIn("hi", res2)
        # unknown tool
        err = execute_tool("unknown_xyz", "{}")
        self.assertIn("tidak dikenal", err)

class TestSkills(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name) / ".agent" / "skills"
        self.base.mkdir(parents=True)
        # buat skill dummy
        sdir = self.base / "demo"
        sdir.mkdir()
        (sdir / "SKILL.md").write_text("# Demo\nIni skill demo untuk testing\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover(self):
        skills = discover_skills(str(self.base))
        self.assertIn("demo", skills)
        self.assertIn("Demo", skills["demo"]["description"])

    def test_list_load(self):
        ls = list_skills(str(self.base))
        self.assertIn("demo", ls)
        loaded = load_skill("demo", str(self.base))
        self.assertIn("Demo", loaded)
        err = load_skill("missing", str(self.base))
        self.assertIn("tidak ditemukan", err)

    def test_catalog(self):
        cat = build_skills_catalog(str(self.base))
        self.assertIn("demo", cat)

class TestHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hdir = pathlib.Path(self.tmp.name) / "hists"
        self.sid = hist_sid()

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_load(self):
        append_history(self.sid, {"role": "user", "content": "hi"}, str(self.hdir))
        append_history(self.sid, {"role": "assistant", "content": "hello"}, str(self.hdir))
        hist = load_history(self.sid, str(self.hdir))
        self.assertEqual(len(hist), 2)
        msgs = load_messages(self.sid, str(self.hdir))
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_save_message(self):
        save_message(self.sid, {"role": "user", "content": "test"}, str(self.hdir))
        self.assertEqual(len(load_history(self.sid, str(self.hdir))), 1)

# Mock LLM untuk test AgentLoop tanpa network
class MockLLM:
    def __init__(self, responses, model="mimo-v2.5-free"):
        # responses: list of dicts {content, tool_calls}
        self.responses = list(responses)
        self.calls = []
        self.extra_bodies = []
        self.model = model
    def chat(self, messages, tools=None, stream=False, on_delta=None, on_tool_delta=None, extra_body=None, timeout=120):
        self.calls.append(messages)
        self.extra_bodies.append(extra_body)
        if not self.responses:
            return {"content": "done", "tool_calls": None, "finish_reason": "stop"}
        r = self.responses.pop(0)
        if on_delta and r.get("content"):
            on_delta(r["content"])
        return r

class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hdir = pathlib.Path(self.tmp.name) / "hists"
        self.base = pathlib.Path(self.tmp.name) / "work"
        self.base.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_simple_no_tools(self):
        from agent_core.prompts import build_system_prompt
        ctx = ContextManager(system_prompt=build_system_prompt())
        mock = MockLLM([{"content": "halo dunia", "tool_calls": None, "finish_reason": "stop"}])
        sid = hist_sid()
        # patch history dir via monkey? AgentLoop pakai global HISTS_DIR, jadi kita override save_message path?
        # workaround: set HIDS_DIR env? Simpler: biarkan pakai default .agent/hists, tapi kita cek loop logic saja
        # Kita buat AgentLoop dan override history save dengan no-op via monkey patch
        import agent_core.history as hist_mod
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = str(self.hdir)
        try:
            loop = AgentLoop(llm=mock, context=ctx, session_id=sid, verbose=False)
            ans = loop.run("hi", stream=False)
            self.assertEqual(ans, "halo dunia")
            self.assertEqual(len(mock.calls), 1)
            # history tersimpan
            self.assertGreater(len(load_history(sid, str(self.hdir))), 0)
        finally:
            hist_mod.HISTS_DIR = orig

    def test_tool_calling_loop(self):
        # Buat file untuk dibaca via tool
        f = self.base / "data.txt"
        f.write_text("secret 123", encoding="utf-8")
        ctx = ContextManager(system_prompt="system")
        # Mock: pertama minta tool read, kedua jawab final
        mock = MockLLM([
            {"content": "", "tool_calls": [{"id": "call_1", "type":"function","function":{"name":"read","arguments": json.dumps({"filePath": str(f)})}}], "finish_reason":"tool_calls"},
            {"content": "isi file adalah secret 123", "tool_calls": None, "finish_reason":"stop"}
        ])
        import agent_core.history as hist_mod
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = str(self.hdir)
        try:
            loop = AgentLoop(llm=mock, context=ctx, session_id=hist_sid(), verbose=False)
            ans = loop.run("baca file data.txt", stream=False)
            self.assertIn("secret 123", ans)
            self.assertEqual(len(mock.calls), 2)
            # context harus punya tool result
            roles = [m["role"] for m in ctx.messages]
            self.assertIn("tool", roles)
        finally:
            hist_mod.HISTS_DIR = orig

    def test_max_iterations(self):
        ctx = ContextManager(system_prompt="sys")
        # selalu minta tool, tak pernah selesai
        mock = MockLLM([
            {"content": "", "tool_calls": [{"id":"c1","type":"function","function":{"name":"bash","arguments": json.dumps({"command":"echo hi"})}}], "finish_reason":"tool_calls"}
        ] * 5)
        import agent_core.history as hist_mod
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = str(self.hdir)
        try:
            loop = AgentLoop(llm=mock, context=ctx, session_id=hist_sid(), max_iterations=3, verbose=False)
            ans = loop.run("loop", stream=False)
            # harus berhenti di max iter, dan ada tool calls 3 kali
            self.assertEqual(len(mock.calls), 3)
        finally:
            hist_mod.HISTS_DIR = orig

class TestPermissions(unittest.TestCase):
    def test_modes(self):
        from agent_core.permissions import PermissionManager
        pm = PermissionManager("ask")
        self.assertFalse(pm.is_auto_allowed("read"))
        self.assertFalse(pm.is_auto_allowed("bash"))
        self.assertTrue(pm.is_auto_allowed("skill_list"))
        self.assertTrue(pm.is_auto_allowed("skill_load"))
        pm.set_mode("accept-all")
        self.assertTrue(pm.is_auto_allowed("read"))
        self.assertTrue(pm.is_auto_allowed("bash"))
        pm.set_mode("accept-fs")
        self.assertTrue(pm.is_auto_allowed("read"))
        self.assertTrue(pm.is_auto_allowed("write"))
        self.assertTrue(pm.is_auto_allowed("glob"))
        self.assertFalse(pm.is_auto_allowed("bash"))

    def test_permission_gate_loop(self):
        # Test loop menghormati deny
        from agent_core.permissions import PermissionManager
        pm = PermissionManager("ask")
        # monkey patch prompt to deny
        pm.prompt = lambda n, a: False
        ctx = ContextManager(system_prompt="sys")
        f = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
        f.write("secret")
        f.close()
        pathlib.Path(f.name).write_text("secret123", encoding="utf-8")
        mock = MockLLM([
            {"content": "", "tool_calls": [{"id":"c1","type":"function","function":{"name":"read","arguments": json.dumps({"filePath": f.name})}}], "finish_reason":"tool_calls"},
            {"content": "denied handled", "tool_calls": None, "finish_reason":"stop"}
        ])
        import agent_core.history as hist_mod
        orig = hist_mod.HISTS_DIR
        tmp = tempfile.TemporaryDirectory()
        hist_mod.HISTS_DIR = tmp.name
        try:
            loop = AgentLoop(llm=mock, context=ctx, session_id=hist_sid(), verbose=False, permission_manager=pm)
            ans = loop.run("baca", stream=False)
            # tool harus DENIED, loop tetap lanjut dan final answer ada
            self.assertIn("denied handled", ans)
            # cek ada tool result DENIED di context
            tool_msgs = [m for m in ctx.messages if m.get("role")=="tool"]
            self.assertTrue(any("DENIED" in m.get("content","") for m in tool_msgs))
        finally:
            hist_mod.HISTS_DIR = orig
            tmp.cleanup()
            os.unlink(f.name)

    def test_accept_fs_loop(self):
        from agent_core.permissions import PermissionManager
        pm = PermissionManager("accept-fs")
        ctx = ContextManager(system_prompt="sys")
        tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(tmp.name)
        f = base / "a.txt"
        f.write_text("hi", encoding="utf-8")
        mock = MockLLM([
            {"content": "", "tool_calls": [{"id":"c1","type":"function","function":{"name":"read","arguments": json.dumps({"filePath": str(f)})}}], "finish_reason":"tool_calls"},
            {"content": "ok", "tool_calls": None, "finish_reason":"stop"}
        ])
        import agent_core.history as hist_mod
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = tmp.name
        try:
            loop = AgentLoop(llm=mock, context=ctx, session_id=hist_sid(), verbose=False, permission_manager=pm)
            ans = loop.run("read", stream=False)
            self.assertIn("ok", ans)
        finally:
            hist_mod.HISTS_DIR = orig
            tmp.cleanup()

class TestCommands(unittest.TestCase):
    def test_compact_command(self):
        from agent_core.context import ContextManager
        ctx = ContextManager(system_prompt="sys", max_tokens=500, keep_recent=2)
        for i in range(10):
            ctx.add_user("x"*200)
            ctx.add_assistant("y"*200)
        before = ctx.token_usage()["tokens"]
        from agent import handle_compact
        handle_compact(ctx)
        after = ctx.token_usage()["tokens"]
        self.assertLess(after, before)
        self.assertIn("COMPACTED", ctx.messages[1]["content"])

    def test_model_switch(self):
        from agent import handle_model
        from agent_core.llm import LLMClient
        llm = LLMClient(model="mimo-v2.5-free")
        handle_model("muse-spark-1.2-contributor-free", llm)
        self.assertEqual(llm.model, "muse-spark-1.2-contributor-free")

    def test_usage(self):
        from agent import handle_usage
        ctx = ContextManager(system_prompt="sys")
        ctx.add_user("hello")
        # should not raise
        handle_usage(ctx)

    def test_skill_handlers(self):
        from agent import handle_skill
        # buat temp skill dir
        tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(tmp.name) / ".agent" / "skills" / "demo"
        base.mkdir(parents=True)
        (base / "SKILL.md").write_text("# Demo\nskill demo", encoding="utf-8")
        import agent_core.skills as skmod
        orig = skmod.SKILLS_DIR
        skmod.SKILLS_DIR = str(pathlib.Path(tmp.name)/ ".agent/skills")
        try:
            handle_skill("")
            handle_skill("demo")
        finally:
            skmod.SKILLS_DIR = orig
            tmp.cleanup()

    def test_think(self):
        from agent import handle_think
        from agent_core.llm import LLMClient
        from agent_core.loop import AgentLoop
        from agent_core.context import ContextManager
        llm = LLMClient(model="mimo-v2.5-free")
        ctx = ContextManager(system_prompt="sys")
        loop = AgentLoop(llm=llm, context=ctx, session_id=hist_sid(), verbose=False)
        # non-support model should return current
        v = handle_think("", llm, loop, "none")
        self.assertEqual(v, "none")
        v2 = handle_think("medium", llm, loop, "none")
        # mimo tidak support, tetap none tapi warn
        # set ke muse-spark
        llm.model = "muse-spark-1.2-contributor-free"
        v3 = handle_think("medium", llm, loop, "none")
        self.assertEqual(v3, "medium")
        self.assertEqual(loop.extra_body.get("reasoning_effort"), "medium")
        v4 = handle_think("xhigh", llm, loop, v3)
        self.assertEqual(v4, "xhigh")
        self.assertEqual(loop.extra_body.get("reasoning_effort"), "xhigh")
        v5 = handle_think("none", llm, loop, v4)
        self.assertEqual(v5, "none")
        self.assertNotIn("reasoning_effort", loop.extra_body)

    def test_tool_call(self):
        from agent import handle_tool_call
        from agent_core.permissions import PermissionManager
        pm = PermissionManager("ask")
        handle_tool_call("accept-all", pm)
        self.assertEqual(pm.mode, "accept-all")
        handle_tool_call("accept-fs", pm)
        self.assertEqual(pm.mode, "accept-fs")
        handle_tool_call("ask", pm)
        self.assertEqual(pm.mode, "ask")

    def test_shell_escape(self):
        from agent import run_shell_direct
        # should not raise, run echo
        run_shell_direct("echo shelltest123")

    def test_reload(self):
        from agent import handle_reload
        from agent_core.context import ContextManager
        from agent_core.llm import LLMClient
        from agent_core.permissions import PermissionManager
        from agent_core.loop import AgentLoop
        ctx = ContextManager(system_prompt="sys")
        ctx.add_user("hi")
        llm = LLMClient(model="mimo-v2.5-free")
        pm = PermissionManager("ask")
        loop = AgentLoop(llm=llm, context=ctx, session_id=hist_sid(), verbose=False)
        handle_reload(ctx, llm, pm, "none")
        self.assertEqual(ctx.messages[1]["content"], "hi")


class TestProviders(unittest.TestCase):
    def test_openai_messages_to_anthropic(self):
        from agent_core.providers import openai_messages_to_anthropic
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"filePath":"a.txt"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read", "content": "file content"},
        ]
        system, anth_msgs = openai_messages_to_anthropic(msgs)
        self.assertEqual(system, "You are helpful")
        self.assertEqual(anth_msgs[0]["role"], "user")
        self.assertEqual(anth_msgs[0]["content"], "hello")
        # assistant should have tool_use block
        self.assertEqual(anth_msgs[1]["role"], "assistant")
        self.assertTrue(isinstance(anth_msgs[1]["content"], list))
        self.assertEqual(anth_msgs[1]["content"][1]["type"], "tool_use")
        self.assertEqual(anth_msgs[1]["content"][1]["name"], "read")
        # tool result should be user with tool_result
        self.assertEqual(anth_msgs[2]["role"], "user")
        self.assertEqual(anth_msgs[2]["content"][0]["type"], "tool_result")
        self.assertEqual(anth_msgs[2]["content"][0]["tool_use_id"], "call_1")

    def test_openai_tools_to_anthropic(self):
        from agent_core.providers import openai_tools_to_anthropic, anthropic_tools_to_openai
        tools = [{"type":"function","function":{"name":"read","description":"read file","parameters":{"type":"object","properties":{"filePath":{"type":"string"}}}}}]
        anth = openai_tools_to_anthropic(tools)
        self.assertEqual(anth[0]["name"], "read")
        self.assertIn("input_schema", anth[0])
        back = anthropic_tools_to_openai(anth)
        self.assertEqual(back[0]["function"]["name"], "read")

    def test_anthropic_response_to_openai(self):
        from agent_core.providers import anthropic_response_to_openai
        content = [{"type":"text","text":"hello "}, {"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"ls"}}]
        res = anthropic_response_to_openai(content, "tool_use")
        self.assertEqual(res["content"], "hello ")
        self.assertEqual(res["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(res["finish_reason"], "tool_calls")

    def test_prepare_payload_anthropic(self):
        from agent_core.providers import prepare_payload_for_provider, ProviderSpec
        # fallback hanya ke opencode, jadi base harus opencode.ai/zen/v1
        p = ProviderSpec("opencode_anthropic", "https://opencode.ai", "/zen/v1/messages", "anthropic")
        msgs = [{"role":"system","content":"sys"},{"role":"user","content":"hi"}]
        body = prepare_payload_for_provider(p, "claude-3", msgs, None, None, False)
        self.assertIn("system", body)
        self.assertEqual(body["system"], "sys")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertIn("max_tokens", body)

    def test_prepare_payload_openai(self):
        from agent_core.providers import prepare_payload_for_provider, ProviderSpec
        p = ProviderSpec("opencode_chat", "https://opencode.ai", "/zen/v1/chat/completions", "openai_chat")
        msgs = [{"role":"user","content":"hi"}]
        tools = [{"type":"function","function":{"name":"read","description":"read","parameters":{"type":"object","properties":{}}}}]
        body = prepare_payload_for_provider(p, "gpt-4", msgs, tools, {"temperature":0.5}, True)
        self.assertEqual(body["model"], "gpt-4")
        self.assertIn("tools", body)
        self.assertEqual(body["temperature"], 0.5)
        self.assertIn("messages", body)

    def test_prepare_payload_responses(self):
        from agent_core.providers import prepare_payload_for_provider, ProviderSpec
        # Sesuai SDK-example sec 2: input + instructions + store:false
        p = ProviderSpec("opencode_responses", "https://opencode.ai", "/zen/v1/responses", "openai_responses")
        msgs = [
            {"role":"system","content":"You are helpful"},
            {"role":"user","content":"Halo"},
            {"role":"assistant","content":"Halo! Ada yang bisa saya bantu?"},
            {"role":"user","content":"Jelaskan REST API."}
        ]
        body = prepare_payload_for_provider(p, "gpt-5.6-luna", msgs, None, None, False)
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertIn("input", body)
        self.assertIn("instructions", body)
        self.assertEqual(body["instructions"], "You are helpful")
        self.assertEqual(body["store"], False)
        self.assertEqual(body["input"][0]["role"], "user")
        self.assertEqual(body["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(body["input"][1]["content"][0]["type"], "output_text")

    def test_candidate_providers(self):
        from agent_core.providers import build_candidate_providers
        # muse-spark should prioritize responses (opencode)
        cands = build_candidate_providers("muse-spark-1.2-contributor-free")
        self.assertEqual(cands[0].sdk, "openai_responses")
        self.assertIn("opencode.ai/zen/v1", cands[0].url)
        # claude should prioritize anthropic via opencode
        cands2 = build_candidate_providers("claude-sonnet-4")
        self.assertEqual(cands2[0].sdk, "anthropic")
        self.assertIn("opencode.ai/zen/v1/messages", cands2[0].url)
        # gpt should prioritize openai_chat via opencode
        cands3 = build_candidate_providers("mimo-v2.5-free")
        self.assertEqual(cands3[0].sdk, "openai_chat")
        self.assertIn("/zen/v1/chat/completions", cands3[0].endpoint)
        # semua candidates harus hanya opencode.ai/zen/v1
        for cand in cands + cands2 + cands3:
            self.assertIn("opencode.ai/zen/v1", cand.url)

    def test_provider_state_persistence(self):
        from agent_core.providers import update_provider_state, get_saved_provider, load_provider_state
        import tempfile, pathlib, json, os
        import agent_core.providers as prov_mod
        orig = prov_mod.PROVIDER_STATE_FILE
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        prov_mod.PROVIDER_STATE_FILE = tmp.name
        try:
            update_provider_state("test-model-xyz", "anthropic", "/v1/messages", "https://api.anthropic.com")
            saved = get_saved_provider("test-model-xyz")
            self.assertEqual(saved["provider"], "anthropic")
            self.assertEqual(saved["endpoint"], "/v1/messages")
            # base_model_id fallback
            saved2 = get_saved_provider("test-model-xyz (free)")
            self.assertEqual(saved2["provider"], "anthropic")
            state = load_provider_state()
            self.assertIn("test-model-xyz", state)
        finally:
            prov_mod.PROVIDER_STATE_FILE = orig
            try:
                os.unlink(tmp.name)
            except: pass

    def test_llm_fallback_chain(self):
        from agent_core.llm import LLMClient
        from agent_core.providers import ProviderSpec
        import urllib.error, json, io
        # Mock urlopen to fail first provider (opencode_chat) then succeed second (opencode_responses) - hanya fallback ke opencode.ai/zen/v1
        orig_urlopen = urllib.request.urlopen
        call_count = {"n":0}
        def fake_urlopen(req, timeout=120):
            call_count["n"] += 1
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            # first call fail with 404
            if call_count["n"] == 1:
                # Pastikan fallback hanya ke opencode
                self.assertIn("opencode.ai/zen/v1", url)
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b'{"error":"not found"}'))
            # second succeed - bisa chat atau responses, keduanya di opencode
            self.assertIn("opencode.ai/zen/v1", url)
            fake_resp = io.BytesIO(json.dumps({"choices":[{"message":{"content":"fallback success","tool_calls":None},"finish_reason":"stop"}]}).encode())
            # mock context manager
            class FakeResp:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self, *a, **kw): return fake_resp.read(*a, **kw) if hasattr(fake_resp, 'read') else b''
            mock = FakeResp()
            mock.read = lambda *a, **kw: json.dumps({"choices":[{"message":{"content":"fallback success","tool_calls":None},"finish_reason":"stop"}]}).encode()
            mock.__enter__ = lambda s: s
            mock.__exit__ = lambda s,*a: False
            return mock
        import agent_core.providers as prov_mod
        orig_state = prov_mod.PROVIDER_STATE_FILE
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        prov_mod.PROVIDER_STATE_FILE = tmp.name
        urllib.request.urlopen = fake_urlopen
        try:
            client = LLMClient(model="mimo-v2.5-free", base_url="https://opencode.ai")
            res = client.chat([{"role":"user","content":"hi"}], stream=False)
            self.assertEqual(res["content"], "fallback success")
            self.assertGreaterEqual(call_count["n"], 2)
            # state should be saved to second provider (openai_responses via opencode)
            saved = prov_mod.get_saved_provider("mimo-v2.5-free")
            self.assertIsNotNone(saved)
            self.assertEqual(saved["provider"], "openai_responses")
            self.assertIn("opencode.ai", saved["base_url"])
            self.assertIn("/zen/v1", saved["endpoint"])
        finally:
            urllib.request.urlopen = orig_urlopen
            prov_mod.PROVIDER_STATE_FILE = orig_state
            try:
                os.unlink(tmp.name)
            except: pass
            # also cleanup state file that may have been created in default location?
            import pathlib as pl
            p = pl.Path(".agent/llm_provider_state.json")
            if p.exists() and p.stat().st_size < 2000:
                # keep minimal
                pass

    def test_history_stays_openai_format(self):
        from agent_core.history import generate_session_id, save_message, load_messages
        import agent_core.history as hist_mod
        import tempfile, pathlib
        tmp = tempfile.TemporaryDirectory()
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = tmp.name
        try:
            sid = generate_session_id()
            # simpan OpenAI format
            save_message(sid, {"role":"user","content":"hello"})
            save_message(sid, {"role":"assistant","content":"hi","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}]})
            save_message(sid, {"role":"tool","tool_call_id":"c1","name":"read","content":"data"})
            msgs = load_messages(sid)
            self.assertEqual(msgs[0]["role"], "user")
            self.assertEqual(msgs[1]["tool_calls"][0]["function"]["name"], "read")
            self.assertEqual(msgs[2]["role"], "tool")
            # Konversi saat request anthropic harus tidak merubah history
            from agent_core.providers import openai_messages_to_anthropic
            system, anth = openai_messages_to_anthropic(msgs)
            # history tetap OpenAI
            self.assertEqual(msgs[0]["content"], "hello")
            # anthropic hasil berbeda
            self.assertTrue(any(m["role"]=="user" for m in anth))
        finally:
            hist_mod.HISTS_DIR = orig
            tmp.cleanup()

    def test_anthropic_streaming_parse(self):
        from agent_core.llm import LLMClient
        import urllib.request, urllib.error, json, io
        # Buat fake anthropic SSE streaming: text + tool_use
        chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1"}}\n\n',
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
            b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"read","input":{}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"filePath\\" : \\"a.txt\\"}"}}\n\n',
            b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        data = b"".join(chunks)
        orig_urlopen = urllib.request.urlopen
        idx = {"i":0}
        class FakeResp:
            def __enter__(self): return self
            def __exit__(self,*a): return False
            def read(self, n=4096):
                if idx["i"] >= len(data):
                    return b""
                chunk = data[idx["i"]: idx["i"]+n]
                idx["i"] += len(chunk)
                return chunk
        def fake_urlopen(req, timeout=120):
            # Pastikan ini adalah anthropic provider
            self.assertIn("anthropic", req.full_url if hasattr(req, 'full_url') else "")
            return FakeResp()
        # patch
        self.orig_url = orig_urlopen
        import agent_core.providers as prov_mod
        orig_state = prov_mod.PROVIDER_STATE_FILE
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        prov_mod.PROVIDER_STATE_FILE = tmp.name
        urllib.request.urlopen = fake_urlopen
        try:
            # Buat client dengan model anthropic, paksa hanya satu candidate anthropic agar tidak fallback
            client = LLMClient(model="claude-sonnet-4")
            # Monkey patch build_candidate_providers untuk hanya return anthropic
            orig_build = prov_mod.build_candidate_providers
            def only_anthropic(m,b=None):
                return [prov_mod.ProviderSpec("anthropic","https://api.anthropic.com","/v1/messages","anthropic")]
            prov_mod.build_candidate_providers = only_anthropic
            # juga patch di llm module
            import agent_core.llm as llm_mod
            orig_build_llm = llm_mod.build_candidate_providers
            llm_mod.build_candidate_providers = only_anthropic
            res = client.chat([{"role":"user","content":"hi"}], stream=True)
            self.assertEqual(res["content"], "Hello world")
            self.assertIsNotNone(res["tool_calls"])
            self.assertEqual(res["tool_calls"][0]["function"]["name"], "read")
            self.assertEqual(res["finish_reason"], "tool_calls")
        finally:
            urllib.request.urlopen = orig_urlopen
            prov_mod.build_candidate_providers = orig_build
            llm_mod.build_candidate_providers = orig_build_llm
            prov_mod.PROVIDER_STATE_FILE = orig_state
            try: os.unlink(tmp.name)
            except: pass

    def test_openai_streaming_parse(self):
        from agent_core.llm import LLMClient
        import urllib.request, json, io
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello "},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\\"file"}}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Path\\":\\"a.txt\\"}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        data = b"".join(chunks)
        orig_urlopen = urllib.request.urlopen
        idx = {"i":0}
        class FakeResp:
            def __enter__(self): return self
            def __exit__(self,*a): return False
            def read(self, n=4096):
                if idx["i"] >= len(data):
                    return b""
                chunk = data[idx["i"]: idx["i"]+n]
                idx["i"] += len(chunk)
                return chunk
        def fake_urlopen(req, timeout=120):
            return FakeResp()
        import agent_core.providers as prov_mod
        orig_state = prov_mod.PROVIDER_STATE_FILE
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        prov_mod.PROVIDER_STATE_FILE = tmp.name
        urllib.request.urlopen = fake_urlopen
        try:
            client = LLMClient(model="mimo-v2.5-free")
            orig_build = prov_mod.build_candidate_providers
            def only_openai(m,b=None):
                return [prov_mod.ProviderSpec("opencode_chat","https://opencode.ai","/zen/v1/chat/completions","openai_chat")]
            prov_mod.build_candidate_providers = only_openai
            import agent_core.llm as llm_mod
            orig_build_llm = llm_mod.build_candidate_providers
            llm_mod.build_candidate_providers = only_openai
            res = client.chat([{"role":"user","content":"hi"}], stream=True)
            self.assertEqual(res["content"], "Hello world")
            self.assertIsNotNone(res["tool_calls"])
            self.assertEqual(res["tool_calls"][0]["function"]["name"], "read")
        finally:
            urllib.request.urlopen = orig_urlopen
            prov_mod.build_candidate_providers = orig_build
            llm_mod.build_candidate_providers = orig_build_llm
            prov_mod.PROVIDER_STATE_FILE = orig_state
            try: os.unlink(tmp.name)
            except: pass

    def test_429_retry_success(self):
        from agent_core.llm import LLMClient
        import urllib.request, urllib.error, json, io, time
        call_count = {"n":0}
        sleep_calls = []
        orig_sleep = time.sleep
        def fake_sleep(s):
            sleep_calls.append(s)
        time.sleep = fake_sleep
        orig_urlopen = urllib.request.urlopen
        def fake_urlopen(req, timeout=120):
            call_count["n"] += 1
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            if call_count["n"] <= 2:
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, io.BytesIO(b'{"error":"rate limit"}'))
            # success on 3rd
            class FakeResp:
                def __enter__(self): return self
                def __exit__(self,*a): return False
                def read(self, *a, **kw):
                    return json.dumps({"choices":[{"message":{"content":"retry success","tool_calls":None},"finish_reason":"stop"}]}).encode()
            return FakeResp()
        import agent_core.providers as prov_mod
        orig_state = prov_mod.PROVIDER_STATE_FILE
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        prov_mod.PROVIDER_STATE_FILE = tmp.name
        urllib.request.urlopen = fake_urlopen
        try:
            client = LLMClient(model="mimo-v2.5-free")
            # only one candidate to test retry without fallback
            orig_build = prov_mod.build_candidate_providers
            def only_one(m,b=None):
                return [prov_mod.ProviderSpec("opencode_chat","https://opencode.ai","/zen/v1/chat/completions","openai_chat")]
            prov_mod.build_candidate_providers = only_one
            import agent_core.llm as llm_mod
            orig_build_llm = llm_mod.build_candidate_providers
            llm_mod.build_candidate_providers = only_one
            res = client.chat([{"role":"user","content":"hi"}], stream=False)
            self.assertEqual(res["content"], "retry success")
            self.assertEqual(call_count["n"], 3)
            # cek wait: 5 untuk attempt1, 8 untuk attempt2
            self.assertEqual(sleep_calls, [5, 8])
            # tidak fallback, tetap provider sama
            saved = prov_mod.get_saved_provider("mimo-v2.5-free")
            self.assertEqual(saved["provider"], "openai_chat")
        finally:
            time.sleep = orig_sleep
            urllib.request.urlopen = orig_urlopen
            prov_mod.build_candidate_providers = orig_build
            llm_mod.build_candidate_providers = orig_build_llm
            prov_mod.PROVIDER_STATE_FILE = orig_state
            try: os.unlink(tmp.name)
            except: pass

    def test_429_fallback_after_5(self):
        from agent_core.llm import LLMClient
        import urllib.request, urllib.error, json, io, time
        call_count = {"n":0}
        sleep_calls = []
        orig_sleep = time.sleep
        time.sleep = lambda s: sleep_calls.append(s)
        orig_urlopen = urllib.request.urlopen
        def fake_urlopen(req, timeout=120):
            call_count["n"] += 1
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            # untuk provider pertama (opencode_chat), selalu 429 selama 5 kali
            # untuk provider kedua (opencode_responses), sukses
            if "chat/completions" in url:
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, io.BytesIO(b'{"error":"rate"}'))
            else:
                class FakeResp:
                    def __enter__(self): return self
                    def __exit__(self,*a): return False
                    def read(self, *a, **kw):
                        return json.dumps({"choices":[{"message":{"content":"fallback after 429","tool_calls":None},"finish_reason":"stop"}]}).encode()
                return FakeResp()
        import agent_core.providers as prov_mod
        orig_state = prov_mod.PROVIDER_STATE_FILE
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        prov_mod.PROVIDER_STATE_FILE = tmp.name
        urllib.request.urlopen = fake_urlopen
        try:
            client = LLMClient(model="mimo-v2.5-free")
            # candidates: chat then responses
            orig_build = prov_mod.build_candidate_providers
            def two_cands(m,b=None):
                return [
                    prov_mod.ProviderSpec("opencode_chat","https://opencode.ai","/zen/v1/chat/completions","openai_chat"),
                    prov_mod.ProviderSpec("opencode_responses","https://opencode.ai","/zen/v1/responses","openai_responses")
                ]
            prov_mod.build_candidate_providers = two_cands
            import agent_core.llm as llm_mod
            orig_build_llm = llm_mod.build_candidate_providers
            llm_mod.build_candidate_providers = two_cands
            res = client.chat([{"role":"user","content":"hi"}], stream=False)
            self.assertEqual(res["content"], "fallback after 429")
            # harus retry 5 kali untuk chat (4 sleeps: 5,8,11,14) lalu fallback ke responses
            self.assertEqual(sleep_calls, [5, 8, 11, 14])
            self.assertEqual(call_count["n"], 6)  # 5 gagal chat + 1 sukses responses
            saved = prov_mod.get_saved_provider("mimo-v2.5-free")
            self.assertEqual(saved["provider"], "openai_responses")
        finally:
            time.sleep = orig_sleep
            urllib.request.urlopen = orig_urlopen
            prov_mod.build_candidate_providers = orig_build
            llm_mod.build_candidate_providers = orig_build_llm
            prov_mod.PROVIDER_STATE_FILE = orig_state
            try: os.unlink(tmp.name)
            except: pass

    def test_history_model_and_think(self):
        from agent_core.history import generate_session_id, save_message, load_history, get_last_model_and_think
        import agent_core.history as hist_mod
        tmp = tempfile.TemporaryDirectory()
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = tmp.name
        try:
            sid = generate_session_id()
            save_message(sid, {"role":"user","content":"hi"})
            save_message(sid, {"role":"assistant","content":"hello","model":"mimo-v2.5-free","think_variant":"medium"})
            save_message(sid, {"role":"assistant","content":"hi2","model":"muse-spark-1.2-contributor-free","think_variant":"xhigh"})
            last_model, last_think = get_last_model_and_think(sid)
            self.assertEqual(last_model, "muse-spark-1.2-contributor-free")
            self.assertEqual(last_think, "xhigh")
            # cek load_messages tetap filter model
            from agent_core.history import load_messages
            msgs = load_messages(sid)
            self.assertEqual(len(msgs), 3)
            # raw history harus ada model
            raw = load_history(sid)
            self.assertEqual(raw[1].get("model"), "mimo-v2.5-free")
            self.assertEqual(raw[1].get("think_variant"), "medium")
        finally:
            hist_mod.HISTS_DIR = orig
            tmp.cleanup()

    def test_loop_saves_model_think(self):
        from agent_core.history import generate_session_id, load_history, get_last_model_and_think
        import agent_core.history as hist_mod
        tmp = tempfile.TemporaryDirectory()
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = tmp.name
        try:
            sid = generate_session_id()
            ctx = ContextManager(system_prompt="sys")
            mock = MockLLM([{"content":"hello","tool_calls":None,"finish_reason":"stop"}], model="muse-spark-1.2-contributor-free")
            loop = AgentLoop(llm=mock, context=ctx, session_id=sid, verbose=False, extra_body={"reasoning_effort":"high"})
            loop.run("hi", stream=False)
            raw = load_history(sid)
            # cari assistant entry
            assistant_entries = [r for r in raw if r.get("role")=="assistant"]
            self.assertTrue(len(assistant_entries) > 0)
            self.assertEqual(assistant_entries[-1].get("model"), "muse-spark-1.2-contributor-free")
            self.assertEqual(assistant_entries[-1].get("think_variant"), "high")
            last_model, last_think = get_last_model_and_think(sid)
            self.assertEqual(last_model, "muse-spark-1.2-contributor-free")
            self.assertEqual(last_think, "high")
        finally:
            hist_mod.HISTS_DIR = orig
            tmp.cleanup()

    def test_agent_loads_last_model(self):
        from agent_core.history import generate_session_id, save_message, get_last_model_and_think
        import agent_core.history as hist_mod
        import agent_core.llm as llm_mod
        tmp = tempfile.TemporaryDirectory()
        orig = hist_mod.HISTS_DIR
        hist_mod.HISTS_DIR = tmp.name
        try:
            sid = generate_session_id()
            save_message(sid, {"role":"user","content":"hi"})
            save_message(sid, {"role":"assistant","content":"hello","model":"muse-spark-1.2-contributor-free","think_variant":"xhigh"})
            last_model, last_think = get_last_model_and_think(sid)
            self.assertEqual(last_model, "muse-spark-1.2-contributor-free")
            self.assertEqual(last_think, "xhigh")
            # simulasi agent load: llm default mimo, tapi harus override ke history
            llm = llm_mod.LLMClient(model="mimo-v2.5-free", session_id=sid)
            think_variant = "none"
            # mimic agent.py logic
            lm, lt = get_last_model_and_think(sid)
            if lm:
                llm.model = lm
            if lt:
                think_variant = lt
            self.assertEqual(llm.model, "muse-spark-1.2-contributor-free")
            self.assertEqual(think_variant, "xhigh")
        finally:
            hist_mod.HISTS_DIR = orig
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
