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
    def __init__(self, responses):
        # responses: list of dicts {content, tool_calls}
        self.responses = list(responses)
        self.calls = []
        self.extra_bodies = []
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
        v4 = handle_think("none", llm, loop, v3)
        self.assertEqual(v4, "none")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
