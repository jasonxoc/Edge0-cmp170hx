"""serve/test_tool_special_func.py - `<|im_start|>function=NAME>` of a declared tool is `<function=NAME>`.

Seen live (Edge0 145B NVFP4 on a CMP 170HX, pi agent, ~36k prompt): after `<tool_call>` the model wrote the template's turn
token where the `<` of the call tag belongs.  The block came back as answer text and the agent stopped.

    python -m unittest serve.test_tool_special_func -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

BASH = {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}}}
CMD = "cd /Users/jason/src/vmg/portal/readermgmt && grep -n 'function ' controllers/streetteam_controller.php"
LIVE = ("<tool_call>\n<|im_start|>function=bash>\n<parameter=command>\n" + CMD + "\n</parameter>\n</function>\n</tool_call>")


def run(text, stream_tools=False, recover=True, thinking=False, step=3, tools=(BASH,)):
    p = OutputParser(thinking=thinking, tools=list(tools), stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind in ("content", "reasoning"))


class SpecialFunc(unittest.TestCase):
    def test_live_block(self):
        for stream in (False, True):
            for step in (1, 2, 3, 5, 12, 1000):
                for recover in (False, True):
                    evs = run(LIVE, stream, recover, step=step)
                    self.assertEqual(calls(evs), [("bash", {"command": CMD})], (stream, step, recover))
                    self.assertEqual(text(evs).strip(), "", (stream, step, recover))

    def test_token_as_one_delta(self):
        p = OutputParser(thinking=False, tools=[BASH], stream_tools=False, recover=True)
        evs = []
        for d in ("<tool_call>", "\n", "<|im_start|>", "function", "=", "bash", ">\n", "<parameter=command>\nls\n",
                  "</parameter>\n</function>\n</tool_call>"):
            evs += p.feed(d)
        evs += p.finish("stop")
        self.assertEqual(calls(evs), [("bash", {"command": "ls"})])

    def test_in_the_reasoning(self):
        evs = run("hm\n" + LIVE + "\n</think>\n", thinking=True)
        self.assertEqual(calls(evs), [("bash", {"command": CMD})])

    def test_undeclared_name_stays_text(self):
        t = LIVE.replace("function=bash>", "function=nothing>")
        for step in (1, 3, 1000):
            self.assertEqual(calls(run(t, step=step)), [])
            self.assertIn("<|im_start|>function=nothing>", text(run(t, step=step)))

    def test_other_text_kept(self):
        for t in ("a <|im_start|> b", "ends with <|im_start|>", "ends with <|im_", "x <|im_start|>funky", "a <| b", "<|"):
            for step in (1, 3, 1000):
                self.assertEqual(text(run(t, step=step)), t, (t, step))

    def test_no_tools_untouched(self):
        self.assertEqual(text(run(LIVE, tools=())), LIVE)


if __name__ == "__main__":
    unittest.main()
