"""
Tests for voicekit/tools.py.

    python tests/test_tools.py
"""

import os
import sys
import unittest

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from typing import Literal

from voicekit.llm_client import ToolCall
from voicekit.tools import Tool, ToolSet, classify_confirmation, run_tool_calls, tool


def make_tools():
    return ToolSet([
        Tool(
            name="get_status",
            description="Report settings.",
            run=lambda: "All good.",
        ),
        Tool(
            name="set_thinking",
            description="Turn thinking on or off.",
            parameters={"enabled": {"type": "boolean"}},
            required=("enabled",),
            run=lambda enabled: f"Thinking {'on' if enabled else 'off'}.",
            confirm=lambda enabled: f"Turn thinking {'on' if enabled else 'off'}?",
        ),
        Tool(
            name="explode",
            description="Always fails.",
            run=lambda: 1 / 0,
            confirm=lambda: "Explode?",
        ),
    ])


class ToolSetTests(unittest.TestCase):

    def setUp(self):
        self.tools = make_tools()

    def test_schema_format(self):
        schema = self.tools.tools["set_thinking"].schema()

        self.assertEqual(schema, {
            "type": "function",
            "function": {
                "name": "set_thinking",
                "description": "Turn thinking on or off.",
                "parameters": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"],
                },
            },
        })
        self.assertEqual(len(self.tools.schemas()), 3)

    def test_run_passes_arguments(self):
        results = self.tools.run([
            ToolCall("set_thinking", {"enabled": True}),
            ToolCall("get_status", {}),
        ])

        self.assertEqual(results, ["Thinking on.", "All good."])

    def test_run_reports_errors_instead_of_raising(self):
        with self.assertLogs("voicekit.tools", "WARNING"):
            results = self.tools.run([
                ToolCall("nope", {}),
                ToolCall("set_thinking", {"wrong": 1}),
                ToolCall("explode", {}),
            ])

        self.assertIn("no tool called nope", results[0])
        self.assertIn("invalid arguments", results[1])
        self.assertIn("explode failed", results[2])

    def test_needs_confirmation(self):
        status = [ToolCall("get_status", {})]
        change = [ToolCall("set_thinking", {"enabled": True})]

        self.assertFalse(self.tools.needs_confirmation(status))
        self.assertTrue(self.tools.needs_confirmation(change))
        self.assertTrue(self.tools.needs_confirmation(status + change))
        self.assertTrue(self.tools.needs_confirmation(status, always=True))
        self.assertTrue(self.tools.needs_confirmation([ToolCall("unknown", {})]))

    def test_confirmation_question(self):
        question = self.tools.confirmation_question([
            ToolCall("set_thinking", {"enabled": False}),
            ToolCall("get_status", {}),
        ])

        self.assertEqual(question, "Turn thinking off? Run get status?")


class DecoratorTests(unittest.TestCase):

    def test_schema_from_type_hints_and_docstring(self):
        import enum
        from typing import Literal, Optional

        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        @tool
        def paint(room: str, coats: int, color: Color,
                  finish: Literal["matte", "gloss"] = "matte",
                  budget: Optional[float] = None, tags: list[str] = ()) -> str:
            """Paint a room.

            Args:
                room: Which room to paint.
                coats: How many coats.
            """
            return "ok"

        self.assertEqual(paint.name, "paint")
        self.assertEqual(paint.description, "Paint a room.")
        self.assertEqual(paint.required, ("room", "coats", "color"))
        self.assertEqual(paint.parameters, {
            "room": {"type": "string", "description": "Which room to paint."},
            "coats": {"type": "integer", "description": "How many coats."},
            "color": {"type": "string", "enum": ["red", "blue"]},
            "finish": {"enum": ["matte", "gloss"], "type": "string"},
            "budget": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
        })

    def test_options(self):
        @tool(name="lights", description="Switch lights.", return_direct=True,
              confirm=lambda on: "Sure?")
        def set_lights(on: bool):
            return "done"

        self.assertEqual((set_lights.name, set_lights.description), ("lights", "Switch lights."))
        self.assertTrue(set_lights.return_direct)
        self.assertEqual(set_lights.confirm(on=True), "Sure?")

    def test_ctx_parameter_is_injected_not_exposed(self):
        @tool
        def greet(ctx, name: str) -> str:
            """Greet someone."""
            ctx.say(f"Hello {name}")
            return "greeted"

        self.assertNotIn("ctx", greet.parameters)
        self.assertTrue(greet.wants_context)

        context = SimpleNamespace(said=[], say=lambda text: context.said.append(text))
        result = ToolSet([greet]).run([ToolCall("greet", {"name": "Sam"})], ctx=context)

        self.assertEqual(result, ["greeted"])
        self.assertEqual(context.said, ["Hello Sam"])

    def test_plain_functions_can_be_added(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tools = ToolSet()
        tools.add(add)

        self.assertEqual(tools.run([ToolCall("add", {"a": "2", "b": 3})]), ["5"])

    def test_provider_schemas(self):
        @tool
        def ping(host: str) -> str:
            """Ping a host."""
            return "pong"

        self.assertEqual(ping.schema()["function"]["parameters"]["required"], ["host"])
        anthropic = ping.anthropic_schema()
        self.assertEqual(anthropic["input_schema"]["properties"], {"host": {"type": "string"}})
        self.assertTrue(anthropic["eager_input_streaming"])


class ValidationTests(unittest.TestCase):

    def setUp(self):
        @tool
        def configure(name: str, retries: int, enabled: bool,
                      mode: Literal["fast", "safe"] = "safe", ratio: float = 1.0):
            """Configure."""
            return "ok"

        self.tool = configure

    def test_coerces_strings_from_small_models(self):
        arguments, error = self.tool.validate(
            {"name": "x", "retries": "3", "enabled": "true", "ratio": "0.5"}
        )

        self.assertIsNone(error)
        self.assertEqual(arguments, {"name": "x", "retries": 3, "enabled": True, "ratio": 0.5})

    def test_errors(self):
        cases = [
            ({"retries": 1, "enabled": True}, "missing required argument 'name'"),
            ({"name": "x", "retries": "many", "enabled": True}, "'retries' must be of type integer"),
            ({"name": "x", "retries": 1, "enabled": "maybe"}, "'enabled' must be of type boolean"),
            ({"name": "x", "retries": 1, "enabled": True, "mode": "turbo"}, "'mode' must be one of"),
            ("not a dict", "must be an object"),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                _, error = self.tool.validate(arguments)
                self.assertIn(expected, error)


class RunToolCallsTests(unittest.TestCase):

    def setUp(self):
        self.tools = make_tools()
        self.tools.add(Tool(
            name="announce", description="x", run=lambda: "Switching now.",
            return_direct=True,
        ))
        self.ctx = SimpleNamespace(events=[], answers=[])
        self.ctx.emit = self.ctx.events.append
        self.ctx.confirm = lambda question: self.ctx.answers.pop(0)

    def test_events_and_results(self):
        from voicekit.agent import ToolFinished, ToolStarted

        outcome = run_tool_calls(self.tools, [ToolCall("get_status", {}, "c1")], self.ctx)

        self.assertEqual(outcome.results, ["All good."])
        self.assertEqual(outcome.errors, [False])
        self.assertFalse(outcome.ends_turn)
        self.assertEqual([type(e) for e in self.ctx.events], [ToolStarted, ToolFinished])

    def test_direct_results(self):
        outcome = run_tool_calls(self.tools, [ToolCall("announce", {})], self.ctx)

        self.assertTrue(outcome.ends_turn)
        self.assertEqual(outcome.direct, ["Switching now."])

    def test_errors_flagged(self):
        with self.assertLogs("voicekit.tools", "ERROR"):
            outcome = run_tool_calls(self.tools, [ToolCall("explode", {})], self.ctx)

        self.assertEqual(outcome.errors, [True])
        self.assertFalse(outcome.ends_turn)

    def test_confirm_declined(self):
        self.ctx.answers = [False]

        outcome = run_tool_calls(
            self.tools, [ToolCall("set_thinking", {"enabled": True})], self.ctx, confirm=True,
        )

        self.assertEqual(outcome.results, ["The user declined; it was not run."])

    def test_confirm_accepted(self):
        self.ctx.answers = [True]

        outcome = run_tool_calls(
            self.tools, [ToolCall("set_thinking", {"enabled": True})], self.ctx, confirm=True,
        )

        self.assertEqual(outcome.results, ["Thinking on."])

    def test_without_context(self):
        self.assertEqual(
            run_tool_calls(self.tools, [ToolCall("get_status", {})]).results, ["All good."]
        )


class ConfirmationTests(unittest.TestCase):

    def test_yes(self):
        for said in ("Yes.", "Yeah, go ahead.", "Sure!", "OK", "Okay do it",
                     "Yes please", "Go ahead.", "Do it.", "That's right.",
                     "Absolutely", "Alright"):
            with self.subTest(said=said):
                self.assertIs(classify_confirmation(said), True)

    def test_no(self):
        for said in ("No.", "Nope", "No thanks.", "Cancel that.", "Don't.",
                     "Never mind.", "Nevermind", "Stop!", "Not now.",
                     "Forget it."):
            with self.subTest(said=said):
                self.assertIs(classify_confirmation(said), False)

    def test_other(self):
        for said in ("What's the weather like?", "Tell me a joke.",
                     "Yesterday I went to the shop",
                     "Okay so here is a much longer sentence about something",
                     "Nobody knows", ""):
            with self.subTest(said=said):
                self.assertIsNone(classify_confirmation(said))


if __name__ == "__main__":
    unittest.main()
