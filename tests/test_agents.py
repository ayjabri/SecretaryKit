"""
Tests for the agent adapters in voicekit/agents/ and FunctionAgent.

    python tests/test_agents.py              # all fakes (LangGraph/LangChain are real libraries)
    python tests/test_agents.py --server     # + OllamaAgent and LangGraph on your Ollama server
    python tests/test_agents.py --claude     # + ClaudeAgent against the Claude API (costs money)
"""

import json
import os
import sys
import unittest
import warnings
from types import SimpleNamespace

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _take_flag(flag):
    # Before the imports below: tests.fakes imports another test module
    # that reads its own flags.
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


RUN_SERVER_TESTS = _take_flag("--server") or os.environ.get("RUN_SERVER_TESTS") == "1"
RUN_CLAUDE_TESTS = _take_flag("--claude") or os.environ.get("RUN_CLAUDE_TESTS") == "1"

# create_react_agent (used here as a sample graph) is deprecated in
# LangGraph 1.x in favour of langchain.agents.create_agent.
warnings.filterwarnings("ignore")

import httpx

from voicekit import (
    AgentError,
    FunctionAgent,
    LLMClient,
    LLMConfig,
    Status,
    Thinking,
    ToolFinished,
    ToolStarted,
    tool,
)
from voicekit.agents import (
    ClaudeAgent,
    LangChainAgent,
    LangGraphAgent,
    OllamaAgent,
    to_langchain_tools,
)
from voicekit.commands import NOT_HANDLED, CommandRouter
from voicekit.session import control_tools

from tests.fakes import (
    OllamaServer,
    RecordingContext,
    collect_text,
    ollama_lines,
    ollama_tool_call_lines,
)


# --------------------------------------------------
# FunctionAgent
# --------------------------------------------------

class FunctionAgentTests(unittest.TestCase):

    def test_string_result(self):
        agent = FunctionAgent(lambda text, ctx: f"Echo {text}")
        self.assertEqual(collect_text(agent.respond("hi", RecordingContext())), "Echo hi")

    def test_one_argument_function(self):
        agent = FunctionAgent(lambda text: text.upper())
        self.assertEqual(collect_text(agent.respond("hi", RecordingContext())), "HI")

    def test_generator_with_events(self):
        def handle(text, ctx):
            yield Status("working")
            yield "done"

        events = list(FunctionAgent(handle).respond("x", RecordingContext()))

        self.assertIsInstance(events[0], Status)
        self.assertEqual(events[1], "done")

    def test_reset_and_describe(self):
        resets = []
        agent = FunctionAgent(lambda t: t, reset=lambda: resets.append(1), describe="Echo.")

        agent.reset()

        self.assertEqual(resets, [1])
        self.assertEqual(agent.describe(), "Echo.")


# --------------------------------------------------
# OllamaAgent (real LLMClient, fake HTTP server)
# --------------------------------------------------

def make_ollama(server, **kwargs):
    client = LLMClient(LLMConfig(system_prompt="Be brief."), transport=httpx.MockTransport(server))
    agent = OllamaAgent(client, **kwargs)
    agent.start()
    return agent


class OllamaAgentTests(unittest.TestCase):

    def test_plain_reply(self):
        agent = make_ollama(OllamaServer([ollama_lines("Tokyo", ".")]))

        self.assertEqual(collect_text(agent.respond("Capital of Japan?", RecordingContext())),
                         "Tokyo.")

    def test_start_adds_tools_prompt_once(self):
        agent = make_ollama(OllamaServer())
        agent.start()

        prompt = agent.llm.config.system_prompt
        self.assertTrue(prompt.startswith("Be brief."))
        self.assertEqual(prompt.count("provided tools"), 1)

    def test_offers_tools_with_installed_models_enum(self):
        server = OllamaServer([ollama_lines("Hi.")])
        agent = make_ollama(server)

        collect_text(agent.respond("Hi", RecordingContext()))

        tools = {t["function"]["name"]: t for t in server.chat_bodies()[0]["tools"]}
        self.assertIn("set_thinking", tools)
        enum = tools["switch_model"]["function"]["parameters"]["properties"]["model"]["enum"]
        self.assertEqual(enum, ["qwen3.5:4b", "qwen3.6:27b", "llama3.1:8b"])

    def test_information_tool_result_goes_back_to_model(self):
        server = OllamaServer([
            ollama_tool_call_lines(("list_models", {})),
            ollama_lines("You have two Qwens."),
        ])
        agent = make_ollama(server)
        ctx = RecordingContext()

        text = collect_text(agent.respond("What models are there?", ctx))

        self.assertEqual(text, "You have two Qwens.")
        self.assertEqual([e.name for e in ctx.of_type(ToolStarted)], ["list_models"])
        self.assertIn("Qwen 3.6 27B", ctx.of_type(ToolFinished)[0].result)

    def test_direct_tool_is_announced_and_ends_turn(self):
        server = OllamaServer([ollama_tool_call_lines(("switch_model", {"model": "qwen3.6:27b"}))])
        agent = make_ollama(server)

        text = collect_text(agent.respond("Use the bigger model", RecordingContext()))

        self.assertEqual(text, "Switching to Qwen 3.6 27B.")
        self.assertEqual(agent.llm.config.model, "qwen3.6:27b")
        self.assertEqual(len(server.chat_bodies()), 1)  # no second model reply
        self.assertEqual(agent.llm.history[-1]["role"], "tool")

    def test_confirm_tools_asks_first(self):
        server = OllamaServer([
            ollama_tool_call_lines(("set_thinking", {"enabled": True})),
            ollama_lines("Okay, I'll leave it off."),
        ])
        agent = make_ollama(server, confirm_tools=True)
        ctx = RecordingContext(answers=["no"])

        text = collect_text(agent.respond("Think harder", ctx))

        self.assertEqual(ctx.asked, ["Turn thinking on?"])
        self.assertFalse(agent.llm.config.think)
        # The model is told the user declined, and replies itself.
        self.assertIn("declined", server.chat_bodies()[1]["messages"][-1]["content"])
        self.assertEqual(text, "Okay, I'll leave it off.")

    def test_custom_tool(self):
        @tool
        def lights(room: str, on: bool) -> str:
            """Switch the lights in a room."""
            return f"{room} lights {'on' if on else 'off'}."

        server = OllamaServer([
            ollama_tool_call_lines(("lights", {"room": "kitchen", "on": "true"})),
            ollama_lines("Done."),
        ])
        agent = make_ollama(server, tools=[lights], settings_tools=False)
        ctx = RecordingContext()

        collect_text(agent.respond("Kitchen lights on", ctx))

        self.assertEqual(ctx.of_type(ToolFinished)[0].result, "kitchen lights on.")
        self.assertEqual([t["function"]["name"] for t in server.chat_bodies()[0]["tools"]],
                         ["lights"])

    def test_no_tools_for_models_without_support(self):
        server = OllamaServer([ollama_lines("Hi.")])
        agent = make_ollama(server)
        agent.llm.config.model = "llama3.1:8b"

        collect_text(agent.respond("Hi", RecordingContext()))

        self.assertNotIn("tools", server.chat_bodies()[0])

    def test_thinking_emitted(self):
        lines = "\n".join(json.dumps(x) for x in [
            {"message": {"content": "", "thinking": "Let me see."}},
            {"message": {"content": "391."}},
            {"message": {"content": ""}, "done": True},
        ]) + "\n"
        agent = make_ollama(OllamaServer([lines]))
        ctx = RecordingContext()

        self.assertEqual(collect_text(agent.respond("17*23", ctx)), "391.")
        self.assertEqual([e.text for e in ctx.of_type(Thinking)], ["Let me see."])

    def test_cancel_stops_reply(self):
        agent = make_ollama(OllamaServer([ollama_lines("one ", "two ", "three")]))
        ctx = RecordingContext()
        stream = agent.respond("count", ctx)

        first = next(stream)
        ctx.cancel()
        rest = list(stream)

        self.assertEqual((first.text, rest), ("one ", []))

    def test_voice_commands(self):
        agent = make_ollama(OllamaServer())
        router = CommandRouter()
        agent.register_commands(router)

        self.assertEqual(router.dispatch("Set think."),
                         "Turning thinking on. Replies will take longer.")
        self.assertEqual(router.dispatch("Thinking off"), "Turning thinking off.")
        self.assertEqual(router.dispatch("Use Quinn three point six"), "Switching to Qwen 3.6 27B.")
        self.assertIs(router.dispatch("Use Python to sort a list"), NOT_HANDLED)
        self.assertIn("couldn't find", router.dispatch("Switch model to Mistral"))
        self.assertIn("Available models are", router.dispatch("List models."))

    def test_thinking_refused_for_model_without_it(self):
        agent = make_ollama(OllamaServer())
        agent.llm.config.model = "llama3.1:8b"

        self.assertEqual(agent.set_thinking(True), "Llama 3.1 8B doesn't support thinking.")

    def test_describe_and_vocabulary(self):
        agent = make_ollama(OllamaServer())

        self.assertEqual(agent.describe(), "I'm using Qwen 3.5 4B, and thinking is off.")
        self.assertEqual(agent.vocabulary(), ["Llama", "Qwen"])

    def test_session_control_tools_get_announced(self):
        server = OllamaServer([ollama_tool_call_lines(("set_speaking_speed", {"change": "faster"}))])
        agent = make_ollama(server, settings_tools=False)
        agent.tools.add(*control_tools())

        text = collect_text(agent.respond("Talk quicker", RecordingContext()))

        self.assertEqual(text, "speed 1")


# --------------------------------------------------
# ClaudeAgent (fake Anthropic client)
# --------------------------------------------------

def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def claude_reply(text="", thinking="", tool_uses=(), stop_reason=None):
    """One scripted Claude response: deltas to stream + the final message."""

    content = []
    if thinking:
        content.append(block("thinking", thinking=thinking))
    if text:
        content.append(block("text", text=text))
    for tool_id, name, arguments in tool_uses:
        content.append(block("tool_use", id=tool_id, name=name, input=arguments))

    events = []
    if thinking:
        events.append(block("content_block_delta", delta=block("thinking_delta", thinking=thinking)))
    for word in text.split(" ") if text else []:
        events.append(block("content_block_delta", delta=block("text_delta", text=word + " ")))

    return SimpleNamespace(
        events=events,
        message=SimpleNamespace(
            content=content,
            stop_reason=stop_reason or ("tool_use" if tool_uses else "end_turn"),
        ),
    )


class FakeClaudeStream:

    def __init__(self, reply):
        self.reply = reply

    def __enter__(self):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        return iter(self.reply.events)

    def get_final_message(self):
        return self.reply.message


class FakeAnthropic:

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.models = SimpleNamespace(retrieve=lambda model: SimpleNamespace(id=model))

    def _stream(self, **params):
        # Copy the history: the agent keeps appending to the same list.
        self.requests.append({**params, "messages": list(params["messages"])})
        return FakeClaudeStream(self.replies.pop(0))


def make_claude(replies, **kwargs):
    client = FakeAnthropic(replies)
    agent = ClaudeAgent(client=client, **kwargs)
    agent.start()
    return agent, client


class ClaudeAgentTests(unittest.TestCase):

    def test_streams_text_and_thinking(self):
        agent, _ = make_claude([claude_reply("Tokyo is the capital.", thinking="Easy one.")])
        ctx = RecordingContext()

        text = collect_text(agent.respond("Capital of Japan?", ctx))

        self.assertEqual(text.strip(), "Tokyo is the capital.")
        self.assertEqual([e.text for e in ctx.of_type(Thinking)], ["Easy one."])

    def test_request_parameters(self):
        @tool
        def lights(on: bool) -> str:
            """Switch the lights."""
            return "ok"

        agent, client = make_claude([claude_reply("Hi.")], tools=[lights])

        collect_text(agent.respond("Hi", RecordingContext()))

        params = client.requests[0]
        self.assertEqual(params["model"], "claude-opus-5-5")
        self.assertEqual(params["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(params["output_config"], {"effort": "low"})
        self.assertEqual(params["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(params["fallbacks"], "default")
        self.assertEqual(params["tools"][0]["name"], "lights")
        self.assertTrue(params["tools"][0]["eager_input_streaming"])
        self.assertEqual(params["tools"][0]["input_schema"]["required"], ["on"])
        self.assertIn("provided tools", params["system"])

    def test_tool_loop(self):
        @tool
        def weather(city: str) -> str:
            """Current weather for a city."""
            return f"Sunny in {city}."

        agent, client = make_claude([
            claude_reply(tool_uses=[("tu_1", "weather", {"city": "Paris"})]),
            claude_reply("It's sunny in Paris."),
        ], tools=[weather])
        ctx = RecordingContext()

        text = collect_text(agent.respond("Weather in Paris?", ctx))

        self.assertEqual(text.strip(), "It's sunny in Paris.")
        tool_turn = client.requests[1]["messages"][-1]
        self.assertEqual(tool_turn["content"], [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "Sunny in Paris."}
        ])
        self.assertEqual([m["role"] for m in agent.history],
                         ["user", "assistant", "user", "assistant"])

    def test_invalid_tool_input_reported_as_error(self):
        @tool
        def weather(city: str) -> str:
            """Weather."""
            return "Sunny."

        agent, client = make_claude([
            claude_reply(tool_uses=[("tu_1", "weather", {})]),
            claude_reply("Which city?"),
        ], tools=[weather])

        collect_text(agent.respond("Weather?", RecordingContext()))

        result = client.requests[1]["messages"][-1]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("missing required argument", result["content"])

    def test_direct_tool_ends_turn(self):
        agent, client = make_claude(
            [claude_reply(tool_uses=[("tu_1", "set_speaking_speed", {"change": "slower"})])],
            tools=control_tools(),
        )

        text = collect_text(agent.respond("Slow down", RecordingContext()))

        self.assertEqual(text, "speed -1")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(agent.history[-1]["content"][0]["type"], "tool_result")

    def test_refusal(self):
        agent, _ = make_claude([claude_reply(stop_reason="refusal")])

        text = collect_text(agent.respond("something", RecordingContext()))

        self.assertIn("can't help", text)

    def test_api_error_becomes_agent_error_and_forgets_question(self):
        import anthropic
        import httpx2

        error = anthropic.InternalServerError(
            "overloaded",
            response=httpx2.Response(500, request=httpx2.Request("POST", "https://x")),
            body=None,
        )
        agent, _ = make_claude([error])

        with self.assertRaisesRegex(AgentError, "500"):
            collect_text(agent.respond("Hi", RecordingContext()))

        self.assertEqual(agent.history, [])

    def test_interrupted_reply_recorded_as_partial(self):
        agent, _ = make_claude([claude_reply("one two three four")])
        ctx = RecordingContext()
        stream = agent.respond("count", ctx)

        next(stream)
        next(stream)
        stream.close()

        self.assertEqual(agent.history[-1], {"role": "assistant", "content": "one two  ..."})

    def test_unanswered_tool_use_is_closed_next_turn(self):
        agent, client = make_claude([claude_reply("Hello.")])
        agent.history = [
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": [block("tool_use", id="tu_9", name="x", input={})]},
        ]

        collect_text(agent.respond("Never mind", RecordingContext()))

        messages = client.requests[0]["messages"]
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "tu_9")
        self.assertEqual(messages[3], {"role": "user", "content": "Never mind"})

    def test_haiku_gets_no_adaptive_thinking(self):
        agent, client = make_claude([claude_reply("Hi.")], model="claude-haiku-4-5")

        collect_text(agent.respond("Hi", RecordingContext()))

        self.assertNotIn("thinking", client.requests[0])
        self.assertEqual(agent.describe(), "I'm using Claude Haiku 4.5.")

    def test_reset(self):
        agent, _ = make_claude([claude_reply("Hi.")])
        collect_text(agent.respond("Hi", RecordingContext()))

        agent.reset()

        self.assertEqual(agent.history, [])


# --------------------------------------------------
# LangGraph (real library, fake chat models)
# --------------------------------------------------

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolModel(BaseChatModel):
    """Returns scripted messages (which may contain tool calls)."""

    responses: list

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "fake-tool-model"


def streaming_model(*replies):
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def simple_graph(node, checkpointer=True):
    from typing import Annotated, TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    builder = StateGraph(State)
    builder.add_node("agent", node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver() if checkpointer else None)


class LangGraphAgentTests(unittest.TestCase):

    def test_streams_reply_tokens(self):
        model = streaming_model("Hello there friend.")
        graph = simple_graph(lambda state: {"messages": [model.invoke(state["messages"])]})

        text = collect_text(LangGraphAgent(graph).respond("hi", RecordingContext()))

        self.assertEqual(text, "Hello there friend.")

    def test_custom_events_become_status(self):
        from langgraph.config import get_stream_writer

        def node(state):
            writer = get_stream_writer()
            writer("Agent A started.")
            writer({"status": "Agent B finished.", "speak": False})
            return {"messages": [AIMessage(content="All done.")]}

        ctx = RecordingContext()
        text = collect_text(LangGraphAgent(simple_graph(node)).respond("go", ctx))

        statuses = ctx.of_type(Status)
        self.assertEqual([(s.text, s.speak) for s in statuses],
                         [("Agent A started.", True), ("Agent B finished.", False)])
        self.assertEqual(text, "All done.")

    def test_interrupt_is_asked_and_resumed(self):
        from langgraph.types import interrupt

        def node(state):
            answer = interrupt({"question": "Deploy to production?"})
            return {"messages": [AIMessage(content=f"You said {answer}.")]}

        ctx = RecordingContext(answers=["yes please"])
        text = collect_text(LangGraphAgent(simple_graph(node)).respond("deploy", ctx))

        self.assertEqual(ctx.asked, ["Deploy to production?"])
        self.assertEqual(text, "You said yes please.")

    def test_resume_builder(self):
        from langgraph.types import interrupt
        from voicekit import classify_confirmation

        def node(state):
            approved = interrupt("Deploy?")
            return {"messages": [AIMessage(content=f"approved={approved}")]}

        agent = LangGraphAgent(
            simple_graph(node),
            resume_builder=lambda answer, value: classify_confirmation(answer) is True,
        )

        text = collect_text(agent.respond("deploy", RecordingContext(answers=["Yeah, go ahead."])))

        self.assertEqual(text, "approved=True")

    def test_no_answer_stops(self):
        from langgraph.types import interrupt

        def node(state):
            interrupt("Anyone?")
            return {"messages": [AIMessage(content="never")]}

        text = collect_text(LangGraphAgent(simple_graph(node)).respond("x", RecordingContext()))

        self.assertEqual(text, "")

    def test_react_agent_tools_reported(self):
        from langchain_core.tools import tool as lc_tool
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.prebuilt import create_react_agent

        @lc_tool
        def run_tests(suite: str) -> str:
            """Run a test suite."""
            return f"{suite}: 42 passed"

        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "run_tests", "args": {"suite": "api"}, "id": "c1"}]),
            AIMessage(content="All 42 API tests passed."),
        ])
        graph = create_react_agent(model, [run_tests], checkpointer=InMemorySaver())
        ctx = RecordingContext()

        text = collect_text(LangGraphAgent(graph).respond("run the api tests", ctx))

        self.assertEqual(text, "All 42 API tests passed.")
        self.assertEqual([(e.name, e.arguments) for e in ctx.of_type(ToolStarted)],
                         [("run_tests", {"suite": "api"})])
        self.assertEqual(ctx.of_type(ToolFinished)[0].result, "api: 42 passed")

    def test_session_tools_inside_graph(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.prebuilt import create_react_agent

        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "set_speaking_speed",
                                               "args": {"change": "faster"}, "id": "c1"}]),
        ])
        graph = create_react_agent(model, to_langchain_tools(control_tools()),
                                   checkpointer=InMemorySaver())
        ctx = RecordingContext()

        collect_text(LangGraphAgent(graph).respond("talk faster", ctx))

        # return_direct: the session speaks the result; the graph ends.
        self.assertEqual(ctx.said, ["speed 1"])

    def test_subgraph_tokens_quiet_by_default(self):
        inner_model = streaming_model("inner chatter")
        inner = simple_graph(lambda s: {"messages": [inner_model.invoke(s["messages"])]},
                             checkpointer=False)

        def outer_node(state):
            inner.invoke(state)
            return {"messages": [AIMessage(content="Outer answer.")]}

        graph = simple_graph(outer_node)

        quiet = collect_text(LangGraphAgent(graph).respond("x", RecordingContext()))
        self.assertEqual(quiet, "Outer answer.")

    def test_without_checkpointer_history_is_kept(self):
        seen = []

        def node(state):
            seen.append(len(state["messages"]))
            return {"messages": [AIMessage(content="ok")]}

        agent = LangGraphAgent(simple_graph(node, checkpointer=False))
        collect_text(agent.respond("one", RecordingContext()))
        collect_text(agent.respond("two", RecordingContext()))

        self.assertEqual(seen, [1, 3])

        agent.reset()
        collect_text(agent.respond("three", RecordingContext()))
        self.assertEqual(seen[-1], 1)

    def test_graph_errors_become_agent_errors(self):
        def node(state):
            raise RuntimeError("node exploded")

        with self.assertRaisesRegex(AgentError, "node exploded"):
            collect_text(LangGraphAgent(simple_graph(node)).respond("x", RecordingContext()))

    def test_voice_context_outside_session_is_harmless(self):
        from voicekit.agents import voice_context

        ctx = voice_context()
        ctx.say("nobody listening")
        self.assertFalse(ctx.confirm("ok?"))


# --------------------------------------------------
# LangChain Runnable
# --------------------------------------------------

class LangChainAgentTests(unittest.TestCase):

    def test_chat_model_streams_and_keeps_history(self):
        model = streaming_model("Paris is the capital.", "About two million.")
        agent = LangChainAgent(model)

        first = collect_text(agent.respond("Capital of France?", RecordingContext()))
        collect_text(agent.respond("Population?", RecordingContext()))

        self.assertEqual(first, "Paris is the capital.")
        self.assertEqual([m.type for m in agent.history], ["human", "ai", "human", "ai"])

    def test_plain_runnable_with_input_builder(self):
        from langchain_core.runnables import RunnableLambda

        def answer(payload):
            yield "You asked: "
            yield payload["question"]

        agent = LangChainAgent(
            RunnableLambda(answer),
            input_builder=lambda messages: {"question": messages[-1].content},
        )

        self.assertEqual(collect_text(agent.respond("why?", RecordingContext())), "You asked: why?")

    def test_errors_become_agent_errors(self):
        from langchain_core.runnables import RunnableLambda

        def broken(payload):
            raise ValueError("chain broke")

        with self.assertRaisesRegex(AgentError, "chain broke"):
            collect_text(LangChainAgent(RunnableLambda(broken)).respond("x", RecordingContext()))


# --------------------------------------------------
# Live tests
# --------------------------------------------------

@unittest.skipUnless(RUN_SERVER_TESTS, "pass --server to run")
class LiveServerTests(unittest.TestCase):

    def test_ollama_agent_uses_a_tool(self):
        agent = OllamaAgent()
        agent.start()
        ctx = RecordingContext()

        text = collect_text(agent.respond("Show me a list of the available models.", ctx))

        print(f"\n  tools: {[e.name for e in ctx.of_type(ToolStarted)]}  reply: {text.strip()!r}")
        self.assertIn("list_models", [e.name for e in ctx.of_type(ToolStarted)])

    def test_langgraph_react_agent(self):
        from langchain_core.tools import tool as lc_tool
        from langchain_ollama import ChatOllama
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.config import get_stream_writer
        from langgraph.prebuilt import create_react_agent

        @lc_tool
        def run_tests(suite: str) -> str:
            """Run a named test suite and report the result."""
            get_stream_writer()(f"Running the {suite} tests.")
            return f"All 42 {suite} tests passed."

        model = ChatOllama(model=LLMConfig.model, base_url=LLMConfig.base_url, reasoning=False)
        graph = create_react_agent(model, [run_tests], checkpointer=InMemorySaver())
        ctx = RecordingContext()

        text = collect_text(LangGraphAgent(graph).respond("Run the backend tests.", ctx))

        print(f"\n  status: {[s.text for s in ctx.of_type(Status)]}  reply: {text.strip()!r}")
        self.assertTrue(ctx.of_type(ToolStarted))
        self.assertIn("42", text)


@unittest.skipUnless(RUN_CLAUDE_TESTS, "pass --claude to run (uses the Claude API)")
class LiveClaudeTests(unittest.TestCase):

    def test_claude_answers_and_uses_a_tool(self):
        @tool(return_direct=True)
        def set_lights(on: bool) -> str:
            """Turn the room lights on or off."""
            return f"Turning the lights {'on' if on else 'off'}."

        agent = ClaudeAgent(tools=[set_lights])
        agent.start()
        ctx = RecordingContext()

        text = collect_text(agent.respond("Could you switch the lights off?", ctx))

        print(f"\n  reply: {text.strip()!r}")
        self.assertIn("off", text)


if __name__ == "__main__":
    unittest.main()
