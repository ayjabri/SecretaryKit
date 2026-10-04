"""
Tests for voicekit/session.py: the generic voice loop.

    python tests/test_session.py
"""

import os
import sys
import time
import unittest

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit import (
    Agent,
    AgentError,
    FunctionAgent,
    SessionConfig,
    Status,
    TextDelta,
    Thinking,
    ToolFinished,
    ToolSet,
    ToolStarted,
    VoiceSession,
)
from voicekit.llm_client import ToolCall
from voicekit.tools import run_tool_calls

from tests.fakes import FakeRecognizer, FakeSpeaker


def make_session(agent, prompts, **config):
    log, events = [], []
    config.setdefault("greeting", None)
    config.setdefault("resume_delay_ms", 0)
    config.setdefault("wait_cue", None)

    recognizer = FakeRecognizer(prompts, log)
    speaker = FakeSpeaker(log)
    session = VoiceSession(
        agent, recognizer, speaker, SessionConfig(**config),
        on_event=lambda kind, text: events.append((kind, text)),
    )
    return session, recognizer, speaker, log, events


class EchoAgent(Agent):
    """Replies 'You said: <text>' and records calls."""

    def __init__(self):
        self.heard = []
        self.resets = 0

    def respond(self, text, ctx):
        self.heard.append(text)
        yield "You said: "
        yield TextDelta(text)

    def reset(self):
        self.resets += 1

    def describe(self):
        return "I'm the echo agent."


class ConversationTests(unittest.TestCase):

    def test_reply_is_spoken_and_shown(self):
        agent = EchoAgent()
        session, _, speaker, _, events = make_session(agent, ["Hello there"])

        session.run()

        self.assertEqual(agent.heard, ["Hello there"])
        self.assertEqual(speaker.spoken, ["You said: Hello there"])
        self.assertIn(("user", "Hello there"), events)
        shown = "".join(t for k, t in events if k == "assistant_text")
        self.assertEqual(shown, "You said: Hello there")

    def test_startup_order(self):
        session, _, _, log, _ = make_session(EchoAgent(), [])

        session.run()

        self.assertEqual(log[:2], ["speaker.load", "recognizer.load"])

    def test_greeting(self):
        session, _, speaker, _, _ = make_session(EchoAgent(), [], greeting="Hi!")

        session.run()

        self.assertEqual(speaker.spoken, ["Hi!"])

    def test_mic_muted_while_responding(self):
        session, _, _, log, _ = make_session(EchoAgent(), ["Hi"])

        session.run()

        i = log.index("stream:You said: Hi")
        self.assertEqual(log[i - 1], "mute")
        self.assertEqual(log[i + 1], "unmute")

    def test_mic_stays_on_for_speech_barge_in(self):
        session, recognizer, _, log, _ = make_session(
            EchoAgent(), ["Hi"], interruptible=True, barge_in="speech"
        )

        session.run()

        self.assertNotIn("mute", log)
        self.assertIsNone(recognizer.on_speech_start)

    def test_phrase_barge_in_mutes_and_watches(self):
        session, recognizer, _, log, _ = make_session(
            EchoAgent(), ["Hi"], interruptible=True
        )

        session.start()
        self.assertIsNotNone(recognizer.on_heard)
        self.assertIsNone(recognizer.on_speech_start)

        session.run()
        self.assertIn("mute", log)
        self.assertIsNone(recognizer.on_heard)

    def test_function_agent(self):
        session, _, speaker, _, _ = make_session(
            FunctionAgent(lambda text: f"Echo {text}"), ["one", "two"]
        )

        session.run()

        self.assertEqual(speaker.spoken, ["Echo one", "Echo two"])


class EventTests(unittest.TestCase):

    def test_status_spoken_in_order_and_shown(self):
        def agent(text, ctx):
            yield "Starting."
            yield Status("Agent A finished.")
            yield "All done."

        session, _, speaker, _, events = make_session(FunctionAgent(agent), ["Go"])

        session.run()

        self.assertEqual(speaker.spoken, ["Starting. Agent A finished. All done."])
        self.assertIn(("progress", "Agent A finished."), events)

    def test_silent_status_only_shown(self):
        def agent(text, ctx):
            yield Status("internal detail", speak=False)
            yield "Done."

        session, _, speaker, _, events = make_session(FunctionAgent(agent), ["Go"])

        session.run()

        self.assertEqual(speaker.spoken, ["Done."])
        self.assertIn(("progress", "internal detail"), events)

    def test_status_can_be_muted_by_config(self):
        def agent(text, ctx):
            yield Status("progress")
            yield "Done."

        session, _, speaker, _, _ = make_session(
            FunctionAgent(agent), ["Go"], speak_status=False
        )

        session.run()

        self.assertEqual(speaker.spoken, ["Done."])

    def test_emitted_events_from_callbacks(self):
        def agent(text, ctx):
            ctx.emit(Thinking("hmm "))
            ctx.emit(ToolStarted("lookup", {"q": "x"}))
            ctx.emit(ToolFinished("lookup", "found"))
            ctx.emit(Status("Halfway there."))
            yield "Answer."

        session, _, speaker, _, events = make_session(FunctionAgent(agent), ["Q"])

        session.run()

        kinds = [k for k, _ in events]
        self.assertIn("thinking_start", kinds)
        self.assertIn(("thinking", "hmm "), events)
        self.assertIn(("tool", "lookup(q=x)"), events)
        self.assertIn(("tool_result", "lookup: found"), events)
        self.assertEqual(speaker.spoken, ["Halfway there.", "Answer."])
        self.assertLess(kinds.index("thinking_end"), kinds.index("assistant_start"))

    def test_thinking_text_can_be_hidden(self):
        def agent(text, ctx):
            ctx.emit(Thinking("secret"))
            yield "Hi."

        session, _, _, _, events = make_session(
            FunctionAgent(agent), ["Q"], show_thinking=False
        )

        session.run()

        kinds = [k for k, _ in events]
        self.assertIn("thinking_start", kinds)
        self.assertNotIn("thinking", kinds)


class AskTests(unittest.TestCase):

    def test_ask_returns_next_utterance(self):
        answers = []

        def agent(text, ctx):
            answers.append(ctx.ask("Which environment?"))
            yield f"Deploying to {answers[-1]}."

        session, _, speaker, _, events = make_session(
            FunctionAgent(agent), ["Deploy it", "staging"]
        )

        session.run()

        self.assertEqual(answers, ["staging"])
        self.assertEqual(speaker.spoken, ["Which environment?", "Deploying to staging."])
        # The answer was consumed by ask(), not handled as a new prompt.
        self.assertEqual([t for k, t in events if k == "user"], ["Deploy it", "staging"])

    def test_confirm(self):
        results = []

        def agent(text, ctx):
            results.append(ctx.confirm("Deploy to production?"))
            yield "Ok."

        session, _, _, _, _ = make_session(
            FunctionAgent(agent), ["go", "Yes, go ahead.", "again", "No thanks", "again", "purple"]
        )

        session.run()

        self.assertEqual(results, [True, False, False])

    def test_ask_unmutes_microphone(self):
        def agent(text, ctx):
            ctx.ask("Sure?")
            yield "Ok."

        session, _, _, log, _ = make_session(FunctionAgent(agent), ["go", "yes"])

        session.run()

        i = log.index("say:Sure?")
        self.assertEqual(log[i + 1:i + 3], ["unmute", "mute"])

    def test_ask_outside_session_raises(self):
        session, _, _, _, _ = make_session(EchoAgent(), [])

        with self.assertRaises(AgentError):
            session.ask("Anyone there?")


class CommandTests(unittest.TestCase):

    def test_goodbye_ends_session(self):
        agent = EchoAgent()
        session, _, speaker, _, _ = make_session(agent, ["Goodbye.", "never heard"])

        session.run()

        self.assertEqual(agent.heard, [])
        self.assertEqual(speaker.spoken, ["Goodbye!"])

    def test_dismissed(self):
        session, _, speaker, _, _ = make_session(EchoAgent(), ["Dismissed.", "x"])

        session.run()

        self.assertEqual(speaker.spoken, ["Goodbye!"])

    def test_goodbye_inside_sentence_goes_to_agent(self):
        agent = EchoAgent()
        session, _, _, _, _ = make_session(agent, ["How do I say goodbye in French?"])

        session.run()

        self.assertEqual(agent.heard, ["How do I say goodbye in French?"])

    def test_new_conversation_resets_agent(self):
        agent = EchoAgent()
        session, _, speaker, _, _ = make_session(agent, ["New conversation!"])

        session.run()

        self.assertEqual(agent.resets, 1)
        self.assertEqual(speaker.spoken, ["Starting a new conversation."])

    def test_speaking_speed(self):
        session, _, speaker, _, _ = make_session(
            EchoAgent(), ["Speak faster.", "Talk a bit slower.", "Speak slower.", "Normal speed."]
        )
        speeds = []
        speaker.speak = lambda text, wait=False: speeds.append(speaker.config.speed)

        session.run()

        self.assertEqual(speeds, [1.1, 1.0, 0.9, 1.0])

    def test_status_uses_agent_description(self):
        session, _, speaker, _, _ = make_session(EchoAgent(), ["Status."])

        session.run()

        self.assertEqual(speaker.spoken,
                         ["I'm the echo agent. My speaking speed is 1.0."])

    def test_help(self):
        session, _, speaker, _, _ = make_session(EchoAgent(), ["Help."])

        session.run()

        for phrase in ("goodbye", "speak faster", "status"):
            self.assertIn(phrase, speaker.spoken[0])

    def test_agent_can_register_commands(self):
        class Commanding(EchoAgent):
            def register_commands(self, router):
                router.add("lights", r"lights on", lambda: "Lights on.", "lights on")

        session, _, speaker, _, _ = make_session(Commanding(), ["Lights on!"])

        session.run()

        self.assertEqual(speaker.spoken, ["Lights on."])

    def test_command_prefix(self):
        agent = EchoAgent()
        session, _, speaker, _, _ = make_session(
            agent, ["Speak faster.", "Computer, speak faster."], command_prefix="computer"
        )

        session.run()

        self.assertEqual(agent.heard, ["Speak faster."])
        self.assertEqual(speaker.config.speed, 1.1)


class ControlToolTests(unittest.TestCase):

    class ToolAgent(Agent):
        """Calls whichever tool the utterance names, like a model would."""

        def __init__(self):
            self.tools = ToolSet()

        def respond(self, text, ctx):
            name, _, argument = text.partition(" ")
            arguments = {"change": argument} if argument else {}
            outcome = run_tool_calls(self.tools, [ToolCall(name, arguments)], ctx)
            yield " ".join(outcome.direct or outcome.results)

    def test_session_adds_control_tools(self):
        agent = self.ToolAgent()
        make_session(agent, [])

        self.assertEqual(
            sorted(t.name for t in agent.tools),
            ["end_session", "get_status", "new_conversation", "set_speaking_speed"],
        )

    def test_control_tools_can_be_disabled(self):
        agent = self.ToolAgent()
        make_session(agent, [], control_tools=False)

        self.assertEqual(len(agent.tools), 0)

    def test_tools_act_on_the_session(self):
        agent = self.ToolAgent()
        session, _, speaker, _, events = make_session(
            agent, ["set_speaking_speed faster", "get_status", "end_session", "never heard"]
        )

        session.run()

        self.assertEqual(speaker.config.speed, 1.1)
        self.assertEqual(speaker.spoken, [
            "Speaking faster.",
            "My speaking speed is 1.1.",
            "Goodbye!",
        ])
        self.assertIn(("tool", "set_speaking_speed(change=faster)"), events)


class ErrorTests(unittest.TestCase):

    def test_agent_error_is_spoken_and_session_continues(self):
        calls = []

        def agent(text, ctx):
            calls.append(text)
            if text == "first":
                raise AgentError("server down")
            yield "Back online."

        session, _, speaker, _, events = make_session(FunctionAgent(agent), ["first", "second"])

        session.run()

        self.assertEqual(speaker.spoken, ["Sorry, something went wrong.", "Back online."])
        self.assertIn(("error", "server down"), events)

    def test_unexpected_exception_does_not_end_session(self):
        def agent(text, ctx):
            if text == "boom":
                yield "Partial"
                raise ValueError("bad state")
            yield "Fine."

        session, _, speaker, _, events = make_session(FunctionAgent(agent), ["boom", "next"])

        with self.assertLogs("voicekit.session", "ERROR"):
            session.run()

        self.assertEqual(speaker.spoken[-1], "Fine.")
        self.assertTrue(any(k == "error" and "bad state" in t for k, t in events))

    def test_start_failure_propagates(self):
        class Broken(EchoAgent):
            def start(self):
                raise AgentError("Cannot reach the LLM server")

        session, _, _, _, _ = make_session(Broken(), ["Hi"])

        with self.assertRaises(AgentError):
            session.run()


class InterruptTests(unittest.TestCase):

    def test_barge_in_stops_reply_and_closes_agent(self):
        closed = []

        def agent(text, ctx):
            try:
                for i in range(50):
                    if ctx.cancelled:
                        return
                    yield f"word{i} "
            finally:
                closed.append(True)

        session, recognizer, speaker, _, events = make_session(
            FunctionAgent(agent), ["Tell me a story"], interruptible=True,
            barge_in="speech",
        )

        count = [0]

        def on_token(token):
            count[0] += 1
            if count[0] == 3:
                recognizer.on_speech_start()

        speaker.on_token = on_token

        session.run()

        self.assertIn(("interrupted", ""), events)
        self.assertEqual(closed, [True])
        self.assertTrue(session.cancelled)

    def test_speech_while_idle_stops_nothing(self):
        session, _, _, log, _ = make_session(
            EchoAgent(), [], interruptible=True, barge_in="speech"
        )
        session.start()

        session._on_user_speech()

        self.assertNotIn("stop", log)

    def test_speech_while_asking_is_the_answer(self):
        session, _, _, log, _ = make_session(
            EchoAgent(), [], interruptible=True, barge_in="speech"
        )
        session._responding = True
        session._asking = True

        session._on_user_speech()

        self.assertNotIn("stop", log)

    def _story_session(self, heard_at_token3):
        """A long reply; the recognizer 'hears' something at token 3."""

        def agent(text, ctx):
            for i in range(50):
                if ctx.cancelled:
                    return
                yield f"word{i} "

        session, recognizer, speaker, _, events = make_session(
            FunctionAgent(agent), ["Tell me a story"], interruptible=True
        )
        count = [0]

        def on_token(token):
            count[0] += 1
            if count[0] == 3:
                recognizer.on_heard(heard_at_token3)

        speaker.on_token = on_token
        session.run()
        return session, events

    def test_stop_phrase_stops_reply(self):
        for heard in ("Hang on.", "Okay, stop there!", "word2 word3 wait"):
            with self.subTest(heard=heard):
                session, events = self._story_session(heard)
                self.assertIn(("interrupted", ""), events)
                self.assertTrue(session.cancelled)

    def test_other_speech_does_not_stop_reply(self):
        for heard in ("word1 word2", "What a lovely day", "Stopped by the shop"):
            with self.subTest(heard=heard):
                session, events = self._story_session(heard)
                self.assertNotIn(("interrupted", ""), events)

    def test_own_words_are_not_a_stop_request(self):
        session, _, _, _, _ = make_session(EchoAgent(), [], interruptible=True)
        session._said = "Please wait while I check."

        self.assertFalse(session.is_stop_request("please wait while"))
        self.assertTrue(session.is_stop_request("hang on"))

    def test_overheard_while_idle_or_asking_stops_nothing(self):
        session, _, _, log, _ = make_session(EchoAgent(), [], interruptible=True)
        session._on_overheard("stop")
        session._responding = True
        session._asking = True
        session._on_overheard("stop")

        self.assertNotIn("stop", log)

    def test_stop_phrase_alone_is_not_sent_to_agent(self):
        agent = EchoAgent()
        session, _, speaker, _, _ = make_session(agent, ["Hang on.", "Hi"])

        session.run()

        self.assertEqual(agent.heard, ["Hi"])
        self.assertEqual(speaker.spoken, ["You said: Hi"])

    def test_invalid_barge_in(self):
        with self.assertRaises(ValueError):
            make_session(EchoAgent(), [], barge_in="loud")


class WaitCueTests(unittest.TestCase):

    def test_slow_agent_gets_a_cue(self):
        def agent(text, ctx):
            time.sleep(0.3)
            yield "Finally."

        session, _, speaker, _, _ = make_session(
            FunctionAgent(agent), ["Q"], wait_cue="One moment.", wait_cue_after_s=0.05
        )

        session.run()

        self.assertEqual(speaker.spoken, ["One moment.", "Finally."])

    def test_fast_agent_gets_no_cue(self):
        session, _, speaker, _, _ = make_session(
            EchoAgent(), ["Q"], wait_cue="One moment.", wait_cue_after_s=0.5
        )

        session.run()
        time.sleep(0.6)

        self.assertEqual(speaker.spoken, ["You said: Q"])


class ConsoleTests(unittest.TestCase):

    def test_keyboard_session(self):
        from voicekit.console import KeyboardInput, SilentSpeaker

        speaker = SilentSpeaker()
        session = VoiceSession(
            EchoAgent(), KeyboardInput(["hello", "goodbye", "never"]), speaker,
            SessionConfig(greeting="Hi.", resume_delay_ms=0, wait_cue=None),
            on_event=None,
        )

        session.run()

        self.assertEqual(speaker.spoken, ["Hi.", "You said: hello", "Goodbye!"])

    def test_tool_lines_close_the_reply_line(self):
        def agent(text, ctx):
            yield "Checking."
            ctx.emit(ToolStarted("lookup"))
            yield "Found it."

        session, _, _, _, events = make_session(FunctionAgent(agent), ["Q"])

        session.run()

        kinds = [k for k, _ in events]
        tool_index = kinds.index("tool")
        self.assertEqual(kinds[tool_index - 1], "assistant_end")
        self.assertEqual(kinds[tool_index + 1], "assistant_start")


class VocabularyTests(unittest.TestCase):

    def test_agent_vocabulary_taught_to_recognizer(self):
        class Named(EchoAgent):
            def vocabulary(self):
                return ["Qwen", "Llama"]

        session, recognizer, _, _, _ = make_session(Named(), [])

        session.run()

        self.assertEqual(recognizer.config.hotwords, "Qwen, Llama")

    def test_user_vocabulary_not_overwritten(self):
        class Named(EchoAgent):
            def vocabulary(self):
                return ["Qwen"]

        session, recognizer, _, _, _ = make_session(Named(), [])
        recognizer.config.hotwords = "Kubernetes"

        session.run()

        self.assertEqual(recognizer.config.hotwords, "Kubernetes")


if __name__ == "__main__":
    unittest.main()
