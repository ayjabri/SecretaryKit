"""
Agent for Claude in the cloud, via the official Anthropic SDK.

    pip install anthropic
    # credentials: ANTHROPIC_API_KEY, or `ant auth login`

    agent = ClaudeAgent()                     # claude-opus-5-5
    agent.tools.add(my_tool)                  # Claude decides when to call it
    VoiceSession(agent).run()

Replies stream token by token so speech starts immediately. Tool calls
run in a loop until Claude answers in words. The conversation history is
append-only, as Claude's preserved thinking requires.
"""

import logging

from ..agent import Agent, AgentError, TextDelta, Thinking
from ..llm_client import VOICE_SYSTEM_PROMPT, ToolCall
from ..tools import ToolSet, run_tool_calls


logger = logging.getLogger(__name__)


TOOLS_NOTE = (
    "\n\nYou can call the provided tools. Use one only when the user's "
    "request needs it. Tools that change a setting take effect immediately "
    "and the app tells the user, so don't ask for confirmation or announce "
    "the change yourself."
)

# Opt-in server-side retry on another model if a request is declined.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeAgent(Agent):
    """
    model
        Claude model ID. Default: claude-opus-5-5.

    system_prompt
        Defaults to the voice prompt (short spoken replies, no markdown).

    tools
        voicekit Tools (or typed functions) Claude may call.

    effort
        "low" | "medium" | "high" | "xhigh" | "max". Lower is faster and
        cheaper; "low" suits conversation. Raise it for hard tasks.

    show_thinking
        Stream a summary of Claude's reasoning to the screen.

    fallbacks
        If Claude declines a request, let the API retry it on a suitable
        fallback model within the same call.

    confirm_tools
        Ask the user before running tools that define a confirmation
        question.

    client
        An anthropic.Anthropic() instance (or compatible); created on
        start() if omitted.
    """

    def __init__(self, model="claude-opus-5-5", system_prompt=VOICE_SYSTEM_PROMPT,
                 tools=(), effort="low", show_thinking=True, fallbacks=True,
                 confirm_tools=False, max_tokens=64000, max_tool_rounds=8,
                 client=None):

        self.model = model
        self.system_prompt = system_prompt
        self.tools = ToolSet(tools)
        self.effort = effort
        self.show_thinking = show_thinking
        self.fallbacks = fallbacks
        self.confirm_tools = confirm_tools
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.client = client
        self.history = []

    # --------------------------------------------------
    # Agent interface
    # --------------------------------------------------

    def start(self):

        import anthropic

        if self.client is None:
            self.client = anthropic.Anthropic()

        # Fail fast on missing credentials or an unknown model.
        try:
            self.client.models.retrieve(self.model)
        except anthropic.AuthenticationError as exc:
            raise AgentError(
                "Claude: no valid credentials. Set ANTHROPIC_API_KEY or "
                "run `ant auth login`."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise AgentError(f"Claude: unknown model '{self.model}'.") from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError("Cannot reach the Claude API.") from exc

    def respond(self, text, ctx):

        import anthropic

        if self.client is None:
            self.start()

        history = self.history
        self._answer_open_tool_uses(history)
        history.append({"role": "user", "content": text})
        turn_start = len(history) - 1

        spoken = []
        finished = False
        failed = False

        try:
            for _ in range(self.max_tool_rounds + 1):
                spoken = []

                with self._stream(history) as stream:
                    for event in stream:
                        if ctx.cancelled:
                            return

                        if event.type != "content_block_delta":
                            continue

                        delta = event.delta

                        if delta.type == "text_delta" and delta.text:
                            spoken.append(delta.text)
                            yield TextDelta(delta.text)

                        elif delta.type == "thinking_delta" and delta.thinking:
                            ctx.emit(Thinking(delta.thinking))

                    message = stream.get_final_message()

                # The full content, including thinking blocks, goes back
                # unchanged on the next request.
                history.append({"role": "assistant", "content": message.content})
                spoken = []

                if message.stop_reason == "refusal":
                    yield TextDelta(" Sorry, I can't help with that.")
                    break

                if message.stop_reason == "pause_turn":
                    continue

                tool_uses = [b for b in message.content if b.type == "tool_use"]

                if message.stop_reason != "tool_use" or not tool_uses:
                    # end_turn, or max_tokens (a truncated tool input is
                    # never run).
                    break

                calls = [
                    ToolCall(b.name, b.input if isinstance(b.input, dict) else {}, b.id)
                    for b in tool_uses
                ]
                outcome = run_tool_calls(self.tools, calls, ctx, confirm=self.confirm_tools)

                # All results in one user message.
                history.append({"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result,
                        **({"is_error": True} if is_error else {}),
                    }
                    for call, result, is_error
                    in zip(calls, outcome.results, outcome.errors)
                ]})

                if outcome.ends_turn:
                    yield TextDelta(" ".join(outcome.direct))
                    break

            finished = True

        except anthropic.APIStatusError as exc:
            failed = True
            raise AgentError(f"Claude API error {exc.status_code}: {exc.message}") from exc

        except anthropic.APIConnectionError as exc:
            failed = True
            raise AgentError("Cannot reach the Claude API.") from exc

        except ValueError as exc:
            # Tool-call JSON the SDK could not parse at all.
            failed = True
            raise AgentError("Claude returned an unreadable tool call.") from exc

        finally:
            if failed and len(history) == turn_start + 1:
                # Nothing was answered; forget the question so it can be
                # asked again.
                history.pop()
            elif not finished and spoken:
                # Interrupted: record what the user actually heard.
                history.append({"role": "assistant", "content": "".join(spoken) + " ..."})

    def reset(self):
        self.history = []

    def describe(self):
        return f"I'm using {spoken_claude_name(self.model)}."

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------

    def _stream(self, history):

        system = self.system_prompt or ""
        if len(self.tools):
            system += TOOLS_NOTE

        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": history,
        }

        if system:
            params["system"] = system

        if len(self.tools):
            params["tools"] = self.tools.anthropic_schemas()

        if not self.model.startswith("claude-haiku"):
            params["thinking"] = {
                "type": "adaptive",
                "display": "summarized" if self.show_thinking else "omitted",
            }
            params["output_config"] = {"effort": self.effort}

        if self.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"

        return self.client.beta.messages.stream(**params)

    @staticmethod
    def _answer_open_tool_uses(history):
        """
        If the last turn was interrupted between a tool call and its
        result, answer the call so the history stays valid.
        """

        if not history or history[-1]["role"] != "assistant":
            return

        content = history[-1]["content"]

        if isinstance(content, str):
            return

        open_ids = [
            getattr(block, "id", None) for block in content
            if getattr(block, "type", None) == "tool_use"
        ]

        if open_ids:
            history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_id,
                 "content": "Not run: the user interrupted."}
                for tool_id in open_ids
            ]})


def spoken_claude_name(model):
    """'claude-opus-5-5' -> 'Claude Opus 5.5'."""

    parts = model.split("-")

    if len(parts) >= 3 and parts[0] == "claude":
        family = parts[1].capitalize()
        version = ".".join(p for p in parts[2:] if p.isdigit() and len(p) <= 2)
        return f"Claude {family} {version}".strip()

    return model
