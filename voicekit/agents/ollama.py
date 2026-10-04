"""
Agent for a model on Ollama or any OpenAI-compatible server (vLLM,
LM Studio, llama.cpp, LiteLLM...).

    agent = OllamaAgent(LLMConfig(base_url="http://10.0.0.100:11434",
                                  model="qwen3.5:4b"))
    agent.tools.add(my_tool)          # the model decides when to call it
    VoiceSession(agent).run()

Out of the box the model can also manage itself by voice: list models,
switch model and turn thinking on or off ("go with the bigger Qwen",
"think harder"). Disable with settings_tools=False.
"""

import logging
import re
import threading

from ..agent import Agent, TextDelta, Thinking
from ..commands import NOT_HANDLED
from ..llm_client import LLMClient, LLMConfig, LLMError
from ..tools import Tool, ToolSet, run_tool_calls


logger = logging.getLogger(__name__)


TOOLS_PROMPT = (
    "You can call the provided tools. Use one only when the user's request "
    "needs it; answer ordinary questions directly. Tools that change a "
    "setting take effect immediately and the app tells the user, so don't "
    "ask for confirmation or announce the change yourself."
)


class OllamaAgent(Agent):
    """
    llm
        An LLMClient, an LLMConfig, or None for the defaults.

    tools
        Tools (voicekit.tools.Tool or plain typed functions) the model may
        call. More can be added later via agent.tools.add(...).

    settings_tools
        Let the model list/switch models and toggle thinking.

    confirm_tools
        Ask the user before running tools that define a confirmation
        question. By default they run straight away and announce
        themselves.

    use_tools
        Set False to never offer tools (e.g. for models that misuse them).

    tools_prompt
        Appended to the system prompt when tools are offered.
    """

    def __init__(self, llm=None, tools=(), settings_tools=True,
                 confirm_tools=False, use_tools=True, tools_prompt=TOOLS_PROMPT):

        self.llm = llm if isinstance(llm, LLMClient) else LLMClient(llm or LLMConfig())
        self.tools = ToolSet(tools)
        self.confirm_tools = confirm_tools
        self.use_tools = use_tools
        self.tools_prompt = tools_prompt

        self._models = []
        self._tool_support = {}

        if settings_tools:
            self.tools.add(*self.settings_tools())

    # --------------------------------------------------
    # Agent interface
    # --------------------------------------------------

    def start(self):

        self.llm.check_connection()
        self._models = self.llm.list_models()

        prompt = self.llm.config.system_prompt or ""
        if self.use_tools and self.tools_prompt and self.tools_prompt not in prompt:
            self.llm.config.system_prompt = f"{prompt}\n\n{self.tools_prompt}".strip()

        # Load the model into GPU memory on the server in the background.
        threading.Thread(target=self.llm.warm_up, daemon=True).start()

    def respond(self, text, ctx):

        schemas = self._tool_schemas()
        direct_runs = []

        def on_tool_calls(calls):
            outcome = run_tool_calls(self.tools, calls, ctx, confirm=self.confirm_tools)

            if outcome.ends_turn:
                # Speak the result ourselves and end the model's turn, so
                # it doesn't add a second, redundant reply.
                direct_runs.append(outcome)
                return None

            return outcome.results

        stream = self.llm.stream_reply(
            text,
            on_thinking=lambda t: ctx.emit(Thinking(t)),
            tools=schemas,
            on_tool_calls=on_tool_calls if schemas else None,
        )

        said_anything = False

        try:
            for token in stream:
                if ctx.cancelled:
                    return
                said_anything = True
                yield TextDelta(token)
        finally:
            stream.close()

        if direct_runs:
            outcome = direct_runs[0]

            # A tool such as new_conversation may have cleared the history.
            if self.llm.pending_tool_calls:
                self.llm.add_tool_results(outcome.results)

            announcement = " ".join(outcome.direct)
            yield TextDelta((" " if said_anything else "") + announcement)

    def reset(self):
        self.llm.reset()

    def describe(self):
        return (
            f"I'm using {spoken_model_name(self.llm.config.model)}, and "
            f"thinking is {'on' if self.llm.config.think else 'off'}."
        )

    def vocabulary(self):
        """Model family names, so Whisper spells 'Qwen' not 'Quinn'."""
        return sorted({spoken_model_name(m).split(" ")[0] for m in self._models})

    def close(self):
        self.llm.close()

    def register_commands(self, router):

        router.add("think_off", THINK_OFF_PATTERNS,
                   lambda: self.set_thinking(False), "thinking off")
        router.add("think_on", THINK_ON_PATTERNS,
                   lambda: self.set_thinking(True), "thinking on")
        router.add("list_models", LIST_MODELS_PATTERNS,
                   self._list_models_reply, "list models")
        router.add("switch_model", MODEL_PATTERNS,
                   lambda model: self.switch_model(model, strict=True),
                   "switch model to, followed by a name")
        router.add("switch_model_loose", MODEL_LOOSE_PATTERNS,
                   lambda model: self.switch_model(model, strict=False))

    # --------------------------------------------------
    # Settings (shared by voice commands and tools)
    # --------------------------------------------------

    def set_thinking(self, on):

        name = spoken_model_name(self.llm.config.model)

        if on == self.llm.config.think:
            return f"Thinking is already {'on' if on else 'off'}."

        if on and self._supports_thinking() is False:
            return f"{name} doesn't support thinking."

        self.llm.config.think = on

        if on:
            return "Turning thinking on. Replies will take longer."
        return "Turning thinking off."

    def switch_model(self, model, strict=True):
        """
        Switch to a model named loosely ("Quinn three point five").
        With strict=False, returns NOT_HANDLED when nothing matches, so a
        loose voice command falls through to the model.
        """

        try:
            models = self.llm.list_models()
        except LLMError:
            return "Sorry, I couldn't get the model list from the server."

        self._models = models
        resolved = self.llm.resolve_model(model, models)

        if resolved is None:
            if not strict:
                return NOT_HANDLED
            return (
                f"I couldn't find a model called {model}. Available models "
                f"are {join_spoken(spoken_model_name(m) for m in models)}."
            )

        name = spoken_model_name(resolved)

        if resolved == self.llm.config.model:
            return f"I'm already using {name}."

        self.llm.config.model = resolved
        threading.Thread(target=self.llm.warm_up, daemon=True).start()

        reply = f"Switching to {name}."

        if self.llm.config.think and self._supports_thinking() is False:
            self.llm.config.think = False
            reply += " It doesn't support thinking, so thinking is off."

        return reply

    def settings_tools(self):
        """Tools that let the model manage its own model and thinking mode."""

        def list_models():
            try:
                self._models = self.llm.list_models()
            except LLMError:
                return "Error: the server's model list is unavailable."
            # Spoken-style names, since the model tends to repeat them aloud.
            return (
                "Installed models: "
                f"{', '.join(spoken_model_name(m) for m in self._models)}. "
                f"Current model: {spoken_model_name(self.llm.config.model)}."
            )

        return [
            Tool(
                name="list_models",
                description=(
                    "List the language models available on the server and "
                    "say which one is in use."
                ),
                run=list_models,
            ),
            Tool(
                name="switch_model",
                description=(
                    "Switch the assistant to a different language model. "
                    "Only when the user asks to change or use another model."
                ),
                parameters={"model": {
                    "type": "string",
                    "description": "Exact name of an installed model.",
                }},
                required=("model",),
                run=lambda model: self.switch_model(model, strict=True),
                return_direct=True,
                confirm=lambda model: f"Switch to {spoken_model_name(model)}?",
            ),
            Tool(
                name="set_thinking",
                description=(
                    "Turn the model's step-by-step thinking mode on or off. "
                    "On gives more careful but much slower answers."
                ),
                parameters={"enabled": {"type": "boolean"}},
                required=("enabled",),
                run=lambda enabled: self.set_thinking(bool(enabled)),
                return_direct=True,
                confirm=lambda enabled: f"Turn thinking {'on' if enabled else 'off'}?",
            ),
        ]

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------

    def _list_models_reply(self):
        try:
            self._models = self.llm.list_models()
        except LLMError:
            return "Sorry, I couldn't get the model list from the server."

        names = join_spoken(spoken_model_name(m) for m in self._models)
        current = spoken_model_name(self.llm.config.model)
        return f"Available models are {names}. I'm using {current}."

    def _supports_thinking(self):
        try:
            return self.llm.supports_thinking()
        except LLMError:
            return None  # unknown; let the server decide

    def _tool_schemas(self):
        """Tool definitions for this request, or None if not usable."""

        if not self.use_tools or not len(self.tools):
            return None

        model = self.llm.config.model

        if model not in self._tool_support:
            try:
                capabilities = self.llm.model_capabilities()
            except LLMError:
                capabilities = None
            self._tool_support[model] = capabilities is None or "tools" in capabilities

        if not self._tool_support[model]:
            return None

        # Offer exactly the installed models, so the model can't invent one.
        switch = self.tools.get("switch_model")
        if switch is not None and self._models:
            switch.parameters["model"]["enum"] = list(self._models)

        return self.tools.schemas()


# --------------------------------------------------
# Spoken names and exact command patterns
# --------------------------------------------------

def spoken_model_name(model):
    """'qwen3.6:27b' -> 'Qwen 3.6 27B', for reading aloud."""

    name = re.sub(r":latest$", "", model)
    family, _, size = name.partition(":")
    family = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", family).replace("-", " ")
    family = family[:1].upper() + family[1:]

    return f"{family} {size.upper()}".strip()


def join_spoken(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# Matched against the whole utterance after normalisation (lower case,
# punctuation replaced by spaces): "Set no-think." -> "set no think".
_THINK = r"think(?:ing)?(?: mode)?"

THINK_ON_PATTERNS = (
    rf"(?:set |turn on |enable |start |use )?(?:the )?{_THINK}",
    rf"(?:set |turn |switch )?{_THINK} on",
)
THINK_OFF_PATTERNS = (
    rf"(?:set )?no ?{_THINK}",
    rf"(?:turn off|switch off|disable|stop)(?: the)? {_THINK}",
    rf"(?:set |turn |switch )?{_THINK} off",
    r"don t think",
)
MODEL_PATTERNS = (
    r"(?:switch|change|set)(?: the)? model to (?P<model>.+)",
    r"(?:set|use|load)(?: the)? model (?P<model>.+)",
)
# Looser forms; they fall through to the model unless the name matches.
MODEL_LOOSE_PATTERNS = (r"(?:switch to|use|load) (?P<model>.+)",)
LIST_MODELS_PATTERNS = (
    r"(?:list|show)(?: the)?(?: available)? models",
    r"(?:what|which) models (?:are available|are there|do you have|can i use)",
)
