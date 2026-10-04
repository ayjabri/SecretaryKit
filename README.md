# voicekit

Talk to any AI agent by voice. Speech recognition and speech synthesis run on your own GPU; the
"brain" can be anything: a local Ollama model, Claude in the cloud, or a LangGraph orchestrator
running a whole team of agents.

```
microphone → Whisper ─┐                        ┌─→ Kokoro → speakers
                      └→ VoiceSession ⇄ Agent ─┘
                                        │
             OllamaAgent · ClaudeAgent · LangGraphAgent · LangChainAgent · FunctionAgent · your own
```

`VoiceSession` is the voice loop and knows nothing about models or tools. An **agent** turns what
you said into a stream of events: reply text, progress updates, reasoning, tool activity. The
session speaks the reply while it streams, says progress updates aloud, shows the rest on screen,
and lets the agent ask you questions mid-task.

## Install

```bash
pip install "voicekit[cuda,langgraph,claude] @ git+https://github.com/ayjabri/voicekit.git"
```

Extras: `cuda` (GPU libraries for Whisper), `claude` (`anthropic`), `langgraph` (`langgraph`,
`langchain-core`, `langchain-ollama`), `all`. On a GPU, install a CUDA build of PyTorch first
(Kokoro needs it): https://pytorch.org/get-started/locally/. For development: `pip install -e ".[all]"`.

## Quick start

```python
from voicekit import VoiceSession
from voicekit.agents import OllamaAgent

VoiceSession(OllamaAgent()).run()          # qwen3.5:4b on http://10.0.0.100:11434
```

Or the ready-made app: `python voice_assistant.py` (see below).

## Choosing a brain

```python
from voicekit import LLMConfig
from voicekit.agents import OllamaAgent, ClaudeAgent, LangGraphAgent, LangChainAgent, FunctionAgent

OllamaAgent(LLMConfig(model="qwen3.6:27b"))                 # Ollama
OllamaAgent(LLMConfig(api="openai", base_url="http://agents:8000", model="my-agent"))  # vLLM, LM Studio...
ClaudeAgent()                                               # Claude Opus 5.5 (ANTHROPIC_API_KEY or `ant auth login`)
LangGraphAgent(compiled_graph)                              # any LangGraph graph
LangChainAgent(chat_model_or_chain)                         # any LangChain Runnable
FunctionAgent(lambda text, ctx: f"You said {text}")         # any Python function
```

Write your own by subclassing `voicekit.Agent` and implementing `respond(text, ctx)`:

```python
from voicekit import Agent, Status

class Orchestrator(Agent):
    def respond(self, text, ctx):
        yield Status("Sending that to the build agent.")    # spoken progress update
        if not ctx.confirm("This touches production. Go ahead?"):
            yield "Okay, cancelled."
            return
        result = run_build(text)                            # your code
        yield f"Done. {result}"                             # the reply (plain strings are fine)
```

`ctx` gives agents: `say(text)`, `ask(question) -> answer`, `confirm(question) -> bool`,
`emit(event)` (from callbacks or threads), and `cancelled` (the user interrupted).

## Tools: the model decides

Tool-calling agents (`OllamaAgent`, `ClaudeAgent`) take tools defined from plain functions. The
schema comes from type hints and the docstring, and the model decides when to call them:

```python
from typing import Literal
from voicekit import tool

@tool
def set_lights(room: str, state: Literal["on", "off"]) -> str:
    """Switch the lights in a room.

    Args:
        room: Which room, e.g. kitchen.
    """
    return f"The {room} lights are {state}."

agent = OllamaAgent(tools=[set_lights])
```

- `@tool(return_direct=True)`: speak the result as the answer and end the turn ("Switching to
  Qwen 3.6 27B.").
- `@tool(confirm=lambda **args: "Do it?")` with `confirm_tools=True` on the agent: ask before running.
- A `ctx` parameter receives the session, so tools can speak progress or ask questions.

The session adds its own tools to any agent that takes tools: speaking speed, new conversation,
status, end session. So "talk a bit faster" or "let's start over" work with every agent.
`OllamaAgent` also adds model tools: list models, switch model, thinking on or off.

## LangGraph orchestrators

`LangGraphAgent` maps a graph's stream onto the voice:

| In the graph | What you hear and see |
|---|---|
| LLM tokens from the top-level graph | Spoken reply. Sub-agents stay quiet unless `speak_subgraphs=True`. |
| `get_stream_writer()("Agent B is done.")` | Spoken progress update, from any node at any depth |
| Tool calls and results | Shown on screen |
| `interrupt("Deploy to production?")` | Asked aloud; your spoken answer resumes the graph |

```python
from langgraph.checkpoint.memory import InMemorySaver
from voicekit import VoiceSession, control_tools
from voicekit.agents import LangGraphAgent, to_langchain_tools

tools = my_tools + to_langchain_tools(control_tools())    # "talk faster", "goodbye"... inside the graph
graph = build_my_graph(tools).compile(checkpointer=InMemorySaver())
VoiceSession(LangGraphAgent(graph)).run()
```

`voice_context()` gives nodes and tools the live session (`voice_context().say("...")`).
[examples/voice_orchestrator.py](examples/voice_orchestrator.py) is a complete example: a
supervisor with research, test and deploy agents, spoken progress, and deployment approval by voice.

```bash
python examples/voice_orchestrator.py          # talk to it
python examples/voice_orchestrator.py --text   # type instead: no mic or speakers needed
```

## Voice assistant app

```bash
python voice_assistant.py                               # Ollama at 10.0.0.100, qwen3.5:4b
python voice_assistant.py --model qwen3.6:27b --voice am_michael
python voice_assistant.py --backend claude              # Claude Opus 5.5
python voice_assistant.py --backend openai --host http://server:8000
python voice_assistant.py --interruptible --output 7    # headphones: talk over it to interrupt
python voice_assistant.py --list-devices
```

Ask in your own words: "show me the available models", "think more carefully", "go with the bigger
Qwen", "talk a little quicker". Changes are announced as they happen ("Switching to Qwen 3.6 27B.").
Use `--confirm-tools` to be asked first, or `--no-tools` to turn tools off.

### Exact voice commands

These run instantly without involving the model. The whole utterance must be the command, so
"how do I stop thinking about work?" still goes to the model.

| Say | Effect |
|---|---|
| "goodbye" / "dismissed" | Quit |
| "new conversation" | Clear the history |
| "speak faster" / "speak slower" / "normal speed" | Change the speaking rate |
| "status" | Describe the current setup |
| "help" | List the commands |
| "set think" / "set no think" | Thinking on or off (`OllamaAgent`) |
| "switch model to Qwen 3.5" / "list models" | Change model or list them (`OllamaAgent`) |

`--command-prefix computer` requires a wake word ("Computer, speak faster"). Agents can add their
own commands through `register_commands(router)`.

## Session options

`SessionConfig` controls: `interruptible` (talk over the assistant; use headphones),
`wait_cue` (said when the agent is slow to start, default "Let me think about that." after
2.5 s), `speak_status`, `show_thinking`, `control_tools`, `greeting`, the exit and reset phrases,
and more. Each option is documented in its docstring. For a custom UI, pass
`on_event=lambda kind, text: ...` instead of the default console transcript.

## Tests

```bash
python -m unittest                                   # fast: fake devices, models and servers
python tests/test_agents.py --server                 # OllamaAgent + a real LangGraph agent on your server
python tests/test_agents.py --claude                 # ClaudeAgent against the Claude API (costs money)
python tests/test_speech_recognizer.py --gpu --mic   # real Whisper and live microphone
python tests/test_text_to_speech.py --gpu --audio    # real Kokoro, then plays speech
python tests/test_llm_client.py --server             # LLM client against your server
python tests/test_voice_assistant.py --e2e           # full loopback: speak, transcribe, answer, speak
```
