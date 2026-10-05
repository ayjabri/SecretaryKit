"""
Example: steer a multi-agent orchestrator (LangGraph) by voice.

A supervisor model listens to you and delegates to four agents. They
report progress aloud while they work, and deploying to an environment
waits for your spoken approval (LangGraph interrupt). The data analyst
answers questions about CSV files in --data-dir with its own CSV tools.

    python examples/voice_orchestrator.py             # talk to it
    python examples/voice_orchestrator.py --text      # type instead (no mic/speaker)
    python examples/voice_orchestrator.py --data-dir path/to/csvs

Try: "research what's new in Python 3.14", "run the backend tests",
"deploy to staging", "what's the average price in sales.csv by region",
"what's 23 times 47", "what can you do?", "talk a bit faster", "status", "goodbye".

Requires: pip install langchain langgraph langchain-ollama pandas
"""

import argparse
import inspect
import os
import sys
import time
import warnings
from typing import Literal, Optional

import pandas as pd

# Run from anywhere: make the project root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from voicekit import LLMConfig, SessionConfig, VoiceSession, classify_confirmation, control_tools
from voicekit.agents import LangGraphAgent, to_langchain_tools
from voicekit.console import KeyboardInput, SilentSpeaker


SUPERVISOR_PROMPT = """\
You are the voice of an orchestrator that runs a team of agents for the user.
Delegate work with your tools: research for questions that need looking into,
run_tests for testing, deploy for deployments, ask_data_analyst for any
question about data in a CSV file (never guess the numbers), and calculator
for arithmetic. When asked what you can do, call list_capabilities and read
its summary out as it is. Report other results in one or two short spoken
sentences, without markdown or lists. Answer simple questions yourself. The tools
already announce their progress, so don't repeat it.

CSV files you can ask the analyst about: {csv_files}
"""

ANALYST_PROMPT = """\
You are a data analyst. Answer the question using your CSV tools.
Always call describe_csv first to learn the columns, then use aggregate_csv.
Reply with a short factual answer that includes the key numbers. No commentary.
"""

CAPABILITIES_PROMPT = """\
Below are the tools of a voice orchestrator, with their source code. Explain
to the user, in the first person, what you can do. Group related tools, give
one short example of what the user could say for each group, and mention any
safeguards (such as asking for approval). This will be spoken aloud: plain
sentences only, no markdown, lists, code or tool names, under 150 words.
"""


# ---------- Plain tool functions ----------

def calculator(expression: str) -> str:
    """Evaluate a basic math expression, for example '23 * 47'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def make_csv_tools(data_dir):
    """The analyst's CSV tools; relative paths resolve against data_dir."""

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(data_dir, path)

    @tool
    def describe_csv(path: str) -> str:
        """Summarize a CSV file: rows, column names and types, first rows, numeric stats."""
        path = resolve(path)
        if not os.path.exists(path):
            return f"Error: file not found at '{path}'"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return f"Error: {e}"
        parts = [f"{len(df)} rows, {len(df.columns)} columns"]
        parts.append("Columns: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()))
        parts.append("First 3 rows:\n" + df.head(3).to_string())
        numeric = df.select_dtypes("number")
        if not numeric.empty:
            parts.append("Numeric summary:\n" + numeric.describe().round(3).to_string())
        return "\n\n".join(parts)[:5000]

    @tool
    def aggregate_csv(
        path: str,
        column: str,
        stat: Literal["sum", "mean", "min", "max", "count", "median"],
        group_by: Optional[str] = None,
    ) -> str:
        """Compute a statistic over one CSV column, optionally grouped by another column."""
        path = resolve(path)
        if not os.path.exists(path):
            return f"Error: file not found at '{path}'"
        try:
            df = pd.read_csv(path)
            for c in [column] + ([group_by] if group_by else []):
                if c not in df.columns:
                    return f"Error: no column '{c}'. Columns are: {list(df.columns)}"
            if group_by:
                result = df.groupby(group_by)[column].agg(stat)
            else:
                result = getattr(df[column], stat)()
            return str(result)[:5000]
        except Exception as e:
            return f"Error: {e}"

    return [describe_csv, aggregate_csv]


def describe_tools(tools, helpers=None):
    """A text catalogue of tools (name, description, arguments, source code)
    for the model to explain. helpers maps a tool name to the tools of the
    sub-agent it delegates to."""
    helpers = helpers or {}
    sections = []
    for t in tools:
        lines = [f"## {t.name}", t.description, f"Arguments: {t.args or 'none'}"]
        func = getattr(t, "func", None)
        try:
            lines.append("Source:\n" + inspect.getsource(func))
        except (TypeError, OSError):
            pass  # built-in or wrapped tools: the description has to do
        if t.name in helpers:
            lines.append("Delegates to a sub-agent with these tools:")
            lines.append(describe_tools(helpers[t.name]))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_graph(model_name, base_url, data_dir="."):

    model = ChatOllama(model=model_name, base_url=base_url, reasoning=False)

    # Sub-agents: their own LLM calls stay quiet; only their results and the
    # progress updates reach the user. Each call starts a fresh history, so
    # their work never touches the supervisor's conversation.
    researcher = create_agent(
        model, [], system_prompt="You are a research agent. Answer in three sentences."
    )
    csv_tools = make_csv_tools(data_dir)
    analyst = create_agent(model, csv_tools, system_prompt=ANALYST_PROMPT)

    csv_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))

    @tool
    def research(topic: str) -> str:
        """Ask the research agent to look into a topic and summarise it."""
        progress = get_stream_writer()
        progress(f"The research agent is looking into {topic}.")
        result = researcher.invoke({"messages": [("user", topic)]})
        progress("The research agent has finished.")
        return result["messages"][-1].content

    @tool
    def run_tests(suite: str) -> str:
        """Run a test suite (for example backend, frontend or api) and report the result."""
        progress = get_stream_writer()
        progress(f"The test agent is running the {suite} tests.")
        time.sleep(1.0)  # stand-in for real work
        progress("Halfway through.")
        time.sleep(1.0)
        return f"All 128 {suite} tests passed in 2 seconds."

    @tool
    def deploy(environment: Literal["staging", "production"]) -> str:
        """Deploy the current build to an environment. Asks the user to approve first."""
        answer = interrupt(f"Do you want me to deploy to {environment}?")

        if classify_confirmation(answer) is not True:
            return "The user did not approve, so nothing was deployed."

        progress = get_stream_writer()
        progress(f"The deploy agent is rolling out to {environment}.")
        time.sleep(1.5)
        return f"Deployed build 4.2.1 to {environment}. Health checks are green."

    @tool
    def ask_data_analyst(question: str, path: str) -> str:
        """Delegate a question about a CSV file to the data analyst agent. Returns a short factual answer."""
        progress = get_stream_writer()
        progress(f"The data analyst is looking at {os.path.basename(path)}.")
        result = analyst.invoke({"messages": [("user", f"File: {path}\n\nQuestion: {question}")]})
        progress("The data analyst has finished.")
        return result["messages"][-1].content

    summary = []  # cached after the first call

    @tool
    def list_capabilities() -> str:
        """List what the orchestrator can do, with example requests. Use when the user asks for your functions, tools or capabilities."""
        if not summary:
            progress = get_stream_writer()
            progress("Let me look at what I can do.")
            catalogue = describe_tools(
                [t for t in tools if t.name != "list_capabilities"],
                helpers={"ask_data_analyst": csv_tools},
            )
            reply = model.invoke([
                ("system", CAPABILITIES_PROMPT),
                ("user", f"CSV files available: {', '.join(csv_files) or 'none'}\n\n{catalogue}"),
            ])
            summary.append(reply.content)
        return summary[0]

    tools = [research, run_tests, deploy, ask_data_analyst, tool(calculator), list_capabilities]
    tools += to_langchain_tools(control_tools())

    prompt = SUPERVISOR_PROMPT.format(csv_files=", ".join(csv_files) or "none found")
    return create_agent(model, tools, system_prompt=prompt, checkpointer=InMemorySaver())


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", default=LLMConfig.model)
    parser.add_argument("--host", default=LLMConfig.base_url)
    parser.add_argument("--text", action="store_true",
                        help="type and read instead of speaking and listening")
    parser.add_argument("--data-dir", default=".",
                        help="folder of CSV files the data analyst can read")
    args = parser.parse_args(argv)

    agent = LangGraphAgent(
        build_graph(args.model, args.host, args.data_dir),
        description="I'm the orchestrator, with a research agent, a test agent, "
                    "a deploy agent and a data analyst.",
    )

    extras = {}
    if args.text:
        extras = {"recognizer": KeyboardInput(), "speaker": SilentSpeaker()}

    session = VoiceSession(
        agent,
        config=SessionConfig(greeting="Orchestrator ready. What should the team do?"),
        **extras,
    )

    try:
        session.run()
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        session.close()


if __name__ == "__main__":
    main()
