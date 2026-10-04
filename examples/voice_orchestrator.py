"""
Example: steer a multi-agent orchestrator (LangGraph) by voice.

A supervisor model listens to you and delegates to three agents. They
report progress aloud while they work, and deploying to an environment
waits for your spoken approval (LangGraph interrupt).

    python examples/voice_orchestrator.py             # talk to it
    python examples/voice_orchestrator.py --text      # type instead (no mic/speaker)

Try: "research what's new in Python 3.14", "run the backend tests",
"deploy to staging", "talk a bit faster", "status", "goodbye".

Requires: pip install langgraph langchain-ollama
"""

import argparse
import os
import sys
import time
import warnings
from typing import Literal

# Run from anywhere: make the project root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from voicekit import LLMConfig, SessionConfig, VoiceSession, classify_confirmation, control_tools
from voicekit.agents import LangGraphAgent, to_langchain_tools
from voicekit.console import KeyboardInput, SilentSpeaker


SUPERVISOR_PROMPT = """\
You are the voice of an orchestrator that runs a team of agents for the user.
Delegate work with your tools: research for questions that need looking into,
run_tests for testing, deploy for deployments. Report results in one or two
short spoken sentences, without markdown or lists. Answer simple questions
yourself. The tools already announce their progress, so don't repeat it.
"""


def build_graph(model_name, base_url):

    model = ChatOllama(model=model_name, base_url=base_url, reasoning=False)

    # A sub-agent: its own LLM calls stay quiet; only its result and the
    # progress updates reach the user.
    researcher = create_react_agent(
        model, [], prompt="You are a research agent. Answer in three sentences."
    )

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

    tools = [research, run_tests, deploy] + to_langchain_tools(control_tools())

    return create_react_agent(
        model, tools, prompt=SUPERVISOR_PROMPT, checkpointer=InMemorySaver()
    )


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", default=LLMConfig.model)
    parser.add_argument("--host", default=LLMConfig.base_url)
    parser.add_argument("--text", action="store_true",
                        help="type and read instead of speaking and listening")
    args = parser.parse_args(argv)

    agent = LangGraphAgent(
        build_graph(args.model, args.host),
        description="I'm the orchestrator, with a research agent, a test agent and a deploy agent.",
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
