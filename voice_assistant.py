"""
Hands-free voice assistant built on voicekit.

    microphone -> Whisper -> agent (Ollama / OpenAI-compatible / Claude)
               -> Kokoro -> speakers

Usage
-----
    python voice_assistant.py                                  # Ollama, qwen3.5:4b
    python voice_assistant.py --model qwen3.6:27b --voice am_michael
    python voice_assistant.py --backend claude                 # Claude Opus 5.5
    python voice_assistant.py --backend openai --host http://10.0.0.100:8000
    python voice_assistant.py --interruptible                  # "hang on" stops it
    python voice_assistant.py --interruptible --barge-in speech  # headphones
    python voice_assistant.py --list-devices

Ask naturally ("show me the available models", "think harder", "talk a bit
quicker", "let's start over"): the model decides when to call a tool, and
changes are announced as they happen. Exact commands such as "goodbye",
"status", "speak faster" and "set think" also work instantly.

To talk to your own agents instead (LangGraph, LangChain, any function),
see examples/ and the README.
"""

import argparse
import logging
import sys
import warnings

from voicekit import (
    AgentError,
    ContinuousSpeechRecognizer,
    KokoroSpeaker,
    LLMConfig,
    SessionConfig,
    SpeechRecognizerConfig,
    TextToSpeechConfig,
    VoiceSession,
)


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Talk to an LLM with your voice.",
    )

    llm = parser.add_argument_group("LLM")
    llm.add_argument("--backend", choices=("ollama", "openai", "claude"),
                     default="ollama",
                     help="where the model runs (default: %(default)s)")
    llm.add_argument("--host", default=LLMConfig.base_url,
                     help="Ollama / OpenAI-compatible server URL "
                          "(default: %(default)s)")
    llm.add_argument("--model", default=None,
                     help=f"model name (default: {LLMConfig.model}, "
                          "or claude-opus-5-5 for --backend claude)")
    llm.add_argument("--api-key", default=None,
                     help="bearer token for an OpenAI-compatible server")
    llm.add_argument("--think", action="store_true",
                     help="let reasoning models think first (slower)")
    llm.add_argument("--effort", default="low",
                     choices=("low", "medium", "high", "xhigh", "max"),
                     help="Claude effort level (default: %(default)s)")
    llm.add_argument("--system-prompt-file", default=None,
                     help="file with a custom system prompt")
    llm.add_argument("--no-tools", action="store_true",
                     help="don't let the model call tools")
    llm.add_argument("--confirm-tools", action="store_true",
                     help="ask before a tool changes a setting")

    speech = parser.add_argument_group("speech")
    speech.add_argument("--mic", default=None,
                        help="microphone ID or part of its name")
    speech.add_argument("--whisper-model", default="turbo",
                        help="Whisper model (default: %(default)s)")
    speech.add_argument("--language", default="en",
                        help="spoken language code (default: %(default)s)")
    speech.add_argument("--vocabulary", default=None,
                        help="names/jargon to help recognition, "
                             "e.g. 'Kubernetes, Ollama, Qwen'")
    speech.add_argument("--pause", type=float, default=1.5,
                        help="seconds of silence that end your turn "
                             "(default: %(default)s)")

    voice = parser.add_argument_group("voice")
    voice.add_argument("--voice", default="af_heart",
                       help="Kokoro voice (default: %(default)s)")
    voice.add_argument("--speed", type=float, default=1.0,
                       help="speaking rate (default: %(default)s)")
    voice.add_argument("--output", default=None,
                       help="speaker ID or part of its name")

    parser.add_argument("--interruptible", action="store_true",
                        help="say 'stop' or 'hang on' to cut the assistant off")
    parser.add_argument("--barge-in", choices=("phrase", "speech"),
                        default="phrase",
                        help="with --interruptible: 'phrase' stops only on a "
                             "stop phrase (works with speakers); 'speech' "
                             "stops instantly on any speech (needs "
                             "headphones) (default: %(default)s)")
    parser.add_argument("--echo-cancellation", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="remove the assistant's voice from the microphone "
                             "(default: on with --interruptible if livekit is "
                             "installed)")
    parser.add_argument("--command-prefix", default=None,
                        help="wake word required before exact voice "
                             "commands, e.g. 'computer'")
    parser.add_argument("--list-devices", action="store_true",
                        help="list microphones and speakers, then exit")
    parser.add_argument("--debug", action="store_true",
                        help="verbose logging")

    return parser.parse_args(argv)


def build_agent(args):
    """The agent for the chosen backend."""

    system_prompt = LLMConfig.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read()

    if args.backend == "claude":
        from voicekit.agents import ClaudeAgent

        # With --no-tools the session adds no control tools either, so
        # Claude is offered none.
        return ClaudeAgent(
            model=args.model or "claude-opus-5-5",
            system_prompt=system_prompt,
            effort=args.effort,
            confirm_tools=args.confirm_tools,
        )

    from voicekit.agents import OllamaAgent

    return OllamaAgent(
        LLMConfig(
            base_url=args.host,
            api=args.backend,
            model=args.model or LLMConfig.model,
            api_key=args.api_key,
            think=args.think,
            system_prompt=system_prompt,
        ),
        confirm_tools=args.confirm_tools,
        use_tools=not args.no_tools,
    )


def _device(value):
    return int(value) if value is not None and value.isdigit() else value


def main(argv=None):

    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.debug:
        warnings.filterwarnings("ignore")

    if args.list_devices:
        ContinuousSpeechRecognizer.print_microphones()
        print("\nSpeakers:\n")
        for device in KokoroSpeaker.list_output_devices():
            print(f'{device["id"]:>3}  {device["name"]}')
        return 0

    recognizer = ContinuousSpeechRecognizer(SpeechRecognizerConfig(
        mic_device=_device(args.mic),
        model_name=args.whisper_model,
        language=args.language,
        hotwords=args.vocabulary,
        initial_prompt=args.vocabulary,
        prompt_end_ms=int(args.pause * 1000),
    ))

    speaker = KokoroSpeaker(TextToSpeechConfig(
        voice=args.voice,
        lang_code=args.voice[0],
        speed=args.speed,
        output_device=_device(args.output),
    ))

    session = VoiceSession(
        build_agent(args),
        recognizer,
        speaker,
        SessionConfig(
            interruptible=args.interruptible,
            barge_in=args.barge_in,
            echo_cancellation=args.echo_cancellation,
            command_prefix=args.command_prefix,
            control_tools=not args.no_tools,
        ),
    )

    try:
        session.run()
    except KeyboardInterrupt:
        print("\n[stopped]")
    except AgentError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
