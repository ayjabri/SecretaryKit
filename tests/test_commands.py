"""
Tests for voicekit/commands.py.

    python tests/test_commands.py
"""

import os
import sys
import unittest

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit.commands import NOT_HANDLED, CommandRouter, normalize_command


class NormalizeTests(unittest.TestCase):

    def test_normalize(self):
        self.assertEqual(normalize_command("  Set no-think. "), "set no think")
        self.assertEqual(normalize_command("Qwen 3.6, 27B!"), "qwen 3 6 27b")


class RouterTests(unittest.TestCase):

    def setUp(self):
        self.router = CommandRouter()
        self.router.add("on", r"(?:set )?think", lambda: "on", "think on")
        self.router.add_phrases("bye", ["Goodbye", "bye bye"], lambda: "bye")

    def test_whole_utterance_must_match(self):
        self.assertEqual(self.router.dispatch("Set think."), "on")
        self.assertIs(self.router.dispatch("I set think tanks aside"), NOT_HANDLED)

    def test_phrases_ignore_case_and_punctuation(self):
        self.assertEqual(self.router.dispatch("Bye-bye!"), "bye")
        self.assertEqual(self.router.dispatch("GOODBYE."), "bye")

    def test_unmatched_is_not_handled(self):
        self.assertIs(self.router.dispatch("What's the weather?"), NOT_HANDLED)
        self.assertFalse(NOT_HANDLED)

    def test_named_groups_become_arguments(self):
        self.router.add("model", r"use model (?P<name>.+)", lambda name: name)

        self.assertEqual(self.router.dispatch("Use model Qwen 3"), "qwen 3")

    def test_optional_groups_left_out_when_unmatched(self):
        self.router.add("speed", r"faster(?: by (?P<amount>\d+))?",
                        lambda amount="1": amount)

        self.assertEqual(self.router.dispatch("faster"), "1")
        self.assertEqual(self.router.dispatch("faster by 3"), "3")

    def test_handler_can_decline_and_fall_through(self):
        self.router.add("first", r"use (?P<x>.+)", lambda x: NOT_HANDLED)
        self.router.add("second", r"use (?P<x>.+)", lambda x: f"second {x}")

        self.assertEqual(self.router.dispatch("use it"), "second it")

    def test_first_registered_wins(self):
        self.router.add("other", r"set think", lambda: "other")

        self.assertEqual(self.router.dispatch("set think"), "on")

    def test_none_reply_is_handled(self):
        self.router.add("quiet", r"shh", lambda: None)

        self.assertIsNone(self.router.dispatch("Shh."))

    def test_descriptions(self):
        self.assertEqual(self.router.descriptions(), ["think on"])


class PrefixTests(unittest.TestCase):

    def setUp(self):
        self.router = CommandRouter(prefix="Computer")
        self.router.add("on", r"set think", lambda: "on")

    def test_requires_prefix(self):
        self.assertEqual(self.router.dispatch("Computer, set think."), "on")
        self.assertIs(self.router.dispatch("Set think."), NOT_HANDLED)

    def test_prefix_must_be_a_whole_word(self):
        self.assertIs(self.router.dispatch("Computers set think"), NOT_HANDLED)


if __name__ == "__main__":
    unittest.main()
