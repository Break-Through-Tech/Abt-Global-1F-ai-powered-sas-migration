import unittest
from types import SimpleNamespace

from correction_agent.agent import correct, validate_feedback, validate_instructions


class FakeModel:
    def __init__(self, text, reason="STOP"):
        self.text = text
        self.reason = reason
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(text=self.text, response_metadata={"finish_reason": self.reason})


class CorrectionInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.feedback = {"program_id": "program1", "next_state": "TRIAGED",
                         "fix_request": "OUTCOME_MORTALITY: columns differ"}

    def test_accepts_saved_validator_feedback(self):
        self.assertIs(validate_feedback("program1", self.feedback), self.feedback)

    def test_accepts_validator_model_object(self):
        class Result:
            def model_dump(inner):
                return self.feedback
        self.assertEqual(validate_feedback("program1", Result()), self.feedback)

    def test_rejects_wrong_program_passed_and_empty_feedback(self):
        for changed in ({"program_id": "program0"}, {"next_state": "PASSED"}, {"fix_request": None}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_feedback("program1", self.feedback | changed)

    def test_rejects_code_and_empty_instructions(self):
        for reply in ("", "```python\nx = 1\n```", "import pandas as pd\nx = 1"):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                validate_instructions(reply)

    def test_builds_translator_handoff_without_generating_code(self):
        plan = "Compare the output column names and order against the SAS KEEP statement."
        model = FakeModel(plan)
        original = "# failed script\nprint('example')\n"
        result = correct("program1", original, self.feedback, llm=model,
                         history=[{"program_id": "program1", "rank": [-2, 1, 1, 0],
                                   "stderr": "PRIVATE_LOG_SENTINEL"}])
        self.assertEqual(result.instructions, plan)
        self.assertEqual(result.program_id, "program1")
        self.assertFalse(hasattr(result, "code"))
        self.assertIn(plan, result.repair_prompt)
        self.assertIn(original, result.repair_prompt)
        self.assertIn(self.feedback["fix_request"], result.repair_prompt)
        self.assertIn("Translation Agent", result.repair_prompt)
        self.assertNotIn("Diagnose the failed", result.repair_prompt)
        self.assertNotIn("PRIVATE_LOG_SENTINEL", model.messages[1].content)
        self.assertIn("Never write replacement code", model.messages[0].content)

    def test_rejects_truncated_model_response(self):
        with self.assertRaises(RuntimeError):
            correct("program1", "# failed", self.feedback, llm=FakeModel("Partial plan", "MAX_TOKENS"))
