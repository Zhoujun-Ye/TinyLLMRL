import unittest

from reward import extract_answer, format_arithmetic_prompt, score_answer


class RewardTests(unittest.TestCase):
    def test_single_explicit_final_integer(self):
        for text, answer in (
            ("FINAL: 89", 89),
            ("Work: 67 + 22 = 89.\n\nFINAL: 89\n", 89),
            ("Wrong first guess: 88.\r\nFINAL: +89\r\n", 89),
            ("FINAL: -7", -7),
            ("FINAL: 0", 0),
        ):
            with self.subTest(text=text):
                self.assertEqual(score_answer(text, answer, ended=True), 1)

    def test_wrong_ambiguous_or_malformed(self):
        self.assertEqual(score_answer("FINAL: 88", 89, ended=True), 0)
        for text in (
            "", "89", "Maya had 89 beads.", "67 + 22 = 89",
            "The wrong answer is 89.", r"The answer is not \boxed{89}",
            "Answer: 88.\nAnswer: 89", "FINAL: 88\nFINAL: 89",
            "final : 89\nFINAL: 89", "FINAL: 89\nFINAL: 89",
            "Not FINAL: 89", "FINAL: 89 or 90", "FINAL: 89.0",
            "FINAL: 89 beads", "**FINAL: 89**", "FINAL: 8<|im_start|>9",
            "FINAL: 89\nActually, this isn't correct.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text, ended=True))
                self.assertEqual(score_answer(text, 89, ended=True), 0)

    def test_truncated_final_answer_gets_zero(self):
        self.assertEqual(score_answer("FINAL: 89", 89, ended=False), 0)
        self.assertEqual(score_answer("FINAL: 67 + 22 = 89", 89, ended=False), 0)

    def test_equation_must_be_unique_complete_and_valid(self):
        self.assertEqual(score_answer("FINAL: 67 + 22 = 89", 89, ended=True), 1)
        self.assertEqual(score_answer("FINAL: 95 - 79 = 16", 16, ended=True), 1)
        self.assertEqual(score_answer("FINAL: 67 + 21 = 88", 89, ended=True), 0)
        for text in (
            "FINAL: 67 + 21 = 89", "FINAL: 67 + 22 = 89 or 90",
            "FINAL: 67 + 22 = 89\nFINAL: 89",
            "FINAL: 67 + 22 = 89 = 90", "FINAL: 67 + 22.0 = 89",
            "FINAL: 67 + 22 = ", "FINAL: 89 * 1 = 89",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text, ended=True))

    def test_ground_truth_must_be_integer(self):
        for value in (True, None, "89", 89.0):
            with self.subTest(value=value), self.assertRaises(TypeError):
                score_answer("FINAL: 89", value, ended=True)

    def test_format_replaces_legacy_instruction(self):
        formatted = format_arithmetic_prompt("What is 7 + 8? Give only the final number.")
        self.assertTrue(formatted.startswith("What is 7 + 8?\n\n"))
        self.assertNotIn("Give only the final number.", formatted)
        self.assertIn("FINAL: <integer>", formatted)
