import tempfile
import unittest
from pathlib import Path

from datasets.dataset import RLExample, read_examples, write_examples


class DatasetTests(unittest.TestCase):
    def test_roundtrip(self):
        examples = [RLExample("example", "Question", 5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_examples(path, examples)
            self.assertEqual(read_examples(path), examples)

    def test_duplicate_id_rejected(self):
        example = RLExample("duplicate", "A question")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_examples(path, [example, example])
            with self.assertRaises(ValueError):
                read_examples(path)

    def test_invalid_rows_report_location(self):
        for row in ('[]', '{}', '{"id":"x","prompt":""}',
                    '{"id":"x","prompt":1}', '{"id":"x","prompt":"q","metadata":[]}',
                    '{"id":"x","prompt":"q","response":"unexpected"}'):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.jsonl"
                path.write_text(row)
                with self.assertRaisesRegex(ValueError, r"bad.jsonl:1:"):
                    read_examples(path)
