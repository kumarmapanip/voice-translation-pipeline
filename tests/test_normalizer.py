import unittest

from pipeline.normalizer import normalize


class TestNormalizer(unittest.TestCase):
    def test_abbreviations_expanded(self):
        self.assertIn("estimated time of arrival", normalize("The ETA is 5 PM"))
        self.assertIn("as soon as possible", normalize("Send it ASAP"))
        self.assertIn("to be decided", normalize("The date is TBD"))

    def test_named_entities_untouched(self):
        self.assertEqual(normalize("Google Meet"), "Google Meet")
        self.assertEqual(normalize("Asterisk is open source"), "Asterisk is open source")

    def test_time_tokens_normalized(self):
        self.assertEqual(normalize("5pm"), "5 PM")
        self.assertEqual(normalize("5PM"), "5 PM")
        self.assertEqual(normalize("10:30am"), "10:30 AM")

    def test_empty_and_whitespace(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize("   "), "")


if __name__ == "__main__":
    unittest.main()
