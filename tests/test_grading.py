import unittest

import grading as g


class TestReadability(unittest.TestCase):
    def test_flesch_kincaid_simple_is_reasonable(self):
        text = "This is a simple sentence. It is easy to read."
        grade = g.flesch_kincaid(text)
        # Heuristic: only check it isn't extreme.
        self.assertGreaterEqual(grade, 0.0)
        self.assertLessEqual(grade, 12.0)

    def test_flesch_kincaid_complex_harder_than_simple(self):
        simple = "We will send the form today."
        complex_ = (
            "Pursuant to the aforementioned regulatory framework, the department will "
            "expeditiously disseminate the requisite documentation at its earliest convenience."
        )
        self.assertGreater(g.flesch_kincaid(complex_), g.flesch_kincaid(simple))

    def test_sentence_split(self):
        text = "One. Two! Three? Four"
        parts = g.split_sentences(text)
        self.assertEqual(len(parts), 4)


if __name__ == "__main__":
    unittest.main()
