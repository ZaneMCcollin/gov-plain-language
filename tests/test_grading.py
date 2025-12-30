import unittest
import grading as g

class TestReadability(unittest.TestCase):
    def test_simple_text_is_grade8_or_below(self):
        text = "We will send your letter today. It will explain what to do next."
        self.assertLessEqual(g.flesch_kincaid(text), 8.0)

    def test_complex_harder_than_simple(self):
        simple = "We will send the form today."
        complex_ = ("Pursuant to the aforementioned regulatory framework, the department will "
                    "expeditiously disseminate the requisite documentation at its earliest convenience.")
        self.assertGreater(g.flesch_kincaid(complex_), g.flesch_kincaid(simple))

    def test_sentence_split(self):
        s = "Hello world. This is a test! New line\nAnother sentence?"
        parts = g.split_sentences(s)
        self.assertGreaterEqual(len(parts), 3)

if __name__ == "__main__":
    unittest.main()
