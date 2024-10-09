import unittest

from sample_efficiency_evaluation.fact_matcher import FactMatcher

class FactMatcherTest(unittest.TestCase):

    def setUp(self) -> None:
        self.matcher = FactMatcher()