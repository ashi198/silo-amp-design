import unittest

from silo_amp.search import SearchConfig, bounded_search


class SearchTests(unittest.TestCase):
    def expand(self, state):
        if len(state) >= 2:
            return []
        return [(0, state + "a", len(state) == 1, 0.1), (1, state + "b", len(state) == 1, 0.2)]

    def score(self, state):
        return float(state.count("b") * 2 + state.count("a"))

    def test_deterministic_search_is_bounded_ordered_and_terminates(self):
        leaves = bounded_search(
            [""], expand=self.expand, score=self.score,
            config=SearchConfig(mode="deterministic", beam_width=2, max_expansions=4),
        )
        self.assertEqual(len(leaves), 2)
        self.assertEqual([leaf.actions for leaf in leaves], [(1, 0), (1, 1)])

    def test_same_stochastic_seed_repeats_and_different_seed_can_change(self):
        kwargs = dict(roots=[""], expand=self.expand, score=self.score,
                      config=SearchConfig(mode="stochastic", beam_width=1, max_expansions=4))
        first = bounded_search(**kwargs)
        again = bounded_search(**kwargs)
        other = bounded_search(**{**kwargs, "config": SearchConfig(mode="stochastic", beam_width=1, max_expansions=4, seed=1)})
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_expansion_budget_is_hard(self):
        calls = []
        def expand(state):
            calls.append(state)
            return self.expand(state)
        bounded_search([""], expand=expand, score=self.score,
                       config=SearchConfig(max_expansions=1))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
