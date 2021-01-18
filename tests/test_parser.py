from itertools import chain, combinations
from flapy.results_parser import Verdict


def powerset(ss):
    return chain(*map(lambda x: combinations(ss, x), range(len(ss) + 1)))


def test_decide_overall_verdict():
    all_comb = powerset(
        [
            Verdict.PASS,
            Verdict.FAIL,
            Verdict.ERROR,
            Verdict.SKIP,
            Verdict.FLAKY,
            Verdict.PARSE_ERROR,
            Verdict.ZERO_RUNS,
        ]
    )
    for comb in all_comb:
        assert Verdict.decide_overall_verdict(comb) != Verdict.UNDECIDABLE
