from itertools import chain, combinations
import pandas as pd
import numpy as np
import flapy.results_parser as rp


def powerset(ss):
    return chain(*map(lambda x: combinations(ss, x), range(len(ss) + 1)))


def test_decide_overall_verdict():
    all_comb = powerset(
        [
            rp.Verdict.PASS,
            rp.Verdict.FAIL,
            rp.Verdict.ERROR,
            rp.Verdict.SKIP,
            rp.Verdict.FLAKY,
            rp.Verdict.PARSE_ERROR,
            rp.Verdict.ZERO_RUNS,
        ]
    )
    for comb in all_comb:
        assert rp.Verdict.decide_overall_verdict(comb) != rp.Verdict.UNDECIDABLE


def test_results_dir_get_passed_failed():
    reference_pf = pd.read_csv(
        "test_output_reference/passed_failed_20220410_170615.csv",
        converters={
            "Passed_sameOrder": rp.eval_string_to_set,
            "Failed_sameOrder": rp.eval_string_to_set,
            "Error_sameOrder": rp.eval_string_to_set,
            "Skipped_sameOrder": rp.eval_string_to_set,
            "Verdicts_sameOrder": rp.eval_string_to_set,
            "Passed_randomOrder": rp.eval_string_to_set,
            "Failed_randomOrder": rp.eval_string_to_set,
            "Error_randomOrder": rp.eval_string_to_set,
            "Skipped_randomOrder": rp.eval_string_to_set,
            "Verdicts_randomOrder": rp.eval_string_to_set,
        },
    )
    reference_pf = reference_pf.sort_values(
        ["Iteration"] + rp.proj_cols + rp.test_cols
    ).reset_index(drop=True)

    pf = rp.ResultsDir(
        "test_resources/flapy-results-collection/flapy-results_20220410_170615"
    ).get_passed_failed(
        read_resultsDir_cache=False,
        write_resultsDir_cache=False,
        read_iteration_cache=False,
        write_iteration_cache=False,
    )
    pf[
        [
            "Passed_sameOrder",
            "Failed_sameOrder",
            "Error_sameOrder",
            "Skipped_sameOrder",
            "Verdicts_sameOrder",
            "Passed_randomOrder",
            "Failed_randomOrder",
            "Error_randomOrder",
            "Skipped_randomOrder",
            "Verdicts_randomOrder",
        ]
    ] = pf[
        [
            "Passed_sameOrder",
            "Failed_sameOrder",
            "Error_sameOrder",
            "Skipped_sameOrder",
            "Verdicts_sameOrder",
            "Passed_randomOrder",
            "Failed_randomOrder",
            "Error_randomOrder",
            "Skipped_randomOrder",
            "Verdicts_randomOrder",
        ]
    ].applymap(
        rp.eval_string_to_set
    )
    pf["Test_filename"] = pf["Test_filename"].replace("", np.NaN)
    pf["Test_parametrization"] = pf["Test_parametrization"].replace("", np.NaN)
    pf = pf.sort_values(["Iteration"] + rp.proj_cols + rp.test_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(reference_pf, pf, check_dtype=False)


def test_results_dir_collection_get_passed_failed():
    reference_pf = pd.read_csv(
        "test_output_reference/passed_failed.csv",
        converters={
            "Passed_sameOrder": rp.eval_string_to_set,
            "Failed_sameOrder": rp.eval_string_to_set,
            "Error_sameOrder": rp.eval_string_to_set,
            "Skipped_sameOrder": rp.eval_string_to_set,
            "Verdicts_sameOrder": rp.eval_string_to_set,
            "Passed_randomOrder": rp.eval_string_to_set,
            "Failed_randomOrder": rp.eval_string_to_set,
            "Error_randomOrder": rp.eval_string_to_set,
            "Skipped_randomOrder": rp.eval_string_to_set,
            "Verdicts_randomOrder": rp.eval_string_to_set,
        },
    )
    reference_pf = reference_pf.sort_values(
        ["Iteration"] + rp.proj_cols + rp.test_cols
    ).reset_index(drop=True)

    pf = rp.ResultsDirCollection("test_resources/flapy-results-collection").get_passed_failed(
        read_resultsDir_cache=False,
        write_resultsDir_cache=False,
        read_iteration_cache=False,
        write_iteration_cache=False,
    )
    pf[
        [
            "Passed_sameOrder",
            "Failed_sameOrder",
            "Error_sameOrder",
            "Skipped_sameOrder",
            "Verdicts_sameOrder",
            "Passed_randomOrder",
            "Failed_randomOrder",
            "Error_randomOrder",
            "Skipped_randomOrder",
            "Verdicts_randomOrder",
        ]
    ] = pf[
        [
            "Passed_sameOrder",
            "Failed_sameOrder",
            "Error_sameOrder",
            "Skipped_sameOrder",
            "Verdicts_sameOrder",
            "Passed_randomOrder",
            "Failed_randomOrder",
            "Error_randomOrder",
            "Skipped_randomOrder",
            "Verdicts_randomOrder",
        ]
    ].applymap(
        rp.eval_string_to_set
    )
    pf["Test_filename"] = pf["Test_filename"].replace("", np.NaN)
    pf["Test_parametrization"] = pf["Test_parametrization"].replace("", np.NaN)
    pf = pf.sort_values(["Iteration"] + rp.proj_cols + rp.test_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(reference_pf, pf, check_dtype=False)


def test_results_dir_get_tests_overview():
    reference_to = pd.read_csv(
        "test_output_reference/tests_overview_20220410_170615.csv",
        converters={
            "Verdicts_sameOrder": rp.eval_string_to_set,
            "Verdicts_randomOrder": rp.eval_string_to_set,
        },
    )
    reference_to = reference_to.sort_values(rp.proj_cols + rp.test_cols).reset_index(drop=True)

    to = (
        rp.ResultsDir("test_resources/flapy-results-collection/flapy-results_20220410_170615")
        .get_tests_overview(
            read_resultsDir_cache=False,
            write_resultsDir_cache=False,
            read_iteration_cache=False,
            write_iteration_cache=False,
        )
        ._df
    )
    to[["Verdicts_sameOrder", "Verdicts_randomOrder"]] = to[
        ["Verdicts_sameOrder", "Verdicts_randomOrder"]
    ].applymap(rp.eval_string_to_set)
    to["Test_filename"] = to["Test_filename"].replace("", np.NaN)
    to["Test_parametrization"] = to["Test_parametrization"].replace("", np.NaN)
    to = to.sort_values(rp.proj_cols + rp.test_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(reference_to, to, check_dtype=False)


def test_results_dir_collection_get_tests_overview():
    reference_to = pd.read_csv(
        "test_output_reference/tests_overview.csv",
        converters={
            "Verdicts_sameOrder": rp.eval_string_to_set,
            "Verdicts_randomOrder": rp.eval_string_to_set,
        },
    )
    reference_to = reference_to.sort_values(rp.proj_cols + rp.test_cols).reset_index(drop=True)

    to = (
        rp.ResultsDirCollection("test_resources/flapy-results-collection")
        .get_tests_overview(
            read_resultsDir_cache=False,
            write_resultsDir_cache=False,
            read_iteration_cache=False,
            write_iteration_cache=False,
        )
        ._df
    )
    to[["Verdicts_sameOrder", "Verdicts_randomOrder"]] = to[
        ["Verdicts_sameOrder", "Verdicts_randomOrder"]
    ].applymap(rp.eval_string_to_set)
    to["Test_filename"] = to["Test_filename"].replace("", np.NaN)
    to["Test_parametrization"] = to["Test_parametrization"].replace("", np.NaN)
    to = to.sort_values(rp.proj_cols + rp.test_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(reference_to, to, check_dtype=False)
