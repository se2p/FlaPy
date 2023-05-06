import numpy as np

"""
SFFL formulas.
"""


def save_div(x, y) -> float:
    """Division, which returns zero if the divisor is zero instead of a ZeroDivisionError.

    :x: dividend
    :y: divisor
    :returns: quotient

    """
    if y == 0:
        if x > 0:
            return np.inf
        elif x < 0:
            return -np.inf
        else:
            return 0
    else:
        return x / y


def tarantula(
    flaky_covered: int,
    stable_covered: int,
    totalflaky: int,
    totalstable: int,
) -> float:
    """
    The formula to calculate the suspiciousness score according to the Tarantula approach.

    :param flaky_covered: The number of flaky tests covering the current statement.
    :param stable_covered: The number of stable tests covering the current statement.
    :param totalflaky: The total number of flaky tests in the testsuite.
    :param totalstable: The total number of stable tests in the testsuite.
    :return: The suspiciousness score of the current statement.
    """
    result = save_div(
        save_div(flaky_covered, totalflaky),
        (save_div(flaky_covered, totalflaky) + save_div(stable_covered, totalstable)),
    )
    return result


def ochiai(flaky_covered: int, stable_covered: int, totalflaky: int):
    """
    The formula to calculate the suspiciousness score according to the Ochiai approach.

    :param flaky_covered: The number of flaky tests covering the current statement.
    :param stable_covered: The number of stable tests covering the current statement.
    :param totalflaky: The total number of flaky tests in the testsuite.
    :return: The suspiciousness score of the current statement.
    """
    return save_div(flaky_covered, (np.sqrt(totalflaky * (flaky_covered + stable_covered))))


def dStar(
    flaky_covered: int,
    stable_covered: int,
    totalflaky: int,
    exponent: float = 2,
):
    """
    The formula to calculate the suspiciousness score according to the DStar approach.

    :param flaky_covered: The number of flaky tests covering the current statement.
    :param stable_covered:  The number of stable tests covering the current statement.
    :param totalflaky: The total number of flaky tests in the testsuite.
    :param exponent: The exponent used in this formula. In the original paper this was set to 1.
    :return: The suspiciousness score of the current statement.
    """
    return save_div(flaky_covered**exponent, (stable_covered + (totalflaky - flaky_covered)))


def barinel(flaky_covered: int, stable_covered: int, totalflaky: int) -> float:
    return 1 - save_div(stable_covered, stable_covered + flaky_covered)


def op2(flaky_covered: int, stable_covered: int, totalstable: int) -> float:
    return flaky_covered - save_div(stable_covered, totalstable + 1)
