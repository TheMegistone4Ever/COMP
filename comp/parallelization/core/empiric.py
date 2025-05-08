from math import log
from typing import Tuple


def empiric(size: Tuple[int, int]) -> float:
    """
    Calculate an empiric score based on the size of a linear programming problem.

    The score is computed using a formula involving the dimensions (m, n)
    provided in the size of tuple.
    The values of m and n are clamped to a minimum of 1.

    :param size: A tuple (m, n) representing the dimensions of the problem,
                 where m is the number of constraints and n is the number of variables.
    :return: A float score calculated based on the empiric formula.
    """

    return abs(.63 * (m := max(1, size[0])) ** 2.96 * (n := max(1, size[1])) ** .02 * log(n) ** 1.62
               + 4.04 * m ** -4.11 * n ** 2.92)
