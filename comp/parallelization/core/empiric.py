from math import log
from typing import Tuple


def empiric(size: Tuple[int, int]) -> float:
    """
    Empiric function to calculate a score based on the size of a tuple.

    :param size: Tuple representing the size of linear programming problem (m, n).
    :return: An integer score based on the empiric calculation.
    """

    return abs(.63 * (m := max(1, size[0])) ** 2.96 * (n := max(1, size[1])) ** .02 * log(n) ** 1.62
               + 4.04 * m ** -4.11 * n ** 2.92)
