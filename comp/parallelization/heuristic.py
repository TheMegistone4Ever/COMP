from math import log
from typing import List, Tuple


def _heuristic(size: Tuple[int, int]) -> int:
    """
    Heuristic function to calculate a score based on the size of a tuple.

    :param size: Tuple representing the size of linear programming problem (m, n).
    :return: An integer score based on the heuristic calculation.
    """

    return abs(int(.63 * (m := size[0]) ** 2.96 * (n := size[1]) ** .02 * log(n) ** 1.62
                   + 4.04 * m ** -4.11 * n ** 2.92))


def get_order(sizes: List[Tuple[int, int]], threads: int) -> List[List[int]]:
    """
    Assigns the given sizes to the specified number of threads in a balanced manner.

    :param sizes: List of tuples representing the sizes to be assigned.
    :param threads: Number of threads to distribute the sizes across.
    :return: List of lists, where each inner list contains the indices of the sizes assigned to that thread.
    """

    sizes_heuristic = [_heuristic(size) for size in sizes]
    thread_sizes = [0] * threads
    thread_assignments = [[] for _ in range(threads)]
    for i in sorted(range(len(sizes)), key=lambda _: sizes_heuristic[_], reverse=True):
        min_thread = thread_sizes.index(min(thread_sizes))
        thread_assignments[min_thread].append(i)
        thread_sizes[min_thread] += sizes_heuristic[i]
    return thread_assignments


if __name__ == "__main__":
    in_threads, problems_count = 3, 5
    in_sizes = [(i, i + 1) for i in range(1, problems_count * 2 + 1, 2)]
    print(f"Input sizes: {in_sizes}\nInput threads: {in_threads}\nOrder: {get_order(in_sizes, in_threads)}")
