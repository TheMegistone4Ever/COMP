from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Callable, TypeVar, Optional

from comp.utils import assert_positive, assert_non_negative

T = TypeVar("T")


def run_task_group(tasks: List[Callable[[], T]], num_tasks: int, task_indices: List[int]):
    """
    Run a group of tasks in parallel.

    :param tasks: A list of callable tasks to be executed
    :param num_tasks: The total number of tasks
    :param task_indices: A list of indices that specify which tasks to run
    :return: A dictionary mapping task indices to their results
    """

    group_results = dict()
    for index in task_indices:
        if 0 <= index < num_tasks:
            try:
                group_results[index] = tasks[index]()
            except (Exception,):
                group_results[index] = None
    return group_results


class ParallelExecutor:
    def __init__(self, order: List[List[int]], min_threshold: int, num_threads: int):
        self.order = order
        self.min_threshold = min_threshold
        self.num_threads = num_threads

        self.validate_input()

    def execute(self, tasks: List[Callable[[], T]]) -> List[Optional[T]]:
        """
        Execute a list of tasks in parallel using a process pool.

        :param tasks: A list of callable tasks to be executed
        :return: A list of results from the tasks, in the same order as the input tasks
        """

        if (num_tasks := len(tasks)) == 0:
            return list()

        # Do not parallelize if the number of tasks is lower than the threshold
        if num_tasks < self.min_threshold or self.num_threads <= 1:
            return list(map(lambda task: task(), tasks))

        all_results_map = dict()
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            run_task = partial(run_task_group, tasks, num_tasks)
            future_to_group_results = [
                pool.submit(run_task, group)  # type: ignore
                for group in self.order if group
            ]

            for future in future_to_group_results:
                all_results_map.update(future.result())

        results: List[Optional[T]] = [None] * num_tasks
        for i in range(num_tasks):
            if i in all_results_map:
                results[i] = all_results_map.get(i)
            else:
                # This task was not in any scheduled group, run sequentially as a fallback.
                # This might happen if "get_order" doesn't cover all indices.
                # Or if the schedule is faulty.
                # For safety, execute tasks not covered by the schedule.
                try:
                    results[i] = tasks[i]()
                except (Exception,):
                    results[i] = None

        return results

    def validate_input(self) -> None:
        """Validate the input parameters for the parallel executor."""

        assert_positive(self.min_threshold, "min_threshold")
        assert_positive(self.num_threads, "num_threads")
        assert_positive(len(self.order), "len(order)")
        for thread in self.order:
            for task_id in thread:
                assert_non_negative(task_id, f"[{task_id=},{thread=}] in order")
