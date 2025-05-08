from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Callable, TypeVar, Optional, Dict

from comp.utils import assert_positive, assert_non_negative

T = TypeVar("T")


def run_task_group(tasks: List[Callable[[], T]], num_tasks: int, task_indices: List[int]) -> Dict[int, Optional[T]]:
    """
    Execute a specified subgroup of tasks from a larger list of tasks.

    This function iterates through a list of task indices, and for each valid index,
    it executes the corresponding task from the provided list of callable tasks.
    Results are stored in a dictionary mapping the task index to its result.
    If a task fails, its result is stored as None and an error message is printed.

    :param tasks: A list of all callable tasks available for execution.
    :param num_tasks: The total number of tasks in the `tasks` list.
    :param task_indices: A list of integer indices specifying which tasks from the `tasks` list to execute.
    :return: A dictionary mapping each executed task's original index to its result (or None if failed).
    """

    group_results = dict()
    for index in task_indices:
        if 0 <= index < num_tasks:
            try:
                group_results[index] = tasks[index]()
            except Exception as e:
                group_results[index] = None
                print(f"[PAR] Task {index} failed to execute: {e}")
    return group_results


class ParallelExecutor:
    def __init__(self, order: List[List[int]], min_threshold: int, num_threads: int) -> None:
        """
        Initialize the ParallelExecutor with scheduling and execution parameters.

        :param order: A list of lists,
        where each inner list contains task indices defining the execution order for a thread.
        :param min_threshold: The minimum number of tasks required to enable parallel execution.
        :param num_threads: The number of worker threads/processes to use for parallel execution.
        """

        self.order = order
        self.min_threshold = min_threshold
        self.num_threads = num_threads

        self.validate_input()

    def execute(self, tasks: List[Callable[[], T]]) -> List[Optional[T]]:
        """
        Execute a list of tasks, potentially in parallel based on configuration.

        If the number of tasks is below `min_threshold` or `num_threads` is 1 or less,
        tasks are run sequentially.
        Otherwise, tasks are distributed to a process pool according to the `self.order` schedule.
        Tasks not covered by the schedule are run sequentially as a fallback.

        :param tasks: A list of callable tasks to be executed.
        :return: A list containing the results of the tasks, in the same order as the input tasks.
                  Each result can be of type T or None if the task failed or was not executed.
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
                except Exception as exception:
                    results[i] = None
                    print(f"[SEQ] Task {i} failed to execute: {exception}")

        return results

    def validate_input(self) -> None:
        """
        Validate the input parameters provided during the executor's initialization.

        Checks if `min_threshold`, `num_threads`, and the length of `order` are positive.
        It also ensures all task IDs within the `order` schedule are non-negative.
        Raises an AssertionError if any validation fails.
        """

        assert_positive(self.min_threshold, "min_threshold")
        assert_positive(self.num_threads, "num_threads")
        assert_positive(len(self.order), "len(order)")
        for thread in self.order:
            for task_id in thread:
                assert_non_negative(task_id, f"[{task_id=},{thread=}] in order")
