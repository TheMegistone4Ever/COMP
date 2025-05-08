from typing import List, Tuple

from comp.parallelization.core import Device, Operation, empiric


def get_multi_device_heuristic_order(threads: int, operations: List[Operation]) -> List[Device]:
    """
    Assign operations to a specified number of devices using the longest processing time (LPT) heuristic.

    Operations are first sorted by duration in descending order.
    Then, each operation is assigned to the device that currently has the minimum total processing time.
    After all assignments, operation times on each device are updated.

    :param threads: The number of threads (devices) to distribute the operations across.
    :param operations: A list of Operation objects to be scheduled.
    :return: A list of Device objects, each with its assigned operations and updated times.
    """

    operations.sort(key=lambda op: op.duration, reverse=True)

    ordered_devices = [Device() for _ in range(threads)]
    for operation in operations:
        min(ordered_devices, key=lambda dev: dev.end).operations.append(operation)

    for ordered_device in ordered_devices:
        ordered_device.update_operation_times()

    return ordered_devices


def make_permutation_1_1(lagged: Device, advanced: Device, average_deadline: float) -> bool:
    """
    Attempt to swap one operation from a lagged device with one from an advanced device.

    This function tries to improve load balance by moving a longer operation from the
    lagged device to the advanced device.
    A shorter operation from the advanced
    device to the lagged device, if the swap reduces the lagged device's end time
    sufficiently without making it finish before the average deadline by too much.

    :param lagged: The Device instance that is currently finishing later (lagging).
    :param advanced: The Device instance that is currently finishing earlier (advanced).
    :param average_deadline: The target average completion time for devices.
    :return: True if a beneficial swap was performed, False otherwise.
    """

    for i in range(len(lagged.operations)):
        for j in range(len(advanced.operations)):
            if (theta := lagged.operations[i].duration - advanced.operations[j].duration) <= 0:
                continue
            if theta <= (lagged.end - average_deadline):
                lagged.operations.insert(i, advanced.operations.pop(j))
                advanced.operations.insert(j, lagged.operations.pop(i))
                return True
    return False


def make_permutation_1_2(lagged: Device, advanced: Device, average_deadline: float) -> bool:
    """
    Attempt to swap one operation from a lagged device with two from an advanced device.

    This function tries to improve load balance by moving one operation from the lagged
    device and replacing it with two operations from the advanced device if this
    swap reduces the lagged device's end time sufficiently.

    :param lagged: The Device instance that is currently finishing later (lagging).
    :param advanced: The Device instance that is currently finishing earlier (advanced).
    :param average_deadline: The target average completion time for devices.
    :return: True if a beneficial swap was performed, False otherwise.
    """

    for i in range(len(lagged.operations)):
        for j in range(len(advanced.operations)):
            for k in range(j + 1, len(advanced.operations)):
                if (theta := lagged.operations[i].duration
                             - (advanced.operations[j].duration + advanced.operations[k].duration)) <= 0:
                    continue
                if theta <= (lagged.end - average_deadline):
                    lagged.operations.insert(i, advanced.operations.pop(j))
                    lagged.operations.insert(i + 1, advanced.operations.pop(k))
                    advanced.operations.insert(j, lagged.operations.pop(i))
                    return True
    return False


def make_permutation_2_1(lagged: Device, advanced: Device, average_deadline: float) -> bool:
    """
    Attempt to swap two operations from a lagged device with one from an advanced device.

    This function tries to improve load balance by moving two operations from the lagged
    device and replacing them with one operation from the advanced device if this
    swap reduces the lagged device's end time sufficiently.

    :param lagged: The Device instance that is currently finishing later (lagging).
    :param advanced: The Device instance that is currently finishing earlier (advanced).
    :param average_deadline: The target average completion time for devices.
    :return: True if a beneficial swap was performed, False otherwise.
    """

    for i in range(len(lagged.operations)):
        for j in range(i + 1, len(lagged.operations)):
            for k in range(len(advanced.operations)):
                if (theta := (lagged.operations[i].duration
                              + lagged.operations[j].duration) - advanced.operations[k].duration) <= 0:
                    continue
                if theta <= (lagged.end - average_deadline):
                    lagged.operations.insert(i, advanced.operations.pop(k))
                    advanced.operations.insert(k, lagged.operations.pop(i))
                    advanced.operations.insert(k + 1, lagged.operations.pop(j))
                    return True
    return False


def make_permutation_2_2(lagged: Device, advanced: Device, average_deadline: float) -> bool:
    """
    Attempt to swap two operations from a lagged device with two from an advanced device.

    This function tries to improve load balance by moving two operations from the lagged
    device and replacing them with two operations from the advanced device if this
    swap reduces the lagged device's end time sufficiently.

    :param lagged: The Device instance that is currently finishing later (lagging).
    :param advanced: The Device instance that is currently finishing earlier (advanced).
    :param average_deadline: The target average completion time for devices.
    :return: True if a beneficial swap was performed, False otherwise.
    """

    for i in range(len(lagged.operations)):
        for j in range(i + 1, len(lagged.operations)):
            for k in range(len(advanced.operations)):
                for l in range(k + 1, len(advanced.operations)):
                    if (theta := (lagged.operations[i].duration + lagged.operations[j].duration)
                                 - (advanced.operations[k].duration + advanced.operations[l].duration)) <= 0:
                        continue
                    if theta <= (lagged.end - average_deadline):
                        lagged.operations.insert(i, advanced.operations.pop(k))
                        lagged.operations.insert(i + 1, advanced.operations.pop(l))
                        advanced.operations.insert(k, lagged.operations.pop(i))
                        advanced.operations.insert(k + 1, lagged.operations.pop(j))
                        return True
    return False


def get_multi_device_order_A0(threads: int, operations: List[Operation], tolerance: float = 1e-9) -> List[Device]:
    """
    Assign operations to devices aiming for balanced end times using iterative permutation heuristics.

    Initially, operations are assigned using a heuristic (LPT). Then, iterative refinement
    is applied using permutation strategies (1-1, 1-2, 2-1, 2-2 swaps) between the most
    lagged device and advanced devices to balance a load until end times are within tolerance
    or no further improvements can be made.

    :param threads: The number of threads (devices) to distribute tasks across.
    :param operations: A list of Operation objects to be ordered.
    :param tolerance: The tolerance for checking if device end times are balanced.
    :return: A list of Device objects with their assigned operations, balanced as much as possible.
    """

    processed_devices = get_multi_device_heuristic_order(threads, operations)

    average_deadline = sum(op.duration for op in operations) / len(processed_devices)

    while True:
        if not processed_devices:
            break
        first_dev_end = processed_devices[0].end
        if all(abs(dev.end - first_dev_end) < tolerance for dev in processed_devices[1:]):
            break

        lagged_devices = [dev for dev in processed_devices if dev.end > average_deadline + tolerance]
        advanced_devices = [dev for dev in processed_devices if dev.end < average_deadline - tolerance]

        if not lagged_devices or not advanced_devices:
            break

        max_lagged_device = max(lagged_devices, key=lambda dev: dev.end - average_deadline)

        advanced_devices.sort(key=lambda dev: dev.end)

        found_permutation_this_iteration = False
        for advanced_device in advanced_devices:
            if not max_lagged_device.operations:
                continue
            if (make_permutation_1_1(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_1_2(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_2_1(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_2_2(max_lagged_device, advanced_device, average_deadline)):
                found_permutation_this_iteration = True
                break

        if not found_permutation_this_iteration:
            break

    for dev in processed_devices:
        dev.update_operation_times()

    return processed_devices


def get_order(sizes: List[Tuple[int, int]], threads: int) -> List[List[int]]:
    """
    Assign tasks, defined by their sizes, to threads for balanced parallel execution.

    This function converts task sizes into operation durations using an empiric function,
    then uses a multi-device scheduling algorithm (A0) to distribute these operations
    (tasks) across the specified number of threads.
    The result is a list of task indices assigned to each thread.

    :param sizes: A list of tuples, where each tuple (m, n) represents the characteristics
                  (e.g., constraints and variables) of a task.
    :param threads: The number of threads (devices) to distribute the tasks across.
    :return: A list of lists, where each inner list contains the original indices of the
             tasks assigned to the corresponding thread.
    """

    return [[operation.original_index for operation in device.operations]
            for device in get_multi_device_order_A0(
            threads, [Operation(empiric(size_tuple), i) for i, size_tuple in enumerate(sizes)])]


if __name__ == "__main__":
    """Example usage of the heuristic scheduling algorithm."""

    in_threads, problems_count = 3, 5
    in_sizes = [(i, i + 1) for i in range(1, problems_count * 2 + 1, 2)]
    print(f"Input sizes: {in_sizes}\nInput threads: {in_threads}\nOrder: {get_order(in_sizes, in_threads)}")
