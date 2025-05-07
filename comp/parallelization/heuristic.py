from typing import List, Tuple

from comp.parallelization.core import Device, Operation, empiric


def get_multi_device_heuristic_order(threads: int, operations: List[Operation]) -> List[Device]:
    """
    Assigns operations to devices in a balanced manner using a heuristic scheduling algorithm.

    :param threads: Number of threads (devices) to distribute the tasks across.
    :param operations: List of operations to be ordered.
    :return: List of devices with their assigned operations.
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
    This function attempts to swap one operation from the lagged device with one operation from the advanced device.

    :param lagged: Device with operations that are lagging behind.
    :param advanced: Device with operations that are ahead of order.
    :param average_deadline: The average deadline for the operations.
    :return: True if a valid swap was made, False otherwise.
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
    This function attempts to swap one operation from the lagged device with two operations from the advanced device.

    :param lagged: Device with operations that are lagging behind.
    :param advanced: Device with operations that are ahead of order.
    :param average_deadline: The average deadline for the operations.
    :return: True if a valid swap was made, False otherwise.
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
    This function attempts to swap two operations from the lagged device with one operation from the advanced device.

    :param lagged: Device with operations that are lagging behind.
    :param advanced: Device with operations that are ahead of order.
    :param average_deadline: The average deadline for the operations.
    :return: True if a valid swap was made, False otherwise.
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
    This function attempts to swap two operations from the lagged device with two operations from the advanced device.

    :param lagged: Device with operations that are lagging behind.
    :param advanced: Device with operations that are ahead of order.
    :param average_deadline: The average deadline for the operations.
    :return: True if a valid swap was made, False otherwise.
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
    Assigns operations to devices in a balanced manner using a heuristic scheduling algorithm.

    :param threads: Number of threads (devices) to distribute the tasks across.
    :param operations: List of operations to be ordered.
    :param tolerance: Tolerance for checking if devices are balanced.
    :return: List of devices with their assigned operations.
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
    Assigns the given sizes to the specified number of threads in a balanced manner
    using the MPMM-like scheduling algorithm.

    :param sizes: List of tuples representing the characteristics of tasks to be assigned.
                  Each tuple (e.g., (m, n)) is converted to a duration via _heuristic.
    :param threads: Number of threads (devices) to distribute the tasks across.
    :return: List of lists, where each inner list contains the original indices
             of the tasks assigned to that thread.
    """

    return [[operation.original_index for operation in device.operations]
            for device in get_multi_device_order_A0(
            threads, [Operation(empiric(size_tuple), i) for i, size_tuple in enumerate(sizes)])]


if __name__ == "__main__":
    """Example usage of the heuristic scheduling algorithm."""

    in_threads, problems_count = 3, 5
    in_sizes = [(i, i + 1) for i in range(1, problems_count * 2 + 1, 2)]
    print(f"Input sizes: {in_sizes}\nInput threads: {in_threads}\nOrder: {get_order(in_sizes, in_threads)}")
