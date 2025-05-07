from typing import List, Tuple

from comp.parallelization.core import Device, Operation, empiric


# --- Scheduling algorithm components (from your multi_device.py example) ---

def get_multi_device_heuristic_schedule(
        devices: List[Device],
        operations: List[Operation]
) -> Tuple[List[Device], List[Operation]]:
    operations.sort(key=lambda op: op.duration, reverse=True)
    # Create new device instances for the schedule based on the initial ones
    # The original devices in schedule.devices define the number and start times
    scheduled_devices = [Device() for _ in devices]
    # Assign operations to the new device instances
    for operation in operations:
        # Find the device that will finish earliest if this operation is added,
        # This requires calculating the current end time of each device based on ops already on it
        target_device = min(scheduled_devices, key=lambda dev: dev.end)
        target_device.operations.append(operation)

    # Update the schedule's devices
    for new_device in scheduled_devices:
        new_device.update_operation_times()

    return scheduled_devices, operations


def make_permutation_1_1(
        lagged_device: Device,
        advanced_device: Device,
        average_deadline: float
) -> bool:
    for i in range(len(lagged_device.operations)):
        for j in range(len(advanced_device.operations)):
            theta = lagged_device.operations[i].duration - advanced_device.operations[j].duration
            if theta <= 0:
                continue
            # Condition from your provided example
            if theta <= (lagged_device.end - average_deadline):
                lagged_operation = lagged_device.operations.pop(i)
                advanced_operation = advanced_device.operations.pop(j)
                lagged_device.operations.insert(i, advanced_operation)
                advanced_device.operations.insert(j, lagged_operation)
                return True
    return False


def make_permutation_1_2(
        lagged_device: Device,
        advanced_device: Device,
        average_deadline: float
) -> bool:
    for i in range(len(lagged_device.operations)):
        # Iterate backwards for j and k to simplify pop logic or be very careful with indices
        for j_idx in range(len(advanced_device.operations)):
            for k_idx in range(j_idx + 1, len(advanced_device.operations)):
                # Ensure indices are valid after potential pops if iterating forward (safer to copy ops or iterate back)
                # For simplicity, let's assume direct indexing works if we pop carefully
                op_i_lagged = lagged_device.operations[i]
                op_j_advanced = advanced_device.operations[j_idx]
                op_k_advanced = advanced_device.operations[k_idx]

                theta = op_i_lagged.duration - (op_j_advanced.duration + op_k_advanced.duration)
                if theta <= 0:
                    continue

                if theta <= (lagged_device.end - average_deadline):
                    # Perform swap
                    lagged_op_to_move = lagged_device.operations.pop(i)

                    # Pop k_idx first as it's larger
                    adv_op_k = advanced_device.operations.pop(k_idx)
                    adv_op_j = advanced_device.operations.pop(j_idx)

                    lagged_device.operations.insert(i, adv_op_j)
                    lagged_device.operations.insert(i + 1, adv_op_k)  # Insert k after j

                    advanced_device.operations.insert(j_idx, lagged_op_to_move)  # Insert at an original j_idx spot
                    return True
    return False


def make_permutation_2_1(
        lagged_device: Device,
        advanced_device: Device,
        average_deadline: float
) -> bool:
    for i_idx in range(len(lagged_device.operations)):
        for j_idx in range(i_idx + 1, len(lagged_device.operations)):
            for k_idx in range(len(advanced_device.operations)):
                op_i_lagged = lagged_device.operations[i_idx]
                op_j_lagged = lagged_device.operations[j_idx]
                op_k_advanced = advanced_device.operations[k_idx]

                theta = (op_i_lagged.duration + op_j_lagged.duration) - op_k_advanced.duration
                if theta <= 0:
                    continue

                if theta <= (lagged_device.end - average_deadline):
                    # Pop j_idx first as it's larger
                    lagged_op_j = lagged_device.operations.pop(j_idx)
                    lagged_op_i = lagged_device.operations.pop(i_idx)

                    adv_op_k = advanced_device.operations.pop(k_idx)

                    lagged_device.operations.insert(i_idx, adv_op_k)

                    advanced_device.operations.insert(k_idx, lagged_op_i)
                    advanced_device.operations.insert(k_idx + 1, lagged_op_j)
                    return True
    return False


def make_permutation_2_2(
        lagged_device: Device,
        advanced_device: Device,
        average_deadline: float
) -> bool:
    for i_idx in range(len(lagged_device.operations)):
        for j_idx in range(i_idx + 1, len(lagged_device.operations)):
            for k_idx in range(len(advanced_device.operations)):
                for x_idx in range(k_idx + 1, len(advanced_device.operations)):
                    op_i_lagged = lagged_device.operations[i_idx]
                    op_j_lagged = lagged_device.operations[j_idx]
                    op_k_advanced = advanced_device.operations[k_idx]
                    op_x_advanced = advanced_device.operations[x_idx]

                    theta = (op_i_lagged.duration + op_j_lagged.duration) - \
                            (op_k_advanced.duration + op_x_advanced.duration)
                    if theta <= 0:
                        continue

                    if theta <= (lagged_device.end - average_deadline):
                        # Pop from lagged (j_idx then i_idx)
                        lagged_op_j = lagged_device.operations.pop(j_idx)
                        lagged_op_i = lagged_device.operations.pop(i_idx)
                        # Pop from advanced (x_idx then k_idx)
                        adv_op_x = advanced_device.operations.pop(x_idx)
                        adv_op_k = advanced_device.operations.pop(k_idx)

                        # Insert into lagged
                        lagged_device.operations.insert(i_idx, adv_op_k)
                        lagged_device.operations.insert(i_idx + 1, adv_op_x)
                        # Insert into advanced
                        advanced_device.operations.insert(k_idx, lagged_op_i)
                        advanced_device.operations.insert(k_idx + 1, lagged_op_j)
                        return True
    return False


def get_multi_device_schedule_A0(
        devices: List[Device],
        operations: List[Operation]
) -> List[Device]:
    processed_devices, processed_operations = get_multi_device_heuristic_schedule(devices, operations)

    average_deadline = sum(op.duration for op in processed_operations) / len(processed_devices)

    # Balancing loop
    while True:
        # Check for convergence: if all devices end at roughly the same time
        if not processed_devices: break
        first_dev_end = processed_devices[0].end
        if all(abs(dev.end - first_dev_end) < 1e-9 for dev in processed_devices[1:]):
            break

        # Identify lagged and advanced devices
        # Using a small tolerance for float comparisons against average_deadline
        lagged_devices = [dev for dev in processed_devices if dev.end > average_deadline + 1e-9]
        advanced_devices = [dev for dev in processed_devices if dev.end < average_deadline - 1e-9]

        if not lagged_devices or not advanced_devices:
            break  # No further balancing possible

        # Select the most lagged device
        max_lagged_device = max(lagged_devices, key=lambda dev: dev.end - average_deadline)

        # Sort advanced_devices to try permutations with the "most advanced" first (smallest end time)
        advanced_devices.sort(key=lambda dev: dev.end)

        found_permutation_this_iteration = False
        for advanced_device in advanced_devices:
            if not max_lagged_device.operations:  # Cannot move ops from an empty device
                continue
            if not advanced_device.operations and not (
                    make_permutation_1_1(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_2_1(max_lagged_device, advanced_device, average_deadline)
            ):  # If advanced is empty, only 1_1 or 2_1 can move ops to it (by swapping with "nothing")
                # The make_permutation functions assume advanced_device.operations is not empty for some swaps.
                # The provided ones are general. Let's use them as is.
                # If advanced_device has no operations, make_permutation_1_1's inner loop won't run.
                # make_permutation_1_2 needs at least 2 ops on advanced.
                # make_permutation_2_1 can work if advanced_device is empty (transfers 2 from lagged, receives 1) - wait, no, it expects to pop from advanced.
                # The permutations assume operations exist to be swapped.
                # If advanced_device.operations are empty, only transfers *to* it without taking anything *from* it would work.
                # The current permutations are all swaps.
                # Let's proceed with the given permutation functions.
                pass

            if (make_permutation_1_1(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_1_2(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_2_1(max_lagged_device, advanced_device, average_deadline) or
                    make_permutation_2_2(max_lagged_device, advanced_device, average_deadline)):
                found_permutation_this_iteration = True
                break  # Re-evaluate after one successful permutation

        if not found_permutation_this_iteration:
            break  # No balancing permutation found in this full pass

    # Final update of operation's device attribute and their specific start/end times
    # This is important as permutations change ops on devices
    for dev in processed_devices:
        dev.update_operation_times()

    # The schedule object was modified in place by get_multi_device_heuristic_schedule,
    # and the processed_devices list refers to its devices.
    return processed_devices


def get_order(
        sizes: List[Tuple[int, int]],
        threads: int
) -> List[List[int]]:
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
            for device in get_multi_device_schedule_A0(
            [Device() for _ in range(threads)],
            [Operation(empiric(size_tuple), i) for i, size_tuple in enumerate(sizes)]
        )]


if __name__ == "__main__":
    in_threads, problems_count = 3, 5
    in_sizes = [(i, i + 1) for i in range(1, problems_count * 2 + 1, 2)]
    print(f"Input sizes: {in_sizes}\nInput threads: {in_threads}\nOrder: {get_order(in_sizes, in_threads)}")
