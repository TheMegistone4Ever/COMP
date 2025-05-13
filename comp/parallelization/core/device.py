from dataclasses import dataclass, field
from typing import List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .operation import Operation


@dataclass
class Device:
    """Class representing a device in the system with its operations."""

    operations: List[Operation] = field(default_factory=list)

    @property
    def end(self: Self) -> float:
        """
        Calculate the end time of the last operation on this device.

        The end time is determined by summing the durations of all operations
        currently assigned to this device.

        :return: The total duration of all operations on the device, representing its finish time.
        """

        return sum(operation.duration for operation in self.operations)

    def update_operation_times(self) -> None:
        """
        Set the start and end times for all operations assigned to this device.

        This method iterates through the operations on the device, sequentially
        calculating and setting their individual start and end times based on
        their durations and the end times of preceding operations.
        """

        current_op_start_time = .0
        for operation in self.operations:
            operation.start_time_on_device = current_op_start_time
            current_op_start_time += operation.duration
            operation.end_time_on_device = current_op_start_time
