from dataclasses import dataclass, field
from typing import List

from .operation import Operation


@dataclass
class Device:
    operations: List[Operation] = field(default_factory=list)

    @property
    def end(self) -> float:
        """Returns the end time of the last operation on this device."""

        return sum(operation.duration for operation in self.operations)

    def update_operation_times(self) -> None:
        """Sets start and end times for operations on this device."""

        current_op_start_time = 0.0
        for operation in self.operations:
            operation.start_time_on_device = current_op_start_time
            current_op_start_time += operation.duration
            operation.end_time_on_device = current_op_start_time
