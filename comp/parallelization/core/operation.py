from dataclasses import dataclass
from typing import Optional


@dataclass
class Operation:
    """Class representing an operation in the system with duration, device index, and timing information."""

    duration: float
    original_index: int
    start_time_on_device: Optional[float] = None
    end_time_on_device: Optional[float] = None
