from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from comp.models import BaseData

T = TypeVar("T", bound=BaseData)


class BaseSolver(ABC, Generic[T]):
    """Base class for all optimization solvers."""

    def __init__(self, data: T):
        self.data: T = data
        self.setup_done: bool = False

        self.validate_input()

    @abstractmethod
    def print_results(self) -> None:
        """Print the results of the optimization."""

        pass

    @abstractmethod
    def validate_input(self) -> None:
        """Validate the input data for the optimization problem."""

        pass
