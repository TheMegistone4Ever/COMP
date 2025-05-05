from abc import ABC, abstractmethod

from comp.models import BaseData


class BaseSolver(ABC):
    """Base class for all optimization solvers."""

    def __init__(self, data: BaseData):
        self.data = data
        self.setup_done = False

        self.validate_input()

    @abstractmethod
    def print_results(self) -> None:
        """Print the results of the optimization."""

        pass

    @abstractmethod
    def validate_input(self) -> None:
        """Validate the input data for the optimization problem."""

        pass
