from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from comp.models import BaseData

T = TypeVar("T", bound=BaseData)


class BaseSolver(ABC, Generic[T]):
    """Base class for all optimization solvers."""

    def __init__(self, data: T) -> None:
        """
        Initialize the base solver with data and validation.

        Stores the input data and calls the `validate_input` method.
        Sets `setup_done` flag to False.

        :param data: The data object (subclass of BaseData) for the solver.
        """

        self.data: T = data
        self.setup_done: bool = False

        self.validate_input()

    @abstractmethod
    def print_results(self) -> None:
        """
        Print the results of the optimization problem.

        Concrete implementations should define how to format and display
        the solution and relevant metrics.
        """

        pass

    @abstractmethod
    def validate_input(self) -> None:
        """
        Validate the input data for the optimization problem.

        Concrete implementations should define specific checks for their
        required data structures and values.
        """

        pass

    @abstractmethod
    def quality_functional(self) -> Union[str, float]:
        """
        Calculate and return the quality functional of the solved problem.

        Concrete implementations should define how the quality functional
        is computed based on the problem type and solution.
        It can return a numerical value or a string representation if appropriate.

        :return: The calculated quality is functional, as a float or a string.
        """

        pass
