from abc import abstractmethod

from comp.models import CenterData
from .base import BaseSolver


class CenterSolver(BaseSolver[CenterData]):
    """Base class for all center's solvers."""

    def __init__(self, data: CenterData):
        super().__init__(data)

    @abstractmethod
    def add_constraints(self) -> None:
        """Add constraints to the element's solvers."""

        pass

    @abstractmethod
    def get_optimal_solutions(self) -> None:
        """Get optimal solutions for the element's solvers."""

        pass

    def setup(self, get_optimal_solutions=True, add_constraints=True) -> None:
        """Set up the optimization problem."""

        if self.setup_done:
            return

        if get_optimal_solutions:
            self.get_optimal_solutions()
        if add_constraints:
            self.add_constraints()

        self.setup_done = True
