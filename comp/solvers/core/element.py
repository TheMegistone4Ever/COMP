from abc import abstractmethod
from typing import Any, Tuple, Dict, Optional

from ortools.linear_solver import pywraplp

from comp.models import ElementData
from .base import BaseSolver


class ElementSolver(BaseSolver[ElementData]):
    """Base class for all element's solvers."""

    def __init__(self, data: ElementData):
        super().__init__(data)

        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.solved = False
        self.objective_value: Optional[float] = None
        self.solution: Optional[Dict[str, Any]] = None

    @abstractmethod
    def setup_variables(self) -> None:
        """Set up optimization variables."""

        pass

    @abstractmethod
    def setup_constraints(self) -> None:
        """Set up optimization constraints."""

        pass

    @abstractmethod
    def setup_objective(self) -> None:
        """Set up the objective function."""

        pass

    @abstractmethod
    def get_solution(self) -> Dict[str, Any]:
        """Extract the solution from the solver."""

        pass

    @abstractmethod
    def get_plan(self, pos: int) -> Any:
        """Get the plan's vector component for a specific position."""

        pass

    def setup(self, set_variables=True, set_constraints=True, set_objective=True) -> None:
        """Set up the optimization problem."""

        if self.setup_done:
            return

        if set_variables:
            self.setup_variables()
        if set_constraints:
            self.setup_constraints()
        if set_objective:
            self.setup_objective()

        self.setup_done = True

    def solve(self) -> Tuple[float, Any]:
        """Solve the optimization problem."""

        if not self.setup_done:
            raise RuntimeError("Solver setup is not done. Call setup() before solve().")

        if not self.solved:
            self.solved = True
            if self.solver.Solve() == pywraplp.Solver.OPTIMAL:
                self.objective_value = self.solver.Objective().Value()
                self.solution = self.get_solution()
            else:
                self.objective_value = float("inf")
                self.solution = dict()
        return self.objective_value, self.solution

    def get_objective_value(self) -> float:
        """Get the objective value of the optimization."""

        return self.objective_value
