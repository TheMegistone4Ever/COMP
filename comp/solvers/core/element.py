from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, List

from ortools.linear_solver import pywraplp

from comp.models import ElementData
from comp.utils import tab_out, stringify, assert_valid_dimensions, assert_non_negative, assert_positive
from .base import BaseSolver


class ElementSolver(BaseSolver[ElementData]):
    """Base class for all element's solvers."""

    def __init__(self, data: ElementData):
        super().__init__(data)

        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.solved = False
        self.objective_value: Optional[float] = None
        self.solution: Optional[Dict[str, Any]] = None

        self.y_e: List[Any] = list()

    @abstractmethod
    def setup_constraints(self) -> None:
        """Set up optimization constraints for the element."""

        pass

    @abstractmethod
    def setup_objective(self) -> None:
        """Set up the objective function for the element."""

        pass

    @abstractmethod
    def get_solution(self) -> Dict[str, Any]:
        """Extract the solution from the element's solver."""

        pass

    @abstractmethod
    def get_plan_component(self, pos: int) -> Any:
        """Get the vector component of the element's plan at a specific position."""

        pass

    def setup_variables(self) -> None:
        """Set up optimization variables for the element."""

        self.y_e = [
            self.solver.NumVar(0, self.solver.infinity(), f"y_{self.data.config.id}_{i}")
            for i in range(self.data.config.num_decision_variables)
        ]

    def setup(self, set_variables=True, set_constraints=True, set_objective=True) -> None:
        """Set up the optimization problem for the element."""

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
        """Solve the optimization problem for the element."""

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
        """Get the objective value of the optimization for the element problem."""

        return self.objective_value

    def print_results(self) -> None:
        """Print the results of the optimization for the element problem."""

        element_objective, dict_solved = self.solve()

        if element_objective == float("inf"):
            print(f"\nNo optimal solution found for element: {self.data.config.id}.")
            return

        tab_out(f"\nInput data for element {stringify(self.data.config.id)}", (
            ("Element Type", stringify(self.data.config.type)),
            ("Element ID", stringify(self.data.config.id)),
            ("Element Number of Decision Variables", stringify(self.data.config.num_decision_variables)),
            ("Element Number of Constraints", stringify(self.data.config.num_constraints)),
            ("Element Functional Coefficients", stringify(self.data.coeffs_functional)),
            ("Element Resource Constraints", stringify(self.data.resource_constraints)),
            ("Element Aggregated Plan Costs", stringify(self.data.aggregated_plan_costs)),
            ("Element Delta", stringify(self.data.delta)),
            ("Element W", stringify(self.data.w)),
        ))

        print(f"\nElement {stringify(self.data.config.id)} quality functional: {stringify(self.quality_functional())}")

    def validate_input(self) -> None:
        """Validate the input data of the optimization for the element problem."""

        assert_valid_dimensions(
            [self.data.coeffs_functional,
             self.data.resource_constraints[0],
             self.data.resource_constraints[1],
             self.data.resource_constraints[2],
             self.data.aggregated_plan_costs, ],
            [(self.data.config.num_decision_variables,),
             (self.data.config.num_constraints,),
             (self.data.config.num_decision_variables,),
             (self.data.config.num_decision_variables,),
             (self.data.config.num_constraints, self.data.config.num_decision_variables), ],
            ["coeffs_functional",
             "resource_constraints[0]",
             "resource_constraints[1]",
             "resource_constraints[2]",
             "aggregated_plan_costs", ]
        )
        assert_non_negative(
            self.data.delta,
            "data.delta"
        )
        assert_non_negative(
            self.data.config.id,
            "data.config.id"
        )
        assert_positive(
            self.data.config.num_decision_variables,
            "data.config.num_decision_variables"
        )
        assert_positive(
            self.data.config.num_constraints,
            "data.config.num_constraints"
        )
