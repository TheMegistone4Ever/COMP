from abc import abstractmethod
from typing import Dict, Optional, List, Any

from ortools.linear_solver.pywraplp import Solver, Variable

from comp.models import ElementData, ElementSolution
from comp.utils import (
    assert_non_negative,
    assert_positive,
    assert_valid_dimensions,
    stringify,
    tab_out,
)
from .base import BaseSolver


class ElementSolver(BaseSolver[ElementData]):
    """Base class for all element＇s solvers."""

    def __init__(self, data: ElementData) -> None:
        """
        Initialize the ElementSolver.

        Sets up the base solver, creates an OR-Tools GLOP solver instance,
        and initializes solution-related attributes.

        :param data: The ElementData object containing configuration for this element.
        """

        super().__init__(data)

        self.solver = Solver.CreateSolver("GLOP")
        self.solved: bool = False
        self.status: int = -1
        self.solution: Optional[ElementSolution] = None

        self.y_e: List[Variable] = list()

    @abstractmethod
    def setup_constraints(self) -> None:
        """
        Abstract method to set up optimization constraints for the element.

        Concrete element solver implementations must define this to add
        problem-specific constraints to the OR-Tools solver.
        """

        pass

    @abstractmethod
    def setup_objective(self) -> None:
        """
        Abstract method to set up the objective function for the element.

        Concrete element solver implementations must define this to specify
        the goal (e.g., maximization of a linear expression) in the OR-Tools solver.
        """

        pass

    @abstractmethod
    def get_plan(self) -> Dict[str, List[float]]:
        """
        Abstract method to extract the plan from the OR-Tools solver.

        Concrete implementations should retrieve the values of decision variables
        and format them into a dictionary.
        The list should contain float values.

        :return: A dictionary where keys are variable names (e.g., "y_e") and
                 values are lists of their corresponding float solution values.
        """

        pass

    def set_solution(self, solution: ElementSolution) -> None:
        """
        Set the solution of the element＇s solver from pre-computed results.

        This is used when the solution is obtained externally (e.g., by a center solver)
        and needs to be populated into this element solver instance.

        :param solution: The solution object containing the objective value and plan variables.
        """

        self.solution = solution
        self.solved = solution is not None

    @abstractmethod
    def get_plan_component(self, pos: int) -> Variable:
        """
        Abstract method to get a specific component of the element＇s plan variables.

        This is used to access individual decision variables (or expressions involving them)
        by their position/index, typically for constructing constraints or objectives
        in a linked (e.g., center) solver.

        :param pos: The index of the desired plan component (decision variable).
        :return: The plan component, typically an OR-Tools variable object or similar.
        """

        pass

    def setup_variables(self) -> None:
        """
        Set up optimization variables (decision variables) for the element.

        This method creates the primary decision variables (y_e) for the element＇s
        problem within the OR-Tools solver.
        Concrete subclasses might extend this to add more variables.
        """

        self.y_e = [
            self.solver.NumVar(0, self.solver.infinity(), f"y_{self.data.config.id}_{i}")
            for i in range(self.data.config.num_decision_variables)
        ]

    def setup(self, set_variables: bool = True, set_constraints: bool = True, set_objective: bool = True) -> None:
        """
        Set up the complete optimization problem for the element.

        This orchestrates the setup process by optionally calling `setup_variables`,
        `setup_constraints`, and `setup_objective`.
        It ensures setup is done only once.

        :param set_variables: If True, call `setup_variables`.
        :param set_constraints: If True, call `setup_constraints`.
        :param set_objective: If True, call `setup_objective`.
        """

        if self.setup_done:
            return

        if set_variables:
            self.setup_variables()
        if set_constraints:
            self.setup_constraints()
        if set_objective:
            self.setup_objective()

        self.setup_done = True

    def solve(self) -> ElementSolution:
        """
        Solve the optimization problem for the element.

        If the problem hasn’t been set up, it raises a RuntimeError.
        If not already solved, it calls the OR-Tools solver.
        If an optimal solution is found, it stores and returns the objective value and solution variables.
        Otherwise, it returns infinity and an empty dictionary.

        :raises RuntimeError: If `setup()` has not been called first.
        :return: A tuple containing the objective value (float, or float("-inf") if no solution)
                 and a dictionary of solution variables (Dict[str, List[float]]).
        """

        if not self.setup_done:
            raise RuntimeError("Solver setup is not done. Call setup() before solve().")

        if not self.solved:
            self.solved = True
            self.status = self.solver.Solve()
            if self.status in (Solver.OPTIMAL, Solver.FEASIBLE):
                self.solution = ElementSolution(self.solver.Objective().Value(), self.get_plan())
            else:
                self.solution = ElementSolution(float("-inf"), dict())

        return self.solution

    def print_results(self) -> None:
        """
        Print the results of the optimization for the element problem.

        Solves the problem if not already solved.
        If no optimal solution is found, print a message.
        Otherwise, displays input data and the element＇s quality functional.
        Concrete subclasses may extend this to print more specific solution details.
        """

        if (solution := self.solve()).objective == float("-inf") and not solution.plan:
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
        """
        Validate the input data for the element optimization problem.

        Checks dimensions of various coefficient arrays and constraint vectors
        against configured numbers of decision variables and constraints.
        It also validates non-negativity of delta and ID, and positivity of
        variable/constraint counts.
        """

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
        if self.data.delta is not None:
            assert_non_negative(
                self.data.delta,
                "data.delta"
            )
        if self.data.w is not None:
            for w in self.data.w:
                assert_non_negative(
                    w,
                    "data.w"
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

    def get_results_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the solver＇s results.

        :return: A dictionary containing the solver＇s ID, type, status,
                    solution objective, plan, and quality functional.
        """

        return {
            "id": self.data.config.id,
            "type": self.data.config.type.name,
            "status": self.status,
            "solution_objective": self.solution.objective,
            "solution_plan": self.solution.plan,
            "quality_functional": self.quality_functional() if (self.solution and self.solution.plan) else "N/A",
        }
