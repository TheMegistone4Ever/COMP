from typing import Any, Dict, List

from ortools.linear_solver.pywraplp import Variable

from comp.models import ElementData
from comp.solvers.core.element import ElementSolver
from comp.utils import stringify, tab_out


class ElementLinearSecond(ElementSolver):
    """Solver for element-level optimization problems. 2'nd linear model."""

    def __init__(self, data: ElementData) -> None:
        """
        Initialize the ElementLinearSecond solver.

        Calls the constructor of the base ElementSolver.

        :param data: The ElementData object for this element.
        """

        super().__init__(data)

        self.y_star_e: List[Any] = list()

    def setup_variables(self) -> None:
        """
        Set up optimization variables for the second linear element model.

        Calls the base class's `setup_variables` (for y_e) and then add private decision variables (y_star_e).
        """

        super().setup_variables()

        self.y_star_e = [
            self.solver.NumVar(0, self.solver.infinity(), f"y_star_{self.data.config.id}_{i}")
            for i in range(self.data.config.num_decision_variables)
        ]

    def setup_constraints(self) -> None:
        """
        Set up optimization constraints for the second linear element model.

        Adds resource constraints: A_e * (y_e + y_star_e) <= b_e.
        Adds recourse bound constraints: 0 <= b_e_1 <= y_e.
        Adds combined recourse bound constraints: y_e + y_star_e <= b_e_2.
        """

        # Resource constraints: A_e * (y_e + y_star_e) <= b_e
        for i in range(self.data.config.num_constraints):
            self.solver.Add(
                sum(self.data.aggregated_plan_costs[i][j] * (self.y_e[j] + self.y_star_e[j])
                    for j in range(self.data.config.num_decision_variables))
                <= self.data.resource_constraints[0][i]
            )

        # Resource constraints: 0 <= b_e_1 <= y_e
        for i in range(self.data.config.num_decision_variables):
            self.solver.Add(
                self.data.resource_constraints[1][i] <= self.y_e[i]
            )

        # Resource constraints: y_e + y_star_e <= b_e_2
        for i in range(self.data.config.num_decision_variables):
            self.solver.Add(
                self.y_e[i] + self.y_star_e[i] <= self.data.resource_constraints[2][i]
            )

    def setup_objective(self) -> None:
        """Set up the objective function for the second linear element model.

        Maximize c_e^T * y_star_e.
        """

        objective = self.solver.Objective()

        for i, (coeff_func) in enumerate(self.data.coeffs_functional):
            objective.SetCoefficient(
                self.y_star_e[i],
                float(coeff_func)
            )

        objective.SetMaximization()

    def get_solution(self) -> Dict[str, List[float]]:
        """
        Extract solution values for the second linear element model.

        Retrieves the solution values for both decision variables y_e and y_star_e.

        :return: A dictionary with keys "y_e" and "y_star_e", each mapping to a list of float solution values.
        """

        return {
            "y_e": [v.solution_value() for v in self.y_e],
            "y_star_e": [v.solution_value() for v in self.y_star_e],
        }

    def get_plan_component(self, pos: int) -> Variable:
        """
        Get a specific private decision variable (y_star_e[pos]) of the element's plan.

        Used to access an individual private decision variable from the y_star_e vector.

        :param pos: The index of the desired private decision variable in the y_star_e vector.
        :return: The OR-Tools variable object representing y_star_e[pos].
        """

        return self.y_star_e[pos]

    def print_results(self) -> None:
        """
        Print the detailed results of the optimization for this element model.

        Calls the base class's `print_results` and then adds specific output
        for both decision variables (y_e) and private decision variables (y_star_e).
        """

        super().print_results()

        tab_out(f"Optimization results for element {stringify(self.data.config.id)}", (
            ("Decision Variables", stringify((dict_solved := self.solve().plan)["y_e"])),
            ("Private Decision Variables", stringify(dict_solved["y_star_e"])),
        ))

    def quality_functional(self) -> float:
        """
        Calculate the element's quality functional for this model (c_e^T * y_star_e).

        Computes the dot product of the element's functional coefficients (c_e)
        and its solved private decision variables (y_star_e).

        :return: The computed `quality functional` as a float.
        """

        return sum(c_e * y_star_e for c_e, y_star_e in zip(self.data.coeffs_functional, self.solve().plan["y_star_e"]))
