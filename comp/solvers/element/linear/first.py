from typing import Dict, List

from ortools.linear_solver.pywraplp import Variable

from comp.models import ElementData
from comp.solvers.core.element import ElementSolver
from comp.utils import stringify, tab_out


class ElementLinearFirst(ElementSolver):
    """Solver for element-level optimization problems. 1'st linear model."""

    def __init__(self, data: ElementData) -> None:
        """
        Initialize the ElementLinearFirst solver.

        Calls the constructor of the base ElementSolver.

        :param data: The ElementData object for this element.
        """

        super().__init__(data)

    def setup_constraints(self) -> None:
        """
        Set up optimization constraints for the first linear element model.

        Adds resource constraints: A_e * y_e <= b_e.
        Adds bound resource constraints: 0 <= b_e_1 <= y_e <= b_e_2.
        """

        # Resource constraints: A_e * y_e <= b_e
        for i in range(self.data.config.num_constraints):
            self.solver.Add(
                sum(self.data.aggregated_plan_costs[i][j] * self.y_e[j]
                    for j in range(self.data.config.num_decision_variables))
                <= self.data.resource_constraints[0][i]
            )

        # Resource constraints: 0 <= b_e_1 <= y_e <= b_e_2
        for i in range(self.data.config.num_decision_variables):
            self.solver.Add(
                self.data.resource_constraints[1][i] <= self.y_e[i]
            )
            self.solver.Add(
                self.y_e[i] <= self.data.resource_constraints[2][i]
            )

    def setup_objective(self) -> None:
        """
        Set up the objective function for the first linear element model.

        Maximize c_e^T * y_e.
        """

        objective = self.solver.Objective()

        for i, (coeff_func) in enumerate(self.data.coeffs_functional):
            objective.SetCoefficient(
                self.y_e[i],
                float(coeff_func)
            )

        objective.SetMaximization()

    def get_solution(self) -> Dict[str, List[float]]:
        """
        Extract solution values for the first linear element model.

        Retrieves the solution values for the decision variables y_e.

        :return: A dictionary with one key "y_e" mapping to a list of float solution values.
        """

        return {
            "y_e": [v.solution_value() for v in self.y_e],
        }

    def get_plan_component(self, pos: int) -> Variable:
        """
        Get a specific decision variable (y_e[pos]) of the element's plan.

        Used to access an individual decision variable from the y_e vector.

        :param pos: The index of the desired decision variable in the y_e vector.
        :return: The OR-Tools variable object representing y_e[pos].
        """

        return self.y_e[pos]

    def print_results(self) -> None:
        """
        Print the detailed results of the optimization for this element model.

        Calls the base class's `print_results` and then add specific output
        for the decision variables (y_e).
        """

        super().print_results()

        tab_out(f"Optimization results for element {stringify(self.data.config.id)}", (
            ("Decision Variables", stringify(self.solve().plan.get("y_e"))),
        ))

    def quality_functional(self) -> float:
        """
        Calculate the element's quality functional for this model (c_e^T * y_e).

        Computes the dot product of the element's functional coefficients (c_e)
        and its solved decision variables (y_e).

        :return: The computed `quality functional` as a float.
        """

        return sum(c * y for c, y in zip(self.data.coeffs_functional, self.solve().plan.get("y_e")))
