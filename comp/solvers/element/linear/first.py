from typing import Any, Dict

from comp.models import ElementData
from comp.solvers.core.element import ElementSolver
from comp.utils import stringify, tab_out


class ElementLinearFirst(ElementSolver):
    """Solver for element-level optimization problems. 1'st linear model."""

    def __init__(self, data: ElementData):
        super().__init__(data)

    def setup_constraints(self) -> None:
        """Set up constraints for the element problem."""

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
        Set up the objective function for the element problem.

        Max (c_e^T * y_e)
        """

        objective = self.solver.Objective()

        for i, (coeff_func) in enumerate(self.data.coeffs_functional):
            objective.SetCoefficient(
                self.y_e[i],
                float(coeff_func)
            )

        objective.SetMaximization()

    def get_solution(self) -> Dict[str, Any]:
        """Extract solution values with formatting for the element problem."""

        return {
            "y_e": [v.solution_value() for v in self.y_e],
        }

    def get_plan_component(self, pos: int) -> Any:
        """Get the partial functional value for the element problem: c_e[pos]^T * y_e[pos]."""

        return self.y_e[pos]

    def print_results(self) -> None:
        """Print the results of the optimization for the element problem."""

        super().print_results()

        tab_out(f"Optimization results for element {stringify(self.data.config.id)}", (
            ("Decision Variables", stringify(self.solve()[1]["y_e"])),
        ))

    def quality_functional(self) -> float:
        """Calculate the element's quality functional: c_e^T * y_e."""

        return sum(c_e * y_e for c_e, y_e in zip(self.data.coeffs_functional, self.solve()[1]["y_e"]))
