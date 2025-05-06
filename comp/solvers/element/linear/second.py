from typing import Any, Dict, List

from comp.models import ElementData
from comp.solvers.core import ElementSolver
from comp.utils import stringify, tab_out


class ElementLinearSecond(ElementSolver):
    """Solver for element-level optimization problems. 2'nd linear model."""

    def __init__(self, data: ElementData):
        super().__init__(data)

        self.y_star_e: List[Any] = list()

    def setup_variables(self) -> None:
        """Set up optimization variables for the element problem."""

        super().setup_variables()

        self.y_star_e = [
            self.solver.NumVar(0, self.solver.infinity(), f"y_star_{self.data.config.id}_{i}")
            for i in range(self.data.config.num_decision_variables)
        ]

    def setup_constraints(self) -> None:
        """Set up constraints for the element problem."""

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
        """
        Set up the objective function for the element problem.

        Max (c_e^T * y_star_e)
        """

        objective = self.solver.Objective()

        for i, (coeff_func) in enumerate(self.data.coeffs_functional):
            objective.SetCoefficient(
                self.y_star_e[i],
                float(coeff_func)
            )

        objective.SetMaximization()

    def get_solution(self) -> Dict[str, Any]:
        """Extract solution values with formatting for the element problem."""

        return {
            "y_e": [v.solution_value() for v in self.y_e],
            "y_star_e": [v.solution_value() for v in self.y_star_e],
        }

    def get_plan_component(self, pos: int) -> Any:
        """Get the partial functional value for the element problem: c_e[pos]^T * y_star_e[pos]."""

        return self.y_star_e[pos]

    def print_results(self) -> None:
        """Print the results of the optimization for the element problem."""

        super().print_results()

        tab_out(f"Optimization results for element {stringify(self.data.config.id)}", (
            ("Decision Variables", stringify((dict_solved := self.solve()[1])["y_e"])),
            ("Private Decision Variables", stringify(dict_solved["y_star_e"])),
        ))

    def quality_functional(self) -> float:
        """Calculate the element's quality functional: c_e^T * y_star_e."""

        return sum(c_e * y_star_e for c_e, y_star_e in zip(self.data.coeffs_functional, self.solve()[1]["y_star_e"]))
