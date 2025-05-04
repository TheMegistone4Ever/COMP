from typing import List, Any, Dict

from comp.models import ElementData
from comp.solvers import BaseSolver
from comp.utils import assert_valid_dimensions, assert_non_negative, assert_positive, tab_out, stringify


class ElementLinearSecond(BaseSolver):
    """Solver for element-level optimization problems. 2'nd linear model."""

    def __init__(self, data: ElementData):
        super().__init__()
        # Validate input dimensions
        assert_valid_dimensions(
            [data.coeffs_functional,
             data.resource_constraints[0],
             data.resource_constraints[1],
             data.resource_constraints[2],
             data.aggregated_plan_costs,
             data.schedules,
             data.interest,
             data.weight_coefficients, ],
            [(data.config.num_decision_variables,),
             (data.config.num_constraints,),
             (data.config.num_decision_variables,),
             (data.config.num_decision_variables,),
             (data.config.num_constraints, data.config.num_decision_variables),
             (data.config.num_schedules,),
             (data.config.num_constraints, data.config.num_schedules),
             (data.config.num_constraints, data.config.num_schedules), ],
            ["coeffs_functional",
             "resource_constraints[0]",
             "resource_constraints[1]",
             "resource_constraints[2]",
             "aggregated_plan_costs",
             "schedules",
             "interest",
             "weight_coefficients", ]
        )
        assert_non_negative(
            data.delta,
            "data.delta"
        )
        assert_non_negative(
            data.config.id,
            "data.config.id"
        )
        assert_positive(
            data.config.num_decision_variables,
            "data.config.num_decision_variables"
        )
        assert_positive(
            data.config.num_constraints,
            "data.config.num_constraints"
        )
        assert_positive(
            data.config.num_schedules,
            "data.config.num_schedules"
        )

        self.data = data
        self.y_e: List[Any] = list()
        self.y_star_e: List[Any] = list()

    def setup_variables(self) -> None:
        """Set up optimization variables for the element problem."""

        self.y_e = [
            self.solver.NumVar(0, self.solver.infinity(), f"y_{self.data.config.id}_{i}")
            for i in range(self.data.config.num_decision_variables)
        ]

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

        max (c_e^T * y_star_e)
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

    def print_results(self) -> None:
        """Print the results of the optimization for the element problem."""

        element_objective, dict_solved = self.solve()

        if element_objective == float("inf"):
            print("\nNo optimal solution found.")
            return

        tab_out(f"\nInput data for element {stringify(self.data.config.id)}", (
            ("Element Type", stringify(self.data.config.type)),
            ("Element ID", stringify(self.data.config.id)),
            ("Element Number of Decision Variables", stringify(self.data.config.num_decision_variables)),
            ("Element Number of Constraints", stringify(self.data.config.num_constraints)),
            ("Element Number of Schedules", stringify(self.data.config.num_schedules)),
            ("Element Functional Coefficients", stringify(self.data.coeffs_functional)),
            ("Element Resource Constraints", stringify(self.data.resource_constraints)),
            ("Element Aggregated Plan Costs", stringify(self.data.aggregated_plan_costs)),
            ("Element Delta", stringify(self.data.delta)),
            ("Element Schedules", stringify(self.data.schedules)),
            ("Element Interest", stringify(self.data.interest)),
            ("Element Weight Coefficients", stringify(self.data.weight_coefficients)),
        ))

        tab_out(f"\nOptimization results for element {stringify(self.data.config.id)}", (
            ("Decision Variables", stringify(dict_solved["y_e"])),
            ("Private Decision Variables", stringify(dict_solved["y_star_e"])),
        ))

        print(f"\nElement {stringify(self.data.config.id)} quality functional: {stringify(element_objective)}")
