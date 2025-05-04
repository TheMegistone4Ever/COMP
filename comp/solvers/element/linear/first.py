from typing import List, Any, Dict

from comp.models import ElementData
from comp.solvers import BaseSolver
from comp.utils import assert_valid_dimensions, assert_non_negative, assert_positive, tab_out, stringify


class ElementLinearFirst(BaseSolver):
    """Solver for element-level optimization problems. 1'st linear model."""

    def __init__(self, data: ElementData):
        super().__init__()

        self.data = data
        self.y_e: List[Any] = list()

        self.validate_input()

    def setup_variables(self) -> None:
        """Set up optimization variables for the element problem."""

        self.y_e = [
            self.solver.NumVar(0, self.solver.infinity(), f"y_{self.data.config.id}_{i}")
            for i in range(self.data.config.num_decision_variables)
        ]

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

        max (c_e^T * y_e)
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
        ))

        print(f"\nElement {stringify(self.data.config.id)} quality functional: {stringify(element_objective)}")

    def validate_input(self) -> None:
        """Validate the input data of the optimization for the element problem."""

        assert_valid_dimensions(
            [self.data.coeffs_functional,
             self.data.resource_constraints[0],
             self.data.resource_constraints[1],
             self.data.resource_constraints[2],
             self.data.aggregated_plan_costs,
             self.data.schedules,
             self.data.interest,
             self.data.weight_coefficients, ],
            [(self.data.config.num_decision_variables,),
             (self.data.config.num_constraints,),
             (self.data.config.num_decision_variables,),
             (self.data.config.num_decision_variables,),
             (self.data.config.num_constraints, self.data.config.num_decision_variables),
             (self.data.config.num_schedules,),
             (self.data.config.num_constraints, self.data.config.num_schedules),
             (self.data.config.num_constraints, self.data.config.num_schedules), ],
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
        assert_positive(
            self.data.config.num_schedules,
            "data.config.num_schedules"
        )
