from typing import Any, Dict

from comp.models import ElementData
from comp.solvers import BaseSolver
from comp.utils import assert_valid_dimensions, assert_non_negative, assert_positive


class ElementCombinatorialFirst(BaseSolver):
    """Solver for element-level optimization problems. 1'st combinatorial model."""

    def __init__(self, data: ElementData):
        super().__init__(data)

    def setup_variables(self) -> None:
        """Set up optimization variables for the element problem."""

        raise NotImplementedError("setup_variables method not implemented")

    def setup_constraints(self) -> None:
        """Set up constraints for the element problem."""

        raise NotImplementedError("setup_constraints method not implemented")

    def setup_objective(self) -> None:
        """
        Set up the objective function for the element problem.

        min ...
        """

        raise NotImplementedError("setup_objective method not implemented")

    def get_solution(self) -> Dict[str, Any]:
        """Extract solution values with formatting for the element problem."""

        raise NotImplementedError("get_solution method not implemented")

    def print_results(self) -> None:
        """Print the results of the optimization for the element problem."""

        raise NotImplementedError("print_results method not implemented")

    def validate_input(self) -> None:
        """Validate the input data of the optimization for the center problem."""

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
