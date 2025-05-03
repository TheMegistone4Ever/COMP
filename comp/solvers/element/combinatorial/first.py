from typing import List, Any, Dict

from comp.models import ElementData
from comp.solvers import BaseSolver
from comp.utils import assert_valid_dimensions, assert_non_negative, assert_positive


class ElementCombinatorialFirst(BaseSolver):
    """Solver for element-level optimization problems. 1'st combinatorial model."""

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
