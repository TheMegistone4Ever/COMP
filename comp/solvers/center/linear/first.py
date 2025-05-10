from dataclasses import replace
from functools import partial

from comp.models import CenterData, ElementType
from comp.solvers.core import CenterSolver
from comp.solvers.core.element import ElementSolver
from comp.solvers.factories import execute_new_solver_from_data
from comp.utils import lp_sum


class CenterLinearFirst(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData) -> None:
        """
        Initialize the CenterLinearFirst solver.

        This involves initializing the base CenterSolver and pre-calculating
        the optimal values (f_c_opt) for each element's functional using
        parallel execution if configured.

        :param data: The CenterData object containing configuration and parameters for the center.
        """

        super().__init__(data)

        self.f_c_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, replace(
            element_data, coeffs_functional=data.coeffs_functional[e], config=replace(
                element_data.config, type=ElementType.DECENTRALIZED))) for e, element_data in enumerate(data.elements)])

    def modify_constraints(self, element_index: int, element_solver: ElementSolver) -> None:
        """
        Add specific constraints to an element's solver for the first linear model.

        This method ensures the element solver is set up, then adds an equality
        constraint: d_e^T * y_e = f_c_opt_e, where f_c_opt_e is the
        pre-calculated optimal value for the element's functional, according to the center.

        :param element_index: The index of the element whose solver is being modified.
        :param element_solver: The ElementSolver instance for the specific element.
        """

        if not element_solver.setup_done:
            element_solver.setup()

        # Optimality Equality Constraint: d_e^T * y_e = f_c_opt_e
        element_solver.solver.Add(
            lp_sum(self.data.coeffs_functional[element_index][i] * element_solver.get_plan_component(i)
                   for i in range(element_solver.data.config.num_decision_variables))
            == self.f_c_opt[element_index]
        )
