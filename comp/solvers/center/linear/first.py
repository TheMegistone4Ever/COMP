from functools import partial

from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import execute_new_solver_from_data
from comp.utils import copy_coeffs, lp_sum


class CenterLinearFirst(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.f_c_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, copy_coeffs(
            element, data.coeffs_functional[e])) for e, element in enumerate(data.elements)])

    def modify_constraints(self, element_index, element_solver):
        """Add constraints to the element's solver."""

        element_solver.setup()

        # Optimality Equality Constraint: d_e^T * y_e = f_c_opt_e
        element_solver.solver.Add(
            lp_sum(self.data.coeffs_functional[element_index][i] * element_solver.get_plan_component(i)
                   for i in range(element_solver.data.config.num_decision_variables))
            == self.f_c_opt[element_index]
        )
