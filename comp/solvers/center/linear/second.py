from functools import partial

from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import new_element_solver, execute_solver_from_data
from comp.utils import copy_coeffs, lp_sum


class CenterLinearSecond(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solvers = [new_element_solver(copy_coeffs(element, data.coeffs_functional[e]))
                                for e, element in enumerate(data.elements)]
        self.f_el_opt = self.parallel_executor.execute([partial(execute_solver_from_data, element_data.copy())
                                                        for element_data in data.elements])

    def add_constraints(self) -> None:
        """Add constraints to the element's solvers."""

        for e, (element_solver) in enumerate(self.element_solvers):
            element_solver.setup()

            # Optimality Inequality Constraint: c_e^T * y_e >= f_el_opt_e * (1 - delta_e)
            element_solver.solver.Add(
                lp_sum(element_solver.data.coeffs_functional[i] * element_solver.get_plan_component(i)
                       for i in range(element_solver.data.config.num_decision_variables))
                >= self.f_el_opt[e] * (1 - element_solver.data.delta)
            )
