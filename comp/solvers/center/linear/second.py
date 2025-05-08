from functools import partial

from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import execute_new_solver_from_data
from comp.utils import lp_sum


class CenterLinearSecond(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.f_el_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, element_data.copy())
                                                        for element_data in data.elements])

    def modify_constraints(self, element_index, element_solver):
        """Add constraints to the element's solver."""

        element_solver.setup(set_objective=False)

        # Optimality Inequality Constraint: c_e^T * y_e >= f_el_opt_e * (1 - delta_e)
        element_solver.solver.Add(
            lp_sum(element_solver.data.coeffs_functional[i] * element_solver.get_plan_component(i)
                   for i in range(element_solver.data.config.num_decision_variables))
            >= self.f_el_opt[element_index] * (1 - element_solver.data.delta)
        )

        element_objective = element_solver.solver.Objective()

        # Objective: Max (d_e^T * y_e)
        for i, (coeff_func) in enumerate(self.data.coeffs_functional[element_index]):
            element_objective.SetCoefficient(
                element_solver.y_e[i],
                float(coeff_func)
            )

        element_objective.SetMaximization()
