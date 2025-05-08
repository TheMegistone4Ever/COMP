from functools import partial

from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import execute_new_solver_from_data
from comp.utils import copy_coeffs


class CenterLinearThird(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.f_c_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, copy_coeffs(
            element, data.coeffs_functional[e])) for e, element in enumerate(data.elements)])
        self.f_el_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, element_data.copy())
                                                        for element_data in data.elements])

    def modify_constraints(self, element_index, element_solver) -> None:
        """Add constraints to the element's solver."""

        element_solver.setup(set_objective=False)

        element_objective = element_solver.solver.Objective()

        # Objective: Max (d_e^T * y_e + w_e * c_e^T * y_e)
        # d_e^T * y_e
        for i, (coeff_func) in enumerate(self.data.coeffs_functional[element_index]):
            element_objective.SetCoefficient(
                element_solver.y_e[i],
                float(coeff_func)
            )

        # w_e * c_e^T * y_e
        for i, (coeff_func) in enumerate(element_solver.data.coeffs_functional):
            element_objective.SetCoefficient(
                element_solver.get_plan_component(i),
                float(coeff_func * element_solver.data.w)
            )

        element_objective.SetMaximization()
