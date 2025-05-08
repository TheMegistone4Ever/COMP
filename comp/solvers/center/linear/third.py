from functools import partial

from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.core.element import ElementSolver
from comp.solvers.factories import execute_new_solver_from_data
from comp.utils import copy_coeffs


class CenterLinearThird(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData) -> None:
        """
        Initialize the CenterLinearThird solver.

        Initializes the base CenterSolver and pre-calculates both f_c_opt (center's
        perspective optimal for an element) and f_el_opt (element's own optimal)
        values using parallel execution if configured.

        :param data: The CenterData object containing configuration and parameters for the center.
        """

        super().__init__(data)

        self.f_c_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, copy_coeffs(
            element, data.coeffs_functional[e])) for e, element in enumerate(data.elements)])
        self.f_el_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, element_data.copy())
                                                        for element_data in data.elements])

    def modify_constraints(self, element_index: int, element_solver: ElementSolver) -> None:
        """
        Modify the objective for an element's solver for the third linear model.

        Ensures the element solver is set up (without its default goal).
        Sets the element's objective to maximize a weighted sum: d_e^T * y_e + w_e * c_e^T * y_e.
        No additional constraints are added by this method directly, only the goal is modified.

        :param element_index: The index of the element whose solver is being modified.
        :param element_solver: The ElementSolver instance for the specific element.
        """

        if not element_solver.setup_done:
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
