from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import element_solver_fabric
from comp.utils import copy_element_coeffs


class CenterLinearThird(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solvers = [element_solver_fabric(element) for element in data.elements]
        self.optimal_element_solvers = [element_solver_fabric(element) for element in data.elements]
        self.center_element_solvers = [element_solver_fabric(copy_element_coeffs(element, data.coeffs_functional[e]))
                                       for e, element in enumerate(data.elements)]
        self.f_c_opt = list()
        self.f_el_opt = list()

    def add_constraints(self) -> None:
        """Add constraints to the element's solvers."""

        for e, (element_solver) in enumerate(self.element_solvers):
            element_solver.setup(set_objective=False)

            element_objective = element_solver.solver.Objective()

            # Objective: max - ((d_l^T * y_e - f_c_opt_l) + w_e * (c_e^T * y_e - f_el_opt_e))
            # - (d_l^T * y_e)
            for i, (coeff_func) in enumerate(self.data.coeffs_functional[e]):
                element_objective.SetCoefficient(
                    element_solver.y_e[i],
                    - float(coeff_func)
                )

            # - (- f_c_opt_l)
            element_objective.SetOffset(
                element_objective.offset()
                + float(self.f_c_opt[e])
            )

            # - (w_e * c_e^T * y_e)
            for i, (coeff_func) in enumerate(element_solver.data.coeffs_functional):
                element_objective.SetCoefficient(
                    element_solver.get_plan_component(i),
                    - float(coeff_func * element_solver.data.w)
                )

            # - (w_e * (- f_el_opt_e))
            element_objective.SetOffset(
                element_objective.offset()
                + float(self.f_el_opt[e] * element_solver.data.w)
            )

            element_objective.SetMaximization()

    def get_optimal_solutions(self) -> None:
        """Get optimal solutions for the element's solvers."""

        self.f_c_opt = [solver.setup() or solver.solve()[0] for solver in self.center_element_solvers]
        self.f_el_opt = [solver.setup() or solver.solve()[0] for solver in self.optimal_element_solvers]
