from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import element_solver_fabric
from comp.utils import copy_element_coeffs, lp_sum


class CenterLinearFirst(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solvers = [element_solver_fabric(element) for element in data.elements]
        self.center_element_solvers = [element_solver_fabric(copy_element_coeffs(element, data.coeffs_functional[e]))
                                       for e, element in enumerate(data.elements)]
        self.f_c_opt = list()

    def add_constraints(self) -> None:
        """Add constraints to the element's solvers."""

        for e, (element_solver) in enumerate(self.element_solvers):
            element_solver.setup()

            # Optimality Equality Constraint: d_e^T * y_e = f_c_opt_e
            element_solver.solver.Add(
                lp_sum(self.data.coeffs_functional[e][i] * element_solver.get_plan_component(i)
                       for i in range(element_solver.data.config.num_decision_variables))
                == self.f_c_opt[e]
            )

    def get_optimal_solutions(self) -> None:
        """Get optimal solutions for the element's solvers."""

        self.f_c_opt = [solver.setup() or solver.solve()[0] for solver in self.center_element_solvers]
