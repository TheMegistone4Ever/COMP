from comp.models import CenterData
from comp.solvers.core import CenterSolver
from comp.solvers.factories import element_solver_fabric
from comp.utils import (assert_valid_dimensions, assert_non_negative, assert_positive, tab_out, stringify,
                        copy_element_coeffs)


class CenterLinearThird(CenterSolver):
    """Solver for center-level optimization problems. 1'st linear model."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solvers = [element_solver_fabric(element) for element in data.elements]
        self.optimal_element_solvers = [element_solver_fabric(element) for element in data.elements]
        self.center_element_solvers = [element_solver_fabric(copy_element_coeffs(element, data.coeffs_functional[e]))
                                       for e, element in enumerate(data.elements)]
        self.center_element_results = list()
        self.optimal_element_results = list()

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

            # - (-f_c_opt_l)
            element_objective.SetOffset(
                element_objective.offset()
                + float(self.center_element_results[e][0])
            )

            # - (w_e * c_e^T * y_e)
            for i, (coeff_func) in enumerate(element_solver.data.coeffs_functional):
                element_objective.SetCoefficient(
                    element_solver.get_plan(i),
                    - float(coeff_func * element_solver.data.w)
                )

            # - (w_e * (- f_el_opt_e))
            element_objective.SetOffset(
                element_objective.offset()
                + float(self.optimal_element_results[e][0] * element_solver.data.w)
            )

            element_objective.SetMaximization()

    def get_optimal_solutions(self) -> None:
        """Get optimal solutions for the element's solvers."""

        self.center_element_results = [solver.setup() or solver.solve() for solver in self.center_element_solvers]
        self.optimal_element_results = [solver.setup() or solver.solve() for solver in self.optimal_element_solvers]

    def print_results(self) -> None:
        """Print the results of the optimization for the center problem."""

        center_functional = 0

        tab_out(f"\nInput data for center {stringify(self.data.config.id)}", (
            ("Center Type", stringify(self.data.config.type)),
            ("Center ID", stringify(self.data.config.id)),
            ("Center Number of Elements", stringify(self.data.config.num_elements)),
            ("Center Functional Coefficients", stringify(self.data.coeffs_functional)),
        ))

        for e, (element_solver) in enumerate(self.element_solvers):
            element_objective, dict_solved = element_solver.solve()

            if element_objective == float("inf"):
                print(f"\nNo optimal solution found for element {self.data.config.id}, "
                      f"thus no solution found for center.")
                return

            element_solver.print_results()

            center_functional += element_objective

        print(f"\nCenter {stringify(self.data.config.id)} quality functional: {stringify(center_functional)}")

    def validate_input(self) -> None:
        """Validate the input data of the optimization for the center problem."""

        assert_valid_dimensions(
            [self.data.coeffs_functional,
             self.data.elements, ],
            [(self.data.config.num_elements,),
             (self.data.config.num_elements,), ],
            ["coeffs_functional",
             "elements", ]
        )

        assert_positive(
            self.data.config.num_elements,
            "data.config.num_elements"
        )
        assert_non_negative(
            self.data.config.id,
            "data.config.id"
        )
