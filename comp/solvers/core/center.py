from abc import abstractmethod

from comp.models import CenterData
from comp.utils import tab_out, stringify, assert_valid_dimensions, assert_positive, assert_non_negative
from .base import BaseSolver


class CenterSolver(BaseSolver[CenterData]):
    """Base class for all center's solvers."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solvers = list()

    @abstractmethod
    def add_constraints(self) -> None:
        """Add constraints to the element's solvers."""

        pass

    def setup(self, add_constraints=True) -> None:
        """Set up the optimization problem."""

        if self.setup_done:
            return

        if add_constraints:
            self.add_constraints()

        self.setup_done = True

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
