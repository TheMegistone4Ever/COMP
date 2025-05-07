from abc import abstractmethod
from typing import Tuple

from comp.models import CenterData
from comp.parallelization import ParallelExecutor, get_order
from comp.utils import (tab_out, stringify, assert_valid_dimensions, assert_positive, assert_non_negative,
                        get_lp_problem_sizes)
from .base import BaseSolver


class CenterSolver(BaseSolver[CenterData]):
    """Base class for all center's solvers."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solvers = list()
        self.order = get_order(get_lp_problem_sizes(data.elements), data.config.num_threads)
        self.parallel_executor = ParallelExecutor(
            min_threshold=data.config.min_parallelisation_threshold,
            num_threads=data.config.num_threads,
            order=self.order,
        )

    @abstractmethod
    def add_constraints(self) -> None:
        """Add constraints to the element's solvers."""

        pass

    def quality_functional(self) -> Tuple[str, float]:
        """Calculate the center's quality functional: sum_e (d_e^T * y_e)."""

        y_e_list = self.parallel_executor.execute([lambda solver=solver_e: solver.solve()[1]["y_e"]
                                                   for solver_e in self.element_solvers])

        sums = [sum(d_e * y_e for d_e, y_e in zip(self.data.coeffs_functional[e], y))
                for e, y in enumerate(y_e_list) if y is not None]

        return stringify(sums), sum(sums)

    def setup(self, add_constraints=True) -> None:
        """Set up the optimization problem."""

        if self.setup_done:
            return

        if add_constraints:
            self.add_constraints()

        self.setup_done = True

    def print_results(self) -> None:
        """Print the results of the optimization for the center problem."""

        tab_out(f"\nInput data for center {stringify(self.data.config.id)}", (
            ("Center Type", stringify(self.data.config.type)),
            ("Center ID", stringify(self.data.config.id)),
            ("Center Number of Elements", stringify(self.data.config.num_elements)),
            ("Center Functional Coefficients", stringify(self.data.coeffs_functional)),
            ("Center Min Parallelization Threshold", stringify(self.data.config.min_parallelisation_threshold)),
            ("Center Number of Threads", stringify(self.data.config.num_threads)),
            ("Center Parallelization Order", stringify(self.order)),
        ))

        self.parallel_executor.execute([lambda solver=solver_e: solver.print_results()
                                        for solver_e in self.element_solvers])

        print(f"\nCenter {stringify(self.data.config.id)} quality functional: {stringify(self.quality_functional())}")

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
