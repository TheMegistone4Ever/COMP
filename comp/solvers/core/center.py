from abc import abstractmethod
from functools import partial
from typing import Tuple, Dict, List, Callable

from comp.models import CenterData, ElementData
from comp.parallelization import ParallelExecutor, get_order
from comp.solvers.core.element import ElementSolver
from comp.solvers.factories import new_element_solver
from comp.utils import (tab_out, stringify, assert_valid_dimensions, assert_positive, assert_non_negative,
                        get_lp_problem_sizes)
from .base import BaseSolver


def execute_solution_from_callable(
        element_index: int,
        element_data: ElementData,
        modify_constraints: Callable[[int, ElementSolver], None],
) -> Tuple[float, Dict[str, List[float]]]:
    """Execute the solution from a callable."""

    modify_constraints(element_index, (element_solver := new_element_solver(element_data)))
    return element_solver.solve()


class CenterSolver(BaseSolver[CenterData]):
    """Base class for all center's solvers."""

    def __init__(self, data: CenterData):
        super().__init__(data)

        self.element_solutions: List[Tuple[float, Dict[str, List[float]]]] = list()
        self.element_solvers: List[ElementSolver] = list()
        self.order = get_order(get_lp_problem_sizes(data.elements), data.config.num_threads)
        self.parallel_executor = ParallelExecutor(
            min_threshold=data.config.min_parallelisation_threshold,
            num_threads=data.config.num_threads,
            order=self.order,
        )

    @abstractmethod
    def modify_constraints(self, element_index, element_solver) -> None:
        """Add constraints to the element's solver."""

        pass

    def quality_functional(self) -> Tuple[str, float]:
        """Calculate the center's quality functional: sum_e (d_e^T * y_e)."""

        sums = [sum(d_e * y_e for d_e, y_e in zip(self.data.coeffs_functional[e], sol[1]["y_e"]))
                for e, sol in enumerate(self.element_solutions) if sol is not None]
        return stringify(sums), sum(sums)

    def coordinate(self) -> None:
        """Coordinates the optimization problem."""

        if self.setup_done:
            return

        self.element_solutions = self.parallel_executor.execute(
            [partial(execute_solution_from_callable, e, element_data, self.modify_constraints)
             for e, element_data in enumerate(self.data.elements)])

        self.setup_done = True

    def print_results(self) -> None:
        """Print the results of the optimization for the center problem."""

        if not self.setup_done:
            raise RuntimeError("The optimization problem has not been set up yet. Call coordinate() first.")

        tab_out(f"\nInput data for center {stringify(self.data.config.id)}", (
            ("Center Type", stringify(self.data.config.type)),
            ("Center ID", stringify(self.data.config.id)),
            ("Center Number of Elements", stringify(self.data.config.num_elements)),
            ("Center Functional Coefficients", stringify(self.data.coeffs_functional)),
            ("Center Min Parallelization Threshold", stringify(self.data.config.min_parallelisation_threshold)),
            ("Center Number of Threads", stringify(self.data.config.num_threads)),
            ("Center Parallelization Order", stringify(self.order)),
        ))

        if len(self.element_solvers) != len(self.element_solutions):
            for e, (solution, element_data) in enumerate(zip(self.element_solutions, self.data.elements)):
                solver_e = new_element_solver(element_data)
                solver_e.set_solution(solution)
                solver_e.setup()
                self.element_solvers.append(solver_e)

        for solver_e in self.element_solvers:
            solver_e.print_results()

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
