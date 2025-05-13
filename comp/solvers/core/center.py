from abc import abstractmethod
from functools import partial
from typing import Tuple, List, Callable, Dict, Any

from comp.models import CenterData, ElementData, ElementSolution
from comp.parallelization import ParallelExecutor, get_order
from comp.solvers.core.element import ElementSolver
from comp.solvers.factories import new_element_solver
from comp.utils import (assert_non_negative, assert_positive, assert_valid_dimensions, get_lp_problem_sizes,
                        stringify, tab_out, save_to_json as global_save_json_util)
from .base import BaseSolver


def execute_solution_from_callable(
        element_index: int,
        element_data: ElementData,
        modify_constraints: Callable[[int, ElementSolver], None],
) -> ElementSolution:
    """
    Create, configure, and solve an element solver, then return its solution.

    This function instantiates a new element solver using the provided element data.
    It then applies problem-specific modifications via the `modify_constraints`
    callable and solves the element’s optimization problem.

    :param element_index: The index of the element being solved.
    :param element_data: The ElementData for the specific element.
    :param modify_constraints: A callable that takes the element index and the
                               ElementSolver instance to apply specific constraints or objective modifications.
    :return: A tuple containing the objective value (float) and a dictionary
             representing the solution variables (e.g., {"y_e": [values]}).
    """

    modify_constraints(element_index, element_solver := new_element_solver(element_data))
    return element_solver.solve()


class CenterSolver(BaseSolver[CenterData]):
    """Base class for all center’s solvers."""

    def __init__(self, data: CenterData) -> None:
        """
        Initialize the CenterSolver.

        Sets up the base solver, initializes lists for element solutions and solvers,
        determines the parallelization order for elements, and creates a ParallelExecutor instance.

        :param data: The CenterData object containing configuration for the center problem.
        """

        super().__init__(data)

        self.element_solutions: List[ElementSolution] = list()
        self.element_solvers: List[ElementSolver] = list()
        self.order = get_order(get_lp_problem_sizes(data.elements), data.config.num_threads)
        self.parallel_executor = ParallelExecutor(
            min_threshold=data.config.min_parallelisation_threshold,
            num_threads=data.config.num_threads,
            order=self.order,
        )

    @abstractmethod
    def modify_constraints(self, element_index: int, element_solver: ElementSolver) -> None:
        """
        Abstract method to add center-specific constraints to an element’s solver.

        Concrete center solver implementations must provide this method to tailor
        the element’s optimization problem according to the center’s strategy.

        :param element_index: The index of the element whose solver is being modified.
        :param element_solver: The ElementSolver instance for the specific element.
        """

        pass

    def quality_functional(self) -> Tuple[str, float]:
        """
        Calculate the center’s overall quality functional.

        This is typically the sum of (d_e^T * y_e) over all elements, where d_e are
        the center’s coefficients for element e, and y_e is element e’s plan.
        Returns both a string representation of individual sums and the total sum.

        :return: A tuple containing a string representation of the sums for each element
                 and the total sum as a float.
        """

        sums = [sum(d * y for d, y in zip(self.data.coeffs_functional[e], sol.plan.get("y_e")))
                for e, sol in enumerate(self.element_solutions) if sol is not None]
        return stringify(sums), sum(sums)

    def coordinate(self) -> None:
        """
        Coordinate the optimization process for all elements.

        If not already set up, this method executes the solution process for each
        element, potentially in parallel.
        It uses the `execute_solution_from_callable` function, passing `self.modify_constraints
        ` to tailor each element’s problem.
        The results are stored in `self.element_solutions`.
        """

        if self.setup_done:
            return

        self.element_solutions = self.parallel_executor.execute(
            [partial(execute_solution_from_callable, e, element_data, self.modify_constraints)
             for e, element_data in enumerate(self.data.elements)])

        self.setup_done = True

    def print_results(self) -> None:
        """
        Print the comprehensive results of the center’s optimization problem.

        Outputs the input data for the center, its configuration, parallelization order,
        and then calls `print_results` for each element solver.
        Finally, prints the center’s own quality functional.

        :raises RuntimeError: If `coordinate()` has not been called first.
        """

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

        self._populate_element_solvers()
        for solver_e in self.element_solvers:
            solver_e.print_results()

        print(f"\nCenter {stringify(self.data.config.id)} quality functional: {stringify(self.quality_functional())}")

    def validate_input(self) -> None:
        """
        Validate the input data for the center optimization problem.

        Checks dimensions of `coeffs_functional` and `elements` against `num_elements`.
        Ensures `num_elements` is positive and `id` is non-negative.
        """

        assert_valid_dimensions(
            [self.data.elements, ],
            [(self.data.config.num_elements,), ],
            ["elements", ]
        )
        assert_positive(
            self.data.config.num_elements,
            "data.config.num_elements"
        )
        assert_non_negative(
            self.data.config.id,
            "data.config.id"
        )
        assert_positive(
            self.data.config.num_threads,
            "data.config.num_threads"
        )
        assert_positive(
            self.data.config.min_parallelisation_threshold,
            "data.config.min_parallelisation_threshold"
        )

    def get_results_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the center optimization results.

        :return: A dictionary containing the center’s ID, type, number of elements,
        """

        if not self.setup_done:
            raise RuntimeError("The optimization problem has not been coordinated yet. Call coordinate() first.")

        self._populate_element_solvers()

        center_qf_str, center_qf_val = self.quality_functional()

        return {
            "center_id": self.data.config.id,
            "center_type": self.data.config.type.name,
            "num_elements": self.data.config.num_elements,
            "parallelization_order": self.order,
            "element_results": list(map(lambda solver: solver.get_results_dict(), self.element_solvers)),
            "center_quality_functional_summary_str": center_qf_str,
            "center_quality_functional_total": center_qf_val
        }

    def save_results_to_json(self, filepath: str) -> None:
        """
        Save the results of the center optimization to a JSON file.

        :param filepath: Path to the JSON file where results will be saved.
        """

        global_save_json_util(self.get_results_dict(), filepath)

    def _populate_element_solvers(self) -> None:
        """Populate the element_solvers list with new ElementSolver instances based on the element data."""

        if len(self.element_solvers) != len(self.element_solutions):
            for e, (solution, element_data) in enumerate(zip(self.element_solutions, self.data.elements)):
                solver_e = new_element_solver(element_data)
                solver_e.set_solution(solution)
                solver_e.setup()
                self.element_solvers.append(solver_e)
