from functools import partial
from typing import Dict, List, Optional, Any
from typing import Tuple

from numpy import ndarray
from ortools.linear_solver.pywraplp import Solver, Variable

from comp.models import CenterData, ElementType
from comp.models import ElementSolution
from comp.solvers.core import CenterSolver
from comp.solvers.core.element import ElementSolver
from comp.utils import lp_sum, stringify, tab_out


def _calculate_element_own_quality(coeffs_functional: ndarray, type_e: ElementType, y_e: List[float],
                                   y_star_e: Optional[List[float]] = None) -> float:
    """
    Calculate the element’s own quality functional value based on its type and solution.
    For NEGOTIATED elements, the quality is c_e^T * y_star_e.
    For DECENTRALIZED elements, the quality is c_e^T * y_e.

    :param coeffs_functional: Coefficients of the functional for the element.
    :param type_e: The type of the element (DECENTRALIZED or NEGOTIATED).
    :param y_e: The decision variable values for the element.
    :param y_star_e: The private decision variable values for NEGOTIATED elements, if applicable.
    :return: The calculated quality functional value as a float.
    """

    return sum(c * y for c, y in zip(coeffs_functional, y_star_e if type_e == ElementType.NEGOTIATED else y_e))


class CenterLinkedFirst(CenterSolver):
    def __init__(self, data: CenterData) -> None:
        super().__init__(data)

        self.solver = Solver.CreateSolver("GLOP")
        self.solved: bool = False
        self.status: int = -1
        self.solution: Optional[ElementSolution] = None
        self.y: List[List[Variable]] = [list() for _ in range(self.data.config.num_elements)]
        self.y_star: List[List[Variable]] = [list() for _ in range(self.data.config.num_elements)]
        self.b: List[List[Variable]] = [list() for _ in range(self.data.config.num_elements)]

    def modify_constraints(self, element_index: int, element_solver: ElementSolver) -> None:
        pass

    def coordinate(self, tolerance: float = 1e-9) -> None:
        self.setup()
        self.solve()

        self.element_solutions = self.parallel_executor.execute(
            [partial(
                lambda e, data: ElementSolution(
                    objective=_calculate_element_own_quality(
                        data.coeffs_functional, data.config.type,
                        y_e := self.solution.plan.get("y")[e] if self.solution.plan.get("y") else list(),
                        y_star_e := self.solution.plan.get("y_star")[e] if self.solution.plan.get("y_star") else None
                    ),
                    plan={
                        "y_e": y_e,
                        "y_star_e": y_star_e,
                        "b_e": self.solution.plan.get("b")[e] if self.solution.plan.get("b") else list()
                    }
                ),
                e, element_data
            )
                for e, element_data in enumerate(self.data.elements)]
        )

    def setup_constraints(self) -> None:
        # Resource constraints: sum of b_e <= b
        for nc in range(self.data.elements[0].config.num_constraints):
            self.solver.Add(
                lp_sum(b_e[nc] for b_e in self.b) <= self.data.global_resource_constraints[nc]
            )

        for e, (element) in enumerate(self.data.elements):
            if element.config.type == ElementType.DECENTRALIZED:
                # Resource constraints: A_e * y_e <= b_e
                for i in range(element.config.num_constraints):
                    self.solver.Add(
                        lp_sum(element.aggregated_plan_costs[i][j] * self.y[e][j]
                               for j in range(element.config.num_decision_variables))
                        <= self.b[e][i]
                    )

                # Resource constraints: 0 <= b_e_1 <= y_e <= b_e_2
                for i in range(element.config.num_decision_variables):
                    self.solver.Add(
                        element.resource_constraints[1][i] <= self.y[e][i]
                    )
                    self.solver.Add(
                        self.y[e][i] <= element.resource_constraints[2][i]
                    )

                # Optimality Inequality Constraint: c_e^T * y_e >= f_e * (1 - delta_e)
                self.solver.Add(
                    lp_sum(element.coeffs_functional[i] * self.y[e][i]
                           for i in range(element.config.num_decision_variables))
                    >= self.data.f[e] * (1 - element.delta)
                )
            else:
                # Resource constraints: A_e * (y_e + y_star_e) <= b_e
                for i in range(element.config.num_constraints):
                    self.solver.Add(
                        lp_sum(element.aggregated_plan_costs[i][j] * (self.y[e][j] + self.y_star[e][j])
                               for j in range(element.config.num_decision_variables))
                        <= self.b[e][i]
                    )

                # Resource constraints: 0 <= b_e_1 <= y_e
                for i in range(element.config.num_decision_variables):
                    self.solver.Add(
                        element.resource_constraints[1][i] <= self.y[e][i]
                    )

                # Resource constraints: y_e + y_star_e <= b_e_2
                for i in range(element.config.num_decision_variables):
                    self.solver.Add(
                        self.y[e][i] + self.y_star[e][i] <= element.resource_constraints[2][i]
                    )

                # Optimality Inequality Constraint: c_e^T * y_star_e >= f_e * (1 - delta_e)
                self.solver.Add(
                    lp_sum(element.coeffs_functional[i] * self.y_star[e][i]
                           for i in range(element.config.num_decision_variables))
                    >= self.data.f[e] * (1 - element.delta)
                )

    def setup_objective(self) -> None:
        """
        Set up the objective function for the first linked model.

        Maximize sum of d_e^T * y_e.
        """

        objective = self.solver.Objective()

        for e, (y_e) in enumerate(self.y):
            for i, (coeff_func) in enumerate(self.data.coeffs_functional[e]):
                objective.SetCoefficient(
                    y_e[i],
                    float(coeff_func)
                )

        objective.SetMaximization()

    def setup_variables(self) -> None:
        """
        Set up optimization variables for the first linked model.

        Creates decision variables for each element’s b_0, y, and y_star.
        """

        for i, (element) in enumerate(self.data.elements):
            self.b[i] = [
                self.solver.NumVar(0, self.solver.infinity(), f"b_{self.data.config.id}_{i}_{j}")
                for j in range(element.config.num_constraints)
            ]

            self.y[i] = [
                self.solver.NumVar(0, self.solver.infinity(), f"y_{self.data.config.id}_{i}_{j}")
                for j in range(element.config.num_decision_variables)
            ]

            if element.config.type == ElementType.NEGOTIATED:
                self.y_star[i] = [
                    self.solver.NumVar(0, self.solver.infinity(), f"y_star_{self.data.config.id}_{i}_{j}")
                    for j in range(element.config.num_decision_variables)
                ]

    def setup(self, set_variables: bool = True, set_constraints: bool = True, set_objective: bool = True) -> None:
        if self.setup_done:
            return

        if set_variables:
            self.setup_variables()
        if set_constraints:
            self.setup_constraints()
        if set_objective:
            self.setup_objective()

        self.setup_done = True

    def solve(self) -> ElementSolution:
        """
        Solve the optimization problem for the element.

        If the problem has not been set up, it raises a RuntimeError.
        If not already solved, it calls the OR-Tools solver.
        If an optimal solution is found, it stores and returns the objective value and solution variables.
        Otherwise, it returns infinity and an empty dictionary.

        :raises RuntimeError: If `setup()` has not been called first.
        :return: A tuple containing the objective value (float, or float("-inf") if no solution)
                 and a dictionary of solution variables (Dict[str, List[float]]).
        """

        if not self.setup_done:
            raise RuntimeError("Solver setup is not done. Call setup() before solve().")

        if not self.solved:
            self.solved = True
            self.status = self.solver.Solve()
            if self.status in (Solver.OPTIMAL, Solver.FEASIBLE):
                self.solution = ElementSolution(self.solver.Objective().Value(), {
                    "y": [[v.solution_value() for v in y_e] for y_e in self.y],
                    "y_star": [[v.solution_value() for v in y_star_e] for y_star_e in self.y_star],
                    "b": [[v.solution_value() for v in b_e] for b_e in self.b]
                })
            else:
                self.solution = ElementSolution()

        return self.solution

    def quality_functional(self) -> Tuple[str, float]:
        sums = [sum(d * y for d, y in zip(self.data.coeffs_functional[e], sol))
                for e, sol in enumerate(self.solution.plan.get("y")) if sol is not None]
        return stringify(sums), sum(sums)

    def print_results(self, print_details: bool = True, tolerance: float = 1e-9) -> None:
        super().print_results(False)

        if not print_details:
            return

        self._populate_element_solvers()
        for solver_e in self.element_solvers:
            if self.solution.objective == float("-inf") and not self.solution.plan:
                print(f"\nNo optimal solution found for center {self.data.config.id}.")
                return

            input_data = [
                ("Element Type", stringify(solver_e.data.config.type)),
                ("Element ID", stringify(solver_e.data.config.id)),
                ("Element Number of Decision Variables", stringify(solver_e.data.config.num_decision_variables)),
                ("Element Number of Constraints", stringify(solver_e.data.config.num_constraints)),
                ("Element Functional Coefficients", stringify(solver_e.data.coeffs_functional)),
                ("Element Resource Constraints", stringify(solver_e.data.resource_constraints)),
                ("Element Aggregated Plan Costs", stringify(solver_e.data.aggregated_plan_costs)),
                ("Element Delta", stringify(solver_e.data.delta)),
            ]

            tab_out(f"\nInput data for element {stringify(solver_e.data.config.id)}", input_data)

            print(f"\nElement {stringify(solver_e.data.config.id)} "
                  f"quality functional: {stringify(solver_e.quality_functional())}")

            optimization_data = [
                ("Resource Constraints", stringify((dict_solved := solver_e.solve().plan)["b_e"])),
                ("Decision Variables", stringify(dict_solved["y_e"])),
            ]
            if solver_e.data.config.type == ElementType.NEGOTIATED:
                optimization_data.append(
                    ("Private Decision Variables", stringify(dict_solved["y_star_e"]))
                )

            tab_out(f"Optimization results for element {stringify(solver_e.data.config.id)}", optimization_data)

    def get_results_dict(self, tolerance: float = 1e-9) -> Dict[str, Any]:
        """
        Get a dictionary representation of the center optimization results.

        :param tolerance: Tolerance for checking optimality.
        :return: A dictionary containing the center’s ID, type, number of elements,
                 and the solution if available.
        """

        if not self.setup_done:
            self.coordinate()

        base_results = super().get_results_dict()

        base_results.update({
            "status": self.status,
            "solution": {
                "objective_value": self.solution.objective if self.solution else None,
                "b": self.solution.plan.get("b") if self.solution else None,
            }
        })

        return base_results
