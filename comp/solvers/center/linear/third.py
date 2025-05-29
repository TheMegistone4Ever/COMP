from dataclasses import replace
from functools import partial
from typing import Dict, List, Optional, Any

from comp.models import CenterData, ElementData, ElementSolution, ElementType
from comp.solvers.core import CenterSolver
from comp.solvers.core.element import ElementSolver
from comp.solvers.factories import new_element_solver, execute_new_solver_from_data
from comp.utils import stringify, tab_out


def _calculate_element_own_quality(element_data: ElementData, solution_data: ElementSolution) -> float:
    """
    Calculate the element’s own quality functional value based on its type and solution.
    For NEGOTIATED elements, the quality is c_e^T * y_star_e.
    For DECENTRALIZED elements, the quality is c_e^T * y_e.

    :param element_data: The ElementData instance contains configuration and coefficients.
    :param solution_data: The ElementSolution instance containing the solved plan.
    :return: The calculated quality functional value as a float.
    """

    return sum(c * y for c, y in zip(element_data.coeffs_functional, solution_data.plan.get("y_star_e")
    if element_data.config.type == ElementType.NEGOTIATED else solution_data.plan.get("y_e")))


class CenterLinearThird(CenterSolver):
    """Solver for center-level optimization problems. 1’st linear model."""

    def __init__(self, data: CenterData) -> None:
        """
        Initialize the CenterLinearThird solver.

        This solver implements a weighted balance strategy.
        It iterates through various weight (w) values for each element, solving the element’s problem
        with a combined goal (center’s objective + w * element’s objective).
        It then selects the `w` for each element that maximizes the element’s own objective function.

        Initializes attributes to store solutions for all `w` values and the chosen solution
        for each element.

        :param data: The CenterData object containing configuration and parameters for the center.
        """

        super().__init__(data)

        self.f_c_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, replace(
            element_data, coeffs_functional=data.coeffs_functional[e], config=replace(
                element_data.config, type=ElementType.DECENTRALIZED))) for e, element_data in enumerate(data.elements)])
        self.f_el_opt = self.parallel_executor.execute([partial(execute_new_solver_from_data, element_data.copy())
                                                        for element_data in data.elements])
        self.all_element_solutions: List[Dict[float, ElementSolution]] = [dict() for _ in data.elements]
        self.chosen_element_solutions_info = [(.0, ElementSolution()) for _ in data.elements]

    def _modify_element_objective_with_w(self, e: int, element_solver: ElementSolver, w_scalar: float) -> None:
        """
        Modify the objective function of an element’s solver to incorporate a weight `w_scalar`.

        The objective becomes: Max (d_e^T * y_e + w_scalar * c_e^T * y_plan_component).
        - d_e are the center’s coefficients for element `e`.
        - c_e are the element’s own coefficients.
        - y_e are the element’s decision variables relevant to the center’s part of the objective.
        - y_plan_component are the element’s decision variables relevant to its own part of the objective
          (y_e for DECENTRALIZED, y_star_e for NEGOTIATED).

        :param e: The index of the element.
        :param element_solver: The ElementSolver instance for the specific element.
        :param w_scalar: The weight coefficient (w_e) to apply.
        """

        if not element_solver.setup_done:
            element_solver.setup(set_objective=False)

        element_objective = element_solver.solver.Objective()

        # Objective: Max (d_e^T * y_e + w_e * c_e^T * y_e)
        if self.data.elements[e].config.type == ElementType.DECENTRALIZED:
            # (d_e^T + w_e * c_e^T) * y_e
            for i, (center_coeff, element_coeff) in enumerate(zip(self.data.coeffs_functional[e],
                                                                  element_solver.data.coeffs_functional)):
                element_objective.SetCoefficient(
                    element_solver.get_plan_component(i),
                    float(center_coeff + w_scalar * element_coeff)
                )
        else:
            # d_e^T * y_e
            for i, (coeff_func) in enumerate(self.data.coeffs_functional[e]):
                element_objective.SetCoefficient(
                    element_solver.y_e[i],
                    float(coeff_func)
                )

            # w_e * c_e^T * y_star_e
            for i, (coeff_func) in enumerate(element_solver.data.coeffs_functional):
                element_objective.SetCoefficient(
                    element_solver.get_plan_component(i),
                    float(coeff_func * w_scalar)
                )

        element_objective.SetMaximization()

    def modify_constraints(self, e: int, element_solver: ElementSolver) -> None:
        """
        This method fulfills the `CenterSolver` abstract interface requirement.
        For `CenterLinearThird`’s specific weighted balance strategy, the primary logic for
        modifying element goals with various weights `w` is handled within its overridden
        `coordinate` method, which uses `_solve_element_for_specific_w` and, in turn,
        `_modify_element_objective_with_w`.

        This `modify_constraints` implementation is generally not invoked during the main
        coordination flow of `CenterLinearThird` because `CenterLinearThird.coordinate()`
        does not rely on the base class’s `coordinate` method (which would call this).
        If it were called directly, it would configure the `element_solver` to optimize
        based on the center’s coefficients for that element, corresponding to a
        scenario where the weight for the element’s own goal is zero.

        :param e: The index of the element.
        :param element_solver: The ElementSolver instance for the specific element.
        """

        pass

    def _solve_element_for_specific_w(self, e: int, element_data_copy: ElementData, w_scalar: float) -> ElementSolution:
        """
        Create and solve an element’s optimization problem for a specific weight `w_scalar`.

        A new element solver is instantiated, its goal is modified using
        `_modify_element_objective_with_w`, and then it’s solved.

        :param e: The index of the element.
        :param element_data_copy: A copy of the ElementData for the element.
                                  A copy is used to avoid side effects if the data is modified.
        :param w_scalar: The weight coefficient (w_e) to apply.
        :return: The ElementSolution obtained by solving the element’s problem with the given `w_scalar`.
        """

        element_solver = new_element_solver(element_data_copy)
        self._modify_element_objective_with_w(e, element_solver, w_scalar)
        return element_solver.solve()

    def coordinate(self, tolerance: float = 1e-9) -> None:
        """
        Coordinate the optimization process for all elements using the weighted balance strategy.

        If not already set up, this method performs the following steps:
        1. For each element and for each weight `w` specified in `element_data.w`:
           A. Creates a task to solve the element’s subproblem with that `w`.
           B. The subproblem’s objective is Max (d_e^T * y_e + w * c_e^T * y_plan_component).
        2. Execute these tasks, potentially in parallel.
        3. Stores all solutions (for each `w`) in `self.all_element_solutions`.
        4. For each element:
           A. Iterates through its solutions obtained for different `w` values.
           B. Select the `w` and corresponding solution that maximizes the element’s
              own quality functional (c_e^T * y_plan_component), calculated by
              `_calculate_element_own_quality`.
           C. Stores best selected `w` and solution in `self.chosen_element_solutions_info`.
           D. Appends the chosen `ElementSolution` to `self.element_solutions` (used by base class).
        5. It marks the setup as done.

        :param tolerance: The tolerance for comparing floating-point numbers.
        """

        if self.setup_done:
            return

        tasks, task_identifiers = list(), list()

        for e, element_data in enumerate(self.data.elements):
            if element_data.w is None or len(element_data.w) == 0:
                self.all_element_solutions[e] = dict()
                continue

            for w_val_np in element_data.w:
                task_identifiers.append((e, w_scalar := float(w_val_np)))
                tasks.append(partial(self._solve_element_for_specific_w, e, element_data.copy(), w_scalar))

        for (e, w_scalar), solution in zip(task_identifiers, self.parallel_executor.execute(tasks)):
            if solution is not None:
                self.all_element_solutions[e][w_scalar] = solution

        for e in range(len(self.data.elements)):
            best_w_for_element: Optional[float] = None
            max_element_qf_e: float = float("-inf")
            current_best_solution_tuple: ElementSolution = ElementSolution()

            if not self.all_element_solutions[e]:
                self.chosen_element_solutions_info[e] = (.0, ElementSolution())
                self.element_solutions.append(ElementSolution())
                continue

            for w_val in sorted(self.all_element_solutions[e].keys()):
                sol_info_tuple = self.all_element_solutions[e][w_val]

                if sol_info_tuple is None or not sol_info_tuple.plan:
                    continue

                element_qf_e = _calculate_element_own_quality(self.data.elements[e], sol_info_tuple)

                if (element_qf_e - max_element_qf_e) > tolerance:
                    max_element_qf_e = element_qf_e
                    best_w_for_element = w_val
                    current_best_solution_tuple = sol_info_tuple

            if best_w_for_element is not None:
                self.chosen_element_solutions_info[e] = (best_w_for_element, current_best_solution_tuple)
                self.element_solutions.append(current_best_solution_tuple)
            else:
                self.chosen_element_solutions_info[e] = (.0, ElementSolution())
                self.element_solutions.append(ElementSolution())

        self.setup_done = True

    def print_results(self, print_details: bool = True, tolerance: float = 1e-9) -> None:
        """
        Print the comprehensive results of the center’s optimization problem for the weighted balance strategy.

        This extends the base `CenterSolver.print_results()` method.
        For each element, it prints:
        - A table showing results for all evaluated `w` values, including:
            - `w` value
            - Element’s functional value (c^T * y_plan_component)
            - Center’s contribution to the objective (d^T * y_e)
            - Combined objective value (d^T * y_e + w * c^T * y_plan_component)
            - An indicator (*) for the chosen `w` value.
        - Detailed information about the solution chosen for that element:
            - Chosen `w`
            - Chosen plan (y_e)
            - Chosen element functional value
            - Chosen center contribution
            - Chosen combined objective value

        :param print_details: If True, print additional details about the optimization results.
        :param tolerance: The tolerance for comparing floating-point numbers.
        """

        super().print_results(print_details)

        for e, element_data in enumerate(self.data.elements):
            chosen_w, chosen_solution_info = self.chosen_element_solutions_info[e]

            print(f"\nElement {stringify(element_data.config.id)} (Type: {element_data.config.type}):")
            print(f"\nElement Optimal (f_el_opt): {stringify(self.f_el_opt[e])}")
            print(f"Center Optimal (f_c_opt): {stringify(self.f_c_opt[e])}")
            if not self.all_element_solutions[e]:
                print(f"No solutions found for any w value for element {element_data.config.id}.")
                continue

            results_table_data = list()
            for w_val in sorted(self.all_element_solutions[e].keys()):
                sol_info = self.all_element_solutions[e][w_val]
                elem_func_str, center_contr_str = "N/A", "N/A"
                obj_str = stringify(combined_obj) if (combined_obj := sol_info.objective) != float("-inf") else "N/A"
                if sol_info.plan.get("y_e"):
                    elem_func = _calculate_element_own_quality(element_data, sol_info)
                    elem_func_str, center_contr_str, obj_str = map(
                        stringify, (elem_func, combined_obj - w_val * elem_func, combined_obj))
                results_table_data.append([stringify(w_val), elem_func_str, center_contr_str, obj_str,
                                           "*" if abs(w_val - chosen_w) < tolerance else ""])

            tab_out(f"Results for Element {element_data.config.id} across w values", results_table_data,
                    ["w", "Elem QF\n(c^T y)", "Center QF\n(d^T y)", "Combined Obj\n(d^T y + w*c^T y)",
                     "Chosen (based\non max c^T y)"])

            if chosen_solution_info and chosen_solution_info.plan.get("y_e"):
                print(f"Chosen w: {stringify(chosen_w)}")
                print(f"Chosen Plan: {stringify(chosen_solution_info.plan)}")
                if chosen_solution_info.objective != float("-inf"):
                    chosen_elem_func = _calculate_element_own_quality(element_data, chosen_solution_info)
                    chosen_center_contrib = chosen_solution_info.objective - chosen_w * chosen_elem_func
                    print(f"Chosen Element Functional (c^T y_plan): {stringify(chosen_elem_func)}")
                    print(f"Chosen Center Contribution (d^T y_e): {stringify(chosen_center_contrib)}")
                    print(f"Chosen Combined Objective: {stringify(chosen_solution_info.objective)}")
                else:
                    print(f"The chosen solution for w={stringify(chosen_w)} is not optimal or failed.")
            else:
                print("No optimal solution found for the chosen w value.")

    def get_results_dict(self, tolerance: float = 1e-9) -> Dict[str, Any]:
        """
        Get the results of the center optimization problem as a dictionary.

        The results are structured as follows:
        - center_optimal_functional_value: The optimal functional value of the center.
        - element_optimal_functional_value: The optimal functional value of the elements.
        - center_type_specific_results: A dictionary containing the strategy used and detailed results for each element.
        - strategy: The strategy used for the center optimization.
        - weighted_balance_analysis: A list of dictionaries, each containing:
            - element_id: The ID of the element.
            - element_type: The type of the element.
            - f_el_opt: The optimal functional value of the element.
            - f_c_opt: The optimal functional value of the center.
            - solutions_by_w: A dictionary of solutions for each w value, including:
                - element_qf: The element’s own quality functional value.
                - center_qf: The center’s contribution to the objective.
                - combined_objective: The combined objective value.
                - is_chosen: A boolean indicating if this solution was chosen.
                - solution_plan: The plan associated with this solution.
            - chosen_solution_details: A dictionary containing details of the chosen solution, including:
                - chosen_w: The chosen weight (w) value.
                - plan: The chosen plan.
                - element_qf: The element’s own quality functional value for the chosen solution.
                - center_qf: The center’s contribution to the objective for the chosen solution.
                - combined_objective: The combined objective value for the chosen solution.

        :param tolerance: The tolerance for comparing floating-point numbers.
        :return: A dictionary containing the results of the center optimization problem.
        """

        if not self.setup_done:
            self.coordinate()

        base_results = super().get_results_dict()

        specific_results = list()
        for e, element_data in enumerate(self.data.elements):
            chosen_w, chosen_solution_info = self.chosen_element_solutions_info[e]

            all_w_solutions_payload = dict()
            if self.all_element_solutions[e]:
                for w, sol_info in self.all_element_solutions[e].items():
                    elem_qf, obj_s = float("-inf"), sol_info.objective if sol_info else float("-inf")
                    if sol_info.plan.get("y_e"):
                        elem_qf = _calculate_element_own_quality(element_data, sol_info)
                    all_w_solutions_payload[w] = {
                        "element_qf": elem_qf,
                        "center_qf": obj_s - w * elem_qf,
                        "combined_objective": obj_s,
                        "is_chosen": abs(w - chosen_w) < tolerance if chosen_w is not None else False,
                        "solution_plan": sol_info.plan if sol_info else None,
                    }

            chosen_w_payload = None
            if chosen_solution_info and chosen_solution_info.plan:
                chosen_element_qf = _calculate_element_own_quality(element_data, chosen_solution_info)
                chosen_center_qf = float("-inf")
                if chosen_solution_info.plan.get("y_e"):
                    chosen_center_qf = chosen_solution_info.objective - chosen_w * chosen_element_qf
                chosen_w_payload = {
                    "chosen_w": chosen_w,
                    "plan": chosen_solution_info.plan,
                    "element_qf": chosen_element_qf,
                    "center_qf": chosen_center_qf,
                    "combined_objective": chosen_solution_info.objective,
                }

            specific_results.append({
                "element_id": element_data.config.id,
                "element_type": element_data.config.type.name,
                "f_el_opt": self.f_el_opt[e] if self.f_el_opt else "N/A",
                "f_c_opt": self.f_c_opt[e] if self.f_c_opt else "N/A",
                "solutions_by_w": all_w_solutions_payload,
                "chosen_solution_details": chosen_w_payload,
            })

        base_results["center_type_specific_results"] = {
            "strategy": "WEIGHTED_BALANCE",
            "weighted_balance_analysis": specific_results,
        }

        return base_results
