from functools import partial
from typing import Dict, List, Tuple, Optional

from comp.models import CenterData, ElementData, ElementSolutionType, ElementType
from comp.solvers.core import CenterSolver
from comp.solvers.core.element import ElementSolver
from comp.solvers.factories import new_element_solver
from comp.utils import stringify, tab_out


def _calculate_element_own_quality(element_data: ElementData, solution_data: ElementSolutionType) -> float:
    return sum(c_e * y_e for c_e, y_e in zip(element_data.coeffs_functional, solution_data[1]["y_star_e"]
    if element_data.config.type == ElementType.NEGOTIATED else solution_data[1]["y_e"]))


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

        self.element_solutions = list()
        self.all_element_solutions: List[Dict[float, ElementSolutionType]] = [dict() for _ in data.elements]
        self.chosen_element_solutions_info: List[Tuple[float, ElementSolutionType]] = [(0.0, (float('-inf'), dict()))
                                                                                       for _ in data.elements]

    def _modify_element_objective_with_w(self, e: int, element_solver: ElementSolver, w_scalar: float) -> None:
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

            # w_e * c_e^T * y_e
            for i, (coeff_func) in enumerate(element_solver.data.coeffs_functional):
                element_objective.SetCoefficient(
                    element_solver.get_plan_component(i),
                    float(coeff_func * w_scalar)
                )

        element_objective.SetMaximization()

    def modify_constraints(self, e: int, element_solver: ElementSolver) -> None:
        self._modify_element_objective_with_w(e, element_solver, .0)

    def _solve_element_for_specific_w(self, e: int, element_data_copy: ElementData,
                                      w_scalar: float) -> ElementSolutionType:
        element_solver = new_element_solver(element_data_copy)
        self._modify_element_objective_with_w(e, element_solver, w_scalar)
        return element_solver.solve()

    def coordinate(self) -> None:
        if self.setup_done:
            return

        tasks = list()
        task_identifiers = list()

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
            max_element_qf_e: float = -float('inf')
            current_best_solution_tuple: ElementSolutionType = (float('-inf'), dict())

            if not self.all_element_solutions[e]:
                self.chosen_element_solutions_info[e] = (.0, (float('-inf'), dict()))
                self.element_solutions.append((float('-inf'), dict()))
                continue

            for w_val in sorted(self.all_element_solutions[e].keys()):
                sol_info_tuple = self.all_element_solutions[e][w_val]

                if sol_info_tuple is None or not sol_info_tuple[1]:
                    continue

                element_qf_e = _calculate_element_own_quality(self.data.elements[e], sol_info_tuple)

                if (element_qf_e - max_element_qf_e) > 1e-9:
                    max_element_qf_e = element_qf_e
                    best_w_for_element = w_val
                    current_best_solution_tuple = sol_info_tuple

            if best_w_for_element is not None:
                self.chosen_element_solutions_info[e] = (best_w_for_element, current_best_solution_tuple)
                self.element_solutions.append(current_best_solution_tuple)
            else:
                self.chosen_element_solutions_info[e] = (.0, (float('-inf'), dict()))
                self.element_solutions.append((float('-inf'), dict()))

        self.setup_done = True

    def print_results(self) -> None:
        super().print_results()

        for e_idx, element_data in enumerate(self.data.elements):
            chosen_w, chosen_solution_info = self.chosen_element_solutions_info[e_idx]

            print(f"\nElement {stringify(element_data.config.id)} (Type: {element_data.config.type}):")
            if not self.all_element_solutions[e_idx]:
                print("No solutions found for any w value.")
                continue

            results_table_data = list()
            for w_val in sorted(self.all_element_solutions[e_idx].keys()):
                sol_info = self.all_element_solutions[e_idx][w_val]
                elem_func_str, center_contr_str = "N/A", "N/A"
                combined_obj_str = stringify(combined_obj) if (combined_obj := sol_info[0]) != float('inf') else "N/A"
                if sol_info[1].get("y_e"):
                    elem_func = _calculate_element_own_quality(element_data, sol_info)
                    elem_func_str, center_contr_str, combined_obj_str = map(
                        stringify, (elem_func, combined_obj - w_val * elem_func, combined_obj))
                results_table_data.append([stringify(w_val), elem_func_str, center_contr_str, combined_obj_str,
                                           "*" if abs(w_val - chosen_w) < 1e-9 else ""])

            tab_out(f"Results for Element {element_data.config.id} across w values", results_table_data,
                    ["w", "Elem Func (c^T y)", "Center Contr (d^T y)", "Combined Obj (d^T y + w*c^T y)", "Chosen"])

            if chosen_solution_info and chosen_solution_info[1] and "y_e" in chosen_solution_info[1]:
                print(f"Chosen w: {stringify(chosen_w)}")
                print(f"Chosen Plan (y_e): {stringify(chosen_solution_info[1]['y_e'])}")
                if chosen_solution_info[0] != float('-inf'):
                    chosen_elem_func = _calculate_element_own_quality(element_data, chosen_solution_info)
                    chosen_center_contrib = chosen_solution_info[0] - chosen_w * chosen_elem_func
                    print(f"Chosen Element Functional (c^T y): {stringify(chosen_elem_func)}")
                    print(f"Chosen Center Contribution (d^T y): {stringify(chosen_center_contrib)}")
                    print(f"Chosen Combined Objective: {stringify(chosen_solution_info[0])}")
                else:
                    print(f"The chosen solution is not optimal: {stringify(chosen_solution_info)}")
            else:
                print("No optimal solution found for the chosen w value.")
