from comp.models import ElementData, ElementType
from comp.solvers.core.element import ElementSolver
from comp.solvers.element import ElementLinearFirst, ElementLinearSecond


def new_element_solver(data: ElementData) -> ElementSolver:
    """
    Create a specific element solver instance based on the element type in data.

    This factory function inspects the `data.config.type` (ElementType enum)
    and returns an appropriate subclass of ElementSolver (e.g., ElementLinearFirst,
    ElementLinearSecond).

    :param data: The ElementData object containing the configuration, including the element type.
    :raises ValueError: If the `data.config.type` is unknown or not supported.
    :return: An instance of a concrete ElementSolver subclass.
    """

    if data.config.type == ElementType.DECENTRALIZED:
        return ElementLinearFirst(data)
    elif data.config.type == ElementType.NEGOTIATED:
        return ElementLinearSecond(data)
    else:
        raise ValueError(f"Unknown element type for factory: {data.config.type}")


def execute_new_solver_from_data(element_data: ElementData) -> float:
    """
    Create an element solver from data, solve it, and return its objective value.

    This utility function simplifies the process of creating an element solver using
    `new_element_solver`, setting it up, solving it, and then extracting the
    primary objective value from the solution.

    :param element_data: The ElementData object for the element to be solved.
    :return: The objective value (float) of the solved element problem.
    """

    solver = new_element_solver(element_data)
    solver.setup()
    return solver.solve()[0]
