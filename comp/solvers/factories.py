from comp.models import ElementData, ElementType
from comp.solvers.core import ElementSolver
from comp.solvers.element import ElementLinearFirst, ElementLinearSecond


def element_solver_fabric(data: ElementData) -> ElementSolver:
    """
    Factory function to create an element solver based on the specified type.

    Args:
        data: The data associated with the element.

    Returns:
        An instance of the appropriate solver class based on the element type.
    """

    if data.config.type == ElementType.DECENTRALIZED:
        return ElementLinearFirst(data)
    elif data.config.type == ElementType.NEGOTIATED:
        return ElementLinearSecond(data)
    else:
        raise ValueError(f"Unknown element type for factory: {data.config.type}")
