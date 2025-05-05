from comp.models import CenterType, CenterData, ElementData, ElementType
from comp.solvers.core.base import BaseSolver


def element_solver_fabric(data: ElementData) -> BaseSolver:
    """
    Factory function to create an element solver based on the specified type.

    Args:
        data: The data associated with the element.

    Returns:
        An instance of the appropriate solver class based on the element type.
    """

    from comp.solvers import ElementLinearFirst, ElementLinearSecond

    if data.config.type == ElementType.DECENTRALIZED:
        return ElementLinearFirst(data)
    elif data.config.type == ElementType.NEGOTIATED:
        return ElementLinearSecond(data)
    else:
        raise ValueError(f"Unknown element type for factory: {data.config.type}")


def center_solver_fabric(data: CenterData) -> BaseSolver:
    """
    Factory function to create a center solver based on the specified type.

    Args:
        data: The data associated with the center.

    Returns:
        An instance of the appropriate solver class based on the center type.
    """

    from comp.solvers import CenterLinearFirst

    if data.config.type == CenterType.STRICT_PRIORITY:
        return CenterLinearFirst(data)
    elif data.config.type == CenterType.GUARANTEED_CONCESSION:
        raise NotImplementedError("Guaranteed Concession strategy is not implemented yet.")
    elif data.config.type == CenterType.WEIGHTED_BALANCE:
        raise NotImplementedError("Weighted Balance strategy is not implemented yet.")
    else:
        raise ValueError(f"Unknown center type: {data.config.type}")
