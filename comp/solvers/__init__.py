from comp.models import CenterData, CenterType
from .center import CenterLinearFirst, CenterLinearSecond, CenterLinearThird
from .core import BaseSolver, CenterSolver, ElementSolver
from .factories import element_solver_fabric


def center_solver_fabric(data: CenterData) -> CenterSolver:
    """
    Factory function to create a center solver based on the specified type.

    Args:
        data: The data associated with the center.

    Returns:
        An instance of the appropriate solver class based on the center type.
    """

    if data.config.type == CenterType.STRICT_PRIORITY:
        return CenterLinearFirst(data)
    elif data.config.type == CenterType.GUARANTEED_CONCESSION:
        return CenterLinearSecond(data)
    elif data.config.type == CenterType.WEIGHTED_BALANCE:
        return CenterLinearThird(data)
    else:
        raise ValueError(f"Unknown center type: {data.config.type}")


__all__ = [
    "BaseSolver",
    "CenterSolver",
    "ElementSolver",
    "CenterLinearFirst",
    "CenterLinearSecond",
    "CenterLinearThird",
    "element_solver_fabric",
    "center_solver_fabric",
]
