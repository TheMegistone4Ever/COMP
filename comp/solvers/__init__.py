from comp.models import CenterData, CenterType
from .center import CenterLinearFirst, CenterLinearSecond, CenterLinearThird
from .core import BaseSolver, CenterSolver, ElementSolver
from .factories import new_element_solver


def new_center_solver(data: CenterData) -> CenterSolver:
    """
    Create a specific center solver instance based on the center type in data.

    This factory function inspects the `data.config.type` (CenterType enum)
    and returns an appropriate subclass of CenterSolver (e.g., CenterLinearFirst,
    CenterLinearSecond, CenterLinearThird).

    :param data: The CenterData object containing the configuration, including the center type.
    :raises ValueError: If the `data.config.type` is unknown or not supported.
    :return: An instance of a concrete CenterSolver subclass.
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
    "new_element_solver",
    "new_center_solver",
]
