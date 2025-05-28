from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from numpy import ndarray

from comp.utils.json_base_serializer import save_to_json as global_save_json_util
from .base import BaseConfig, BaseData
from .element import ElementData


class CenterType(Enum):
    """
    Enumeration for different types of center coordination strategies.

    STRICT_PRIORITY:
        The center prioritizes its own goals by selecting plans
        that maximize its goal function (d_l^T * y_l),
        and then chooses among these the one most favorable to the element (by maximizing c_l^T * y_l).

    GUARANTEED_CONCESSION:
        The center allows for a controlled concession to the element
        by ensuring the element’s goal (c_l^T * y_l) reaches at least a given proportion
        of its optimal value ((1 - Δ_l) * f_opt_element).

    WEIGHTED_BALANCE:
        The center applies a weighted compromise strategy,
        balancing its own goal and the element’s objective using a positive weight coefficient (ω_l).
        This approach enables iterative adjustment toward a mutually acceptable solution.

    RESOURCE_ALLOCATION_COMPROMISE:
        The center solves a single, coupled optimization problem to allocate a global resource budget
        It aims to maximize its own overall goal (sum of d_l^T * y_l) while ensuring that
        each element meets a predefined target for its own goal like in "GUARANTEED_CONCESSION".
    """

    STRICT_PRIORITY = auto()
    GUARANTEED_CONCESSION = auto()
    WEIGHTED_BALANCE = auto()
    RESOURCE_ALLOCATION_COMPROMISE = auto()


@dataclass(frozen=True)
class CenterConfig(BaseConfig):
    """Configuration data for the system center."""

    min_parallelisation_threshold: Optional[int]
    num_threads: int

    type: CenterType

    num_elements: int  # m


@dataclass(frozen=True)
class CenterData(BaseData):
    """Data container for center-specific optimization parameters."""

    config: CenterConfig
    coeffs_functional: List[ndarray]  # d
    elements: List[ElementData]

    global_resource_constraints: Optional[ndarray] = None  # b

    def save_to_json(self, filepath: str) -> None:
        global_save_json_util(self, filepath)
