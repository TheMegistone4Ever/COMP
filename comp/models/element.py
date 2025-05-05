from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from numpy import ndarray

from .base import BaseConfig, BaseData


class ElementType(Enum):
    """
    Enumeration for different types of elements in the system.

    DECENTRALIZED:
        Element independently forms its own local plan based on its own goals and constraints.
        It interacts with the center primarily through negotiation or coordination protocols.

    NEGOTIATED:
        Element forms its plan in coordination with the center,
        taking into account both its own interests and the planning instructions received from the center.
    """

    DECENTRALIZED = auto()
    NEGOTIATED = auto()


@dataclass(frozen=True)
class ElementConfig(BaseConfig):
    """Configuration data for an element in the system."""

    type: ElementType

    num_decision_variables: int  # n_e
    num_constraints: int  # m_e


@dataclass(frozen=True)
class ElementData(BaseData):
    """Data container for element-specific optimization parameters."""

    config: ElementConfig
    coeffs_functional: ndarray  # c_e
    resource_constraints: Tuple[ndarray, ndarray, ndarray]  # b_e, b_e_1, b_e_2
    aggregated_plan_costs: ndarray  # A_e

    delta: Optional[float]  # delta_e
    w: Optional[float]  # w_e
