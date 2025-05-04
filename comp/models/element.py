from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional

from numpy import ndarray


class ElementType(Enum):
    """Enumeration for different types of elements in the system."""

    COMBINATORIAL = auto()
    LINEAR = auto()


@dataclass(frozen=True)
class ElementConfig:
    """Configuration data for an element in the system."""

    type: ElementType

    id: int
    num_decision_variables: int  # n_e
    num_constraints: int  # m_e

    num_schedules: int  # n_k


@dataclass(frozen=True)
class ElementData:
    """Data container for element-specific optimization parameters."""

    config: ElementConfig
    coeffs_functional: ndarray  # c_e
    resource_constraints: Tuple[ndarray, ndarray, ndarray]  # b_e, b_e_1, b_e_2
    aggregated_plan_costs: ndarray  # A_e

    delta: Optional[float]  # delta_e

    schedules: Optional[ndarray]  # sigma_e
    interest: Optional[ndarray]  # alpha_e
    weight_coefficients: Optional[ndarray]  # w_e
