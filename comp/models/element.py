from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from numpy import ndarray

from .base import BaseConfig, BaseData


@dataclass(frozen=True)
class ElementSolution:
    """Solution data for an element in the system."""

    objective: float = float("-inf")
    plan: Dict[str, List[float | List[float]]] = field(default_factory=dict)


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
    resource_constraints: Tuple[Optional[ndarray], ndarray, ndarray]  # b_e, b_e_1, b_e_2
    aggregated_plan_costs: ndarray  # A_e

    delta: Optional[float] = None  # delta_e
    w: Optional[ndarray] = None  # w_e

    def copy(self) -> Self:
        """
        Create a deep copy of the ElementData instance.

        This method generates a new ElementData object with all its mutable
        attributes (numpy arrays) copied, ensuring that modifications to the
        new instance do not affect the original.
        Immutable attributes like delta and w are assigned directly.

        :return: A new ElementData instance that is a deep copy of the original.
        """

        b_e, b_e_1, b_e_2 = self.resource_constraints
        return ElementData(
            config=self.config,
            coeffs_functional=self.coeffs_functional.copy(),
            resource_constraints=(b_e.copy() if b_e is not None else None, b_e_1.copy(), b_e_2.copy()),
            aggregated_plan_costs=self.aggregated_plan_costs.copy(),
            delta=self.delta,
            w=self.w.copy() if self.w is not None else None,
        )
