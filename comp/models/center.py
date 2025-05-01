from dataclasses import dataclass
from typing import List

from numpy import ndarray

from .element import ElementData


@dataclass(frozen=True)
class CenterConfig:
    """Configuration data for the system center."""

    id: int
    num_elements: int  # m


@dataclass(frozen=True)
class CenterData:
    """Data container for center-specific optimization parameters."""

    config: CenterConfig
    coeffs_functional: List[ndarray]  # d
    elements: List[ElementData]
