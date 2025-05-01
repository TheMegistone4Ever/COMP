from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SystemConfig:
    """System-wide configuration parameters."""

    NUM_ELEMENTS: int  # m
    VEC_NUM_DECISION_VARIABLES: List[int]  # n_e
    VEC_NUM_CONSTRAINTS: List[int]  # m_e
    VEC_RESOURCE_CONSTRAINTS_LOWER: List[float]  # b_e_1
    VEC_RESOURCE_CONSTRAINTS_UPPER: List[float]  # b_e_2
