from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class BaseConfig:
    """Base configuration data for the system."""

    id: int
    type: Enum


@dataclass(frozen=True)
class BaseData:
    """Base data container for system-specific optimization parameters."""

    config: BaseConfig
