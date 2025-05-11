from dataclasses import is_dataclass, asdict
from enum import Enum
from json import dump
from typing import Any

from numpy import ndarray, floating, isinf, isnan, integer


def _json_serializer(obj: Any) -> Any:
    """
    Serialize objects to JSON-friendly formats.
    This function handles numpy arrays, Enums, dataclasses, and other types

    :param obj: The object to serialize.
    :return: JSON-friendly representation of the object.
    :raises TypeError: If the object type is not serializable.
    """

    if isinstance(obj, ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, (float, floating)):
        if isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if isnan(obj):
            return "NaN"
        return float(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, integer):
        return int(obj)
    if isinstance(obj, (list, dict, str, int, bool)) or obj is None:
        return obj
    raise TypeError(f"Type {type(obj)} with value {obj!r} not serializable")


def save_to_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file with custom serialization.

    :param data: Data to be saved can be a dataclass, list, or dictionary.
    :param filepath: Path to the JSON file where data will be saved.
    """

    with open(filepath, "w") as f:
        dump(data, f, default=_json_serializer, indent=2)
