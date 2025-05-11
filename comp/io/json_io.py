from dataclasses import is_dataclass, fields
from enum import Enum
from json import load
from typing import Type, TypeVar

from numpy import ndarray, array

from comp.models import CenterConfig, CenterData, ElementConfig, ElementData

T_dataclass = TypeVar("T_dataclass")


def _parse_dataclass(cls: Type[T_dataclass], data: dict) -> T_dataclass:
    """
    Parse a dictionary into a dataclass instance, handling Enums and nested dataclasses.

    :param cls: The dataclass type to parse into.
    :param data: Dictionary containing the data to parse.
    :return: An instance of the dataclass populated with the parsed data.
    """

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = dict()
    for name, field_type in field_types.items():
        if name not in data:
            continue
        value = data[name]
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            kwargs[name] = field_type[value]
        elif is_dataclass(field_type):
            kwargs[name] = _parse_dataclass(field_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _parse_element_data(data: dict) -> ElementData:
    """
    Parse element data from a dictionary, converting lists to numpy arrays.

    :param data: Dictionary containing element data.
    :return: ElementData object containing the parsed data.
    """

    def to_array(lst: list | None) -> ndarray | None:
        return array(lst, dtype=float) if lst is not None else None

    rc_raw = data.get("resource_constraints", list())
    return ElementData(
        config=_parse_dataclass(ElementConfig, data["config"]),
        coeffs_functional=to_array(data["coeffs_functional"]),
        resource_constraints=(to_array(rc_raw[0]), to_array(rc_raw[1]), to_array(rc_raw[2]),
                              ) if rc_raw else (None, None, None),
        aggregated_plan_costs=to_array(data["aggregated_plan_costs"]),
        delta=data.get("delta"),
        w=to_array(data.get("w")),
    )


def load_center_data_from_json(filepath: str) -> CenterData:
    """
    Load center data from a JSON file with custom parsing.

    :param filepath: Path to the JSON file.
    :return: CenterData object containing the loaded data.
    """

    with open(filepath, "r") as f:
        raw_data = load(f)

    return CenterData(
        config=_parse_dataclass(CenterConfig, raw_data["config"]),
        coeffs_functional=[array(cf, dtype=float) for cf in raw_data["coeffs_functional"]],
        elements=[_parse_element_data(el) for el in raw_data["elements"]],
    )
