from dataclasses import is_dataclass, asdict
from enum import Enum
from json import dump, load
from typing import Any

from numpy import ndarray, isinf, isnan, int_, intc, intp, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, array

from comp.models import CenterConfig, CenterData, CenterType, ElementConfig, ElementData, ElementType


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, float):
        if isinf(obj) and obj > 0: return "Infinity"
        if isinf(obj) and obj < 0: return "-Infinity"
        if isnan(obj): return "NaN"
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (int_, intc, intp, int8,
                        int16, int32, int64, uint8,
                        uint16, uint32, uint64)):
        return int(obj)
    if isinstance(obj, (float16, float32, float64)):
        if isinf(obj) and obj > 0: return "Infinity"
        if isinf(obj) and obj < 0: return "-Infinity"
        if isnan(obj): return "NaN"
        return float(obj)

    if isinstance(obj, (list, dict, str, int, bool)) or obj is None:
        return obj

    raise TypeError(f"Type {type(obj)} with value {obj!r} not serializable")


def save_to_json(data: Any, filepath: str) -> None:
    with open(filepath, "w") as f:
        dump(data, f, default=_json_serializer, indent=2)


def _parse_element_config(data: dict) -> ElementConfig:
    return ElementConfig(
        id=data["id"],
        type=ElementType[data["type"]],
        num_decision_variables=data["num_decision_variables"],
        num_constraints=data["num_constraints"]
    )


def _parse_element_data(data: dict) -> ElementData:
    rc_raw = data["resource_constraints"]
    return ElementData(
        config=_parse_element_config(data["config"]),
        coeffs_functional=array(data["coeffs_functional"], dtype=float),  # Assuming float for coeffs
        resource_constraints=(
            array(rc_raw[0], dtype=float),
            array(rc_raw[1], dtype=float),
            array(rc_raw[2], dtype=float)
        ),
        aggregated_plan_costs=array(data["aggregated_plan_costs"], dtype=float),
        delta=data.get("delta"),
        w=array(data["w"], dtype=float) if data.get("w") is not None else None
    )


def _parse_center_config(data: dict) -> CenterConfig:
    return CenterConfig(
        id=data["id"],
        type=CenterType[data["type"]],
        min_parallelisation_threshold=data.get("min_parallelisation_threshold"),
        num_threads=data["num_threads"],
        num_elements=data["num_elements"]
    )


def load_center_data_from_json(filepath: str) -> CenterData:
    with open(filepath, "r") as f:
        raw_data = load(f)

    center_config = _parse_center_config(raw_data["config"])
    elements_data = [_parse_element_data(el_data) for el_data in raw_data["elements"]]

    center_data = CenterData(
        config=center_config,
        coeffs_functional=[array(cf, dtype=float) for cf in raw_data["coeffs_functional"]],
        elements=elements_data
    )
    return center_data
