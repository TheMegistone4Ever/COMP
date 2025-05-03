from dataclasses import replace
from enum import ReprEnum
from numbers import Number
from typing import Union, List, Any, Sequence, Optional, TypeVar, Protocol, Iterable

from numpy import ndarray, array
from tabulate import tabulate

from ..models.element import ElementData


def tab_out(subscription: str, data: Sequence[Sequence[str]], headers: List[str] = ("Parameter", "Value")) -> None:
    """Pretty-prints a table with the given data and headers."""

    print(f"\n{subscription}:")
    print(tabulate(data, headers, "grid"))


def stringify(tensor: Union[ReprEnum, Number, Iterable[Any], ndarray], indent: int = 4, precision: int = 2) -> str:
    """
    Formats n-dimensional tensors (nested lists/arrays) for better readability.

    Args:
        tensor: Input tensor (can be a number, list, nested list, or numpy array)
        indent: Number of spaces for each level of indentation
        precision: Number of decimal places for floating-point numbers

    Returns:
        str: Formatted string representation of the tensor

    Examples:
        >>> stringify(42)
        42
        >>> stringify([1, 2, 3])
        [1, 2, 3]
        >>> stringify([[1, 2], [3, 4]])
        [
            [1, 2],
            [3, 4]
        ]
    """

    def convert_ndarrays(obj):
        """Helper function to convert numpy arrays to list recursively"""

        if isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_ndarrays(item) for item in obj)
        return obj

    tensor = convert_ndarrays(tensor)

    def format_number(x: Number) -> str:
        """Helper function to format numbers with consistent precision"""

        if isinstance(x, float):
            return str(round(x, precision))
        return str(x)

    def is_nested(x: Any) -> bool:
        """Helper function to check if an object is a nested structure"""

        return isinstance(x, list) and any(isinstance(item, (list, ndarray)) for item in x)

    def format_recursive(x: Any, level: int = 0) -> str:
        """Recursively format nested structures"""

        # Handle enums with custom __str__ method
        if isinstance(x, ReprEnum):
            return str(x)

        # Base case: handle numbers
        if isinstance(x, Number):
            return format_number(x)

        # Handle non-nested lists
        if isinstance(x, list) and not is_nested(x):
            elements = [format_number(item) if isinstance(item, Number) else str(item) for item in x]
            return f"[{", ".join(elements)}]"

        # Handle nested structures
        if isinstance(x, list):
            spacer = " " * (level * indent)
            next_spacer = " " * ((level + 1) * indent)
            elements = [format_recursive(item, level + 1) for item in x]
            open_bracket, close_bracket = ("[", "]") if isinstance(x, list) else ("(", ")")
            return f"{open_bracket}\n{next_spacer}" + f",\n{next_spacer}".join(elements) + f"\n{spacer}{close_bracket}"
        return str(x)

    return format_recursive(tensor)


def copy_element_coeffs(element: ElementData, coeffs_functional: Optional[ndarray] = None) -> ElementData:
    """Creates a copy of an ElementData instance with optionally modified coeffs_functional."""

    return element if coeffs_functional is None else replace(element, coeffs_functional=coeffs_functional)


class SupportsAdd(Protocol):
    """Protocol for objects that support the addition operator."""

    def __add__(self, other: "SupportsAdd") -> "SupportsAdd": ...


T = TypeVar("T", bound=SupportsAdd)


def lp_sum(variables: Iterable[T]) -> T | Any:
    """
    Summation function for linear programming variables that supports any sequence
    of elements implementing the addition operator.

    Args:
        variables: A sequence of addable elements (typically LP variables or expressions)

    Returns:
        The sum of all elements in the sequence.
        Returns 0 if a sequence is empty.
    """

    iterator = iter(variables)
    try:
        result = next(iterator)
    except StopIteration:
        return 0

    for value in iterator:
        result += value

    return result


if __name__ == "__main__":
    """ Test the stringify function with various inputs. """

    num_0d = 42
    list_1d = [1, 2, 3]
    list_2d = [[1, 2], [3, 4]]
    array_1d = array([1, 2, 3])
    array_2d = array([[1, 2], [3, 4]])
    tensor_3d_int = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    tensor_3d_float = array(tensor_3d_int, dtype=float) + .123456789

    print(f"Number:\n{stringify(num_0d)}", end="\n\n")
    print(f"1D List:\n{stringify(list_1d)}", end="\n\n")
    print(f"2D List:\n{stringify(list_2d)}", end="\n\n")
    print(f"1D Array:\n{stringify(array_1d)}", end="\n\n")
    print(f"2D Array:\n{stringify(array_2d)}", end="\n\n")
    print(f"3D Tensor (int):\n{stringify(tensor_3d_int)}", end="\n\n")
    print(f"3D Tensor (float, 6 d.p.):\n{stringify(tensor_3d_float, precision=6)}", end="\n\n")
