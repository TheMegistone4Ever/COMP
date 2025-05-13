from enum import ReprEnum
from numbers import Number
from typing import Any, Iterable, List, Protocol, Sequence, TypeVar, Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from numpy import array, ndarray
from tabulate import tabulate

from comp.models.element import ElementData


def tab_out(subscription: str, data: Sequence[Sequence[str]], headers: List[str] = ("Parameter", "Value")) -> None:
    """
    Print data in a formatted table using the `tabulate` library.

    :param subscription: A title or description for the table, printed before it.
    :param data: A sequence of sequences (e.g., list of lists) representing table rows.
    :param headers: A list of strings for table headers.
    Defaults to ("Parameter", "Value").
    """

    print(f"\n{subscription}:\n{tabulate(data, headers, "grid")}")


def stringify(tensor: Any, indent: int = 4, precision: int = 2) -> str:
    """
    Format n-dimensional tensors (nested lists/arrays/tuples, numbers, enums) for readable string output.

    This function recursively formats the input, converting numpy arrays to lists,
    applying specified precision to floats, and indenting nested structures.

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

    :param tensor: The input data to format (e.g., number, list, tuple, numpy array, ReprEnum).
    :param indent: The number of spaces for each indentation level in nested structures.
    :param precision: The number of decimal places for floating-point numbers.
    :return: A string representation of the formatted tensor.
    """

    def convert_ndarrays(obj: Any) -> Any:
        """
        Recursively convert numpy arrays within a nested structure to Python lists.

        Other list/tuple structures are preserved.
        Non-collection items are returned as is.

        :param obj: The object to process, potentially containing numpy arrays.
        :return: The object with all numpy arrays converted to lists.
        """

        if isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_ndarrays(item) for item in obj)
        return obj

    tensor = convert_ndarrays(tensor)

    def format_number(x: Number) -> str:
        """
        Format a number as a string, applying precision to floats.

        :param x: The number (int, float, etc.) to format.
        :return: String representation of the number, rounded if `float`.
        """

        if isinstance(x, float):
            return str(round(x, precision))
        return str(x)

    def is_nested(x: Any) -> bool:
        """
        Check if an object is a nested list or tuple.

        A structure is considered nested if it’s a list or tuple containing
        at least one other list, tuple, or numpy array as an element.

        :param x: The object to check.
        :return: True if the object is a nested list/tuple, False otherwise.
        """

        return isinstance(x, (list, tuple)) and any(isinstance(item, (list, tuple, ndarray)) for item in x)

    def format_recursive(x: Any, level: int = 0) -> str:
        """
        Recursively format an object into a string with indentation for nested structures.

        Handles numbers, ReprEnums, non-nested lists/tuples, dictionaries, and nested structures.

        :param x: The object to format.
        :param level: The current nesting level, used for indentation.
        :return: The formatted string representation of the object.
        """

        # Handle enums
        if isinstance(x, ReprEnum):
            return str(x)

        # Handle numbers
        if isinstance(x, Number):
            return format_number(x)

        # Handle dictionaries
        if isinstance(x, dict):
            spacer = " " * (level * indent)
            next_spacer = " " * ((level + 1) * indent)
            items = [f"{next_spacer}{repr(k)}: {format_recursive(v, level + 1)}" for k, v in x.items()]
            return "{\n" + ",\n".join(items) + f"\n{spacer}}}"

        # Handle non-nested lists
        if isinstance(x, list) and not is_nested(x):
            elements = [format_number(item) if isinstance(item, Number) else str(item) for item in x]
            return f"[{", ".join(elements)}]"

        # Handle non-nested tuples
        if isinstance(x, tuple) and not is_nested(x):
            elements = [format_number(item) if isinstance(item, Number) else str(item) for item in x]
            return f"({", ".join(elements)})"

        # Handle nested structures (lists or tuples)
        if isinstance(x, (list, tuple)):
            spacer = " " * (level * indent)
            next_spacer = " " * ((level + 1) * indent)
            elements = [format_recursive(item, level + 1) for item in x]
            open_bracket, close_bracket = ("[", "]") if isinstance(x, list) else ("(", ")")
            return f"{open_bracket}\n{next_spacer}" + f",\n{next_spacer}".join(elements) + f"\n{spacer}{close_bracket}"

        return str(x)

    return format_recursive(tensor)


class SupportsAdd(Protocol):
    """Protocol for objects that support the addition operator."""

    def __add__(self, other: Self) -> Self: ...


T_lp_sum = TypeVar("T_lp_sum", bound=SupportsAdd)


def lp_sum(variables: Iterable[T_lp_sum]) -> Union[T_lp_sum, int]:
    """
    Sum a sequence of elements that support addition, returning 0 for an empty sequence.

    This function is typically used for summing linear programming variables or expressions.
    It iterates through the `variables` and accumulates their sum.

    :param variables: This is iterable of elements that support the `__add__` operator.
    :return: The sum of all elements in `variables`.
    If `variables` is empty, returns 0 (int).
    Otherwise, returns a value of the same type as the elements in `variables`.
    """

    iterator = iter(variables)
    try:
        result = next(iterator)
    except StopIteration:
        return 0

    for value in iterator:
        result += value

    return result


def get_lp_problem_sizes(data: List[ElementData]) -> List[Tuple[int, int]]:
    """
    Extract the linear programming problem sizes (constraints, variables) for a list of elements.

    For each ElementData object in the input list, this function retrieves the number
    of constraints and decision variables from its configuration.

    :param data: A list of ElementData objects.
    :return: A list of tuples, where each tuple (num_constraints, num_decision_variables)
             corresponds to an element in the input list.
    """

    return [(d.config.num_constraints, d.config.num_decision_variables) for d in data]


if __name__ == "__main__":
    """Test the stringify function with various inputs."""

    num_0d = 42
    list_1d = [1, 2, 3]
    list_2d = [[1, 2], [3, 4]]
    array_1d = array([1, 2, 3])
    array_2d = array([[1, 2], [3, 4]])
    tensor_3d_int = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    tensor_3d_float = array(tensor_3d_int, dtype=float) + .123456789
    tuple_1d = (1, 2, 3)
    tuple_2d = ((1, 2), [3, 4, tuple_1d], (1, 2, 3, 4))
    tuple_combined = (tuple_1d, [1, 2, 3, 4], tuple_2d, [1, 2, 3])

    print(f"Number:\n{stringify(num_0d)}", end="\n\n")
    print(f"1D List:\n{stringify(list_1d)}", end="\n\n")
    print(f"2D List:\n{stringify(list_2d)}", end="\n\n")
    print(f"1D Array:\n{stringify(array_1d)}", end="\n\n")
    print(f"2D Array:\n{stringify(array_2d)}", end="\n\n")
    print(f"3D Tensor (int):\n{stringify(tensor_3d_int)}", end="\n\n")
    print(f"3D Tensor (float, 6 d.p.):\n{stringify(tensor_3d_float, precision=6)}", end="\n\n")
    print(f"1D Tuple:\n{stringify(tuple_1d)}", end="\n\n")
    print(f"2D Tuple:\n{stringify(tuple_2d)}", end="\n\n")
    print(f"Combined Tuple:\n{stringify(tuple_combined)}")
