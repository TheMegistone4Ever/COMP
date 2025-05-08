from typing import Any, List, TypeVar, Tuple

from numpy import array

T = TypeVar("T")


def assert_positive(value: Any, name: str = "") -> None:
    """
    Assert that a given numerical value is strictly positive.

    :param value: The value to check, it will be cast to float for comparison.
    :param name: An optional name for the value, used in the error message.
    :raises AssertionError: If the value is not greater than zero.
    """

    assert float(value) > 0, f"Value {name} must be positive, got {value}"


def assert_non_negative(value: Any, name: str = "") -> None:
    """
    Assert that a given numerical value is non-negative (zero or positive).

    :param value: The value to check, it will be cast to float for comparison.
    :param name: An optional name for the value, used in the error message.
    :raises AssertionError: If the value is less than zero.
    """

    assert float(value) >= 0, f"Value {name} must be non-negative, got {value}"


def assert_valid_dimensions(arrays: List[Any],
                            expected_dims: List[Tuple[int, ...]],
                            names: List[str]) -> None:
    """
    Assert that lists of array-like objects have specified dimensions (shapes).

    Each array in the `arrays` list is compared against the corresponding shape tuple
    in the `expected_dims` list.

    :param arrays: A list of array-like objects (e.g., numpy arrays, lists of lists).
    :param expected_dims: A list of tuples, where each tuple represents the expected
                          shape for the corresponding array in `arrays`.
    :param names: A list of names for the arrays, used in error messages.
    :raises AssertionError: If any, array does not match its expected dimensions.
    """

    for arr, dim, name in zip(arrays, expected_dims, names):
        assert array(arr, dtype="object").shape == dim, (f"Array {name} has invalid dimensions."
                                                         f" Expected {dim}, got {arr.shape}")


def assert_bounds(value: T, bounds: Tuple[T, T], name: str = "") -> None:
    """
    Assert that a value falls within a specified inclusive range [lower, upper].

    :param value: The value to check.
    :param bounds: A tuple (lower_bound, upper_bound) defining the inclusive range.
    :param name: An optional name for the value, used in the error message.
    :raises AssertionError: If the value is less than lower_bound or greater than upper_bound.
    """

    assert bounds[0] <= value <= bounds[1], f"Value {name} must be within bounds {bounds}, got {value}"


if __name__ == "__main__":
    """ Test the assertions module. """

    assert_positive(5, "test_value")
    assert_non_negative(0, "test_value")
    assert_valid_dimensions([array([1, 2]), array([3])], [(2,), (1,)], ["array1", "array2"])
    assert_bounds(5, (0, 10), "test_value")
