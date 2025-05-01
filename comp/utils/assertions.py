from typing import Any, List, TypeVar

from numpy import ndarray, array

T = TypeVar("T")


def assert_positive(value: Any, name: str = "") -> None:
    """Assert that a value is positive."""

    assert float(value) > 0, f"Value {name} must be positive, got {value}"


def assert_non_negative(value: Any, name: str = "") -> None:
    """Assert that a value is non-negative."""

    assert float(value) >= 0, f"Value {name} must be non-negative, got {value}"


def assert_valid_dimensions(arrays: List[ndarray],
                            expected_dims: List[int | tuple[int]],
                            names: List[str]) -> None:
    """Assert that arrays have valid dimensions."""

    for arr, dim, name in zip(arrays, expected_dims, names):
        assert arr.shape == dim, f"Array {name} has invalid dimensions. Expected {dim}, got {arr.shape}"


def assert_bounds(value: T, bounds: tuple[T, T], name: str = "") -> None:
    """Assert that a value is within bounds."""

    assert bounds[0] <= value <= bounds[1], f"Value {name} must be within bounds {bounds}, got {value}"


if __name__ == "__main__":
    """ Test the assertions module. """

    assert_positive(5, "test_value")
    assert_non_negative(0, "test_value")
    assert_valid_dimensions([array([1, 2]), array([3])], [(2,), (1,)], ["array1", "array2"])
    assert_bounds(5, (0, 10), "test_value")
