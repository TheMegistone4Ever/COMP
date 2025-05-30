from dataclasses import replace, dataclass
from enum import Enum, auto
from unittest import TestCase, main

from numpy import array, int64, testing

from comp.models import ElementData, ElementConfig, ElementType, CenterData, CenterType
from comp.parallelization.heuristic import get_order
from comp.solvers import new_center_solver, CenterLinearFirst, CenterLinearSecond, CenterLinearThird, CenterLinkedFirst
from comp.solvers.element import ElementLinearFirst, ElementLinearSecond
from comp.solvers.factories import new_element_solver
from comp.utils import (assert_positive, assert_non_negative, assert_bounds, assert_valid_dimensions, stringify, lp_sum,
                        get_lp_problem_sizes, json_serializer)
from examples import DataGenerator


class TestAssertions(TestCase):
    """Tests for assertion utility functions."""

    def test_assert_positive_logic(self) -> None:
        """Test assert_positive handles valid and invalid cases."""

        try:
            assert_positive(5, "test_val_valid")
        except AssertionError:
            self.fail("assert_positive raised AssertionError unexpectedly for valid input")

        with self.assertRaisesRegex(AssertionError, "must be positive",
                                    msg="assert_positive did not raise error for zero"):
            assert_positive(0, "test_val_zero")

        with self.assertRaisesRegex(AssertionError, "must be positive",
                                    msg="assert_positive did not raise error for negative"):
            assert_positive(-1, "test_val_neg")

    def test_assert_non_negative_logic(self) -> None:
        """Test assert_non_negative handles valid and invalid cases."""

        try:
            assert_non_negative(5, "test_val_pos")
        except AssertionError:
            self.fail("assert_non_negative raised AssertionError unexpectedly for positive input")

        try:
            assert_non_negative(0, "test_val_zero")
        except AssertionError:
            self.fail("assert_non_negative raised AssertionError unexpectedly for zero input")

        with self.assertRaisesRegex(AssertionError, "must be non-negative",
                                    msg="assert_non_negative did not raise error for negative"):
            assert_non_negative(-1, "test_val_neg")

    def test_assert_bounds_valid(self) -> None:
        """Test assert_bounds with a value within bounds."""

        try:
            assert_bounds(5, (0, 10), "test_val")
        except AssertionError:
            self.fail("assert_bounds raised AssertionError unexpectedly")

    def test_assert_bounds_invalid_upper(self) -> None:
        """Test assert_bounds with a value above the upper bound."""

        with self.assertRaisesRegex(AssertionError, "must be within bounds"):
            assert_bounds(11, (0, 10), "test_val")

    def test_assert_valid_dimensions_valid(self) -> None:
        """Test assert_valid_dimensions with matching dimensions."""

        arr1 = array([1, 2])
        arr2 = array([[1], [2]])
        try:
            assert_valid_dimensions([arr1, arr2], [(2,), (2, 1)], ["arr1", "arr2"])
        except AssertionError:
            self.fail("assert_valid_dimensions raised AssertionError unexpectedly")


class TestHelpers(TestCase):
    """Tests for helper utility functions."""

    def test_stringify_nested_list(self) -> None:
        """Test stringify formats nested lists correctly."""

        data = [[1, 2], [3.14159, 4]]
        expected = """[
    [1, 2],
    [3.14, 4]
]"""
        self.assertEqual(stringify(data, precision=2), expected)

    def test_lp_sum_non_empty(self) -> None:
        """Test lp_sum with a non-empty list of numbers."""

        self.assertEqual(lp_sum([1, 2, 3]), 6)

    def test_lp_sum_empty(self) -> None:
        """Test lp_sum with an empty list."""

        self.assertEqual(lp_sum(list()), 0)

    def test_get_lp_problem_sizes(self) -> None:
        """Test get_lp_problem_sizes extracts sizes correctly."""

        elem1_cfg = ElementConfig(id=1, type=ElementType.DECENTRALIZED, num_constraints=2, num_decision_variables=3)
        elem2_cfg = ElementConfig(id=2, type=ElementType.NEGOTIATED, num_constraints=4, num_decision_variables=5)

        elem1_data = ElementData(config=elem1_cfg, coeffs_functional=array(list()),
                                 resource_constraints=(array(list()), array(list()), array(list())),
                                 aggregated_plan_costs=array(list()), delta=None, w=None)
        elem2_data = ElementData(config=elem2_cfg, coeffs_functional=array(list()),
                                 resource_constraints=(array(list()), array(list()), array(list())),
                                 aggregated_plan_costs=array(list()), delta=None, w=None)
        sizes = get_lp_problem_sizes([elem1_data, elem2_data])
        self.assertEqual(sizes, [(2, 3), (4, 5)])


class TestJsonSerializer(TestCase):
    """Tests for the custom JSON serializer."""

    def test_json_serializer_handles_types(self) -> None:
        """Test _json_serializer handles numpy array, enum, float, inf, nan, dataclass, and numpy int."""

        class MyEnum(Enum):
            VAL1 = auto()

        @dataclass
        class SimpleDC:
            x: int = 1

        self.assertEqual(json_serializer(array([1, 2])), [1, 2])
        self.assertEqual(json_serializer(MyEnum.VAL1), "VAL1")
        self.assertEqual(json_serializer(3.14), 3.14)
        self.assertEqual(json_serializer(float("inf")), "Infinity")
        self.assertEqual(json_serializer(float("-inf")), "-Infinity")
        self.assertEqual(json_serializer(float("nan")), "NaN")
        self.assertEqual(json_serializer(SimpleDC(x=5)), {"x": 5})
        self.assertEqual(json_serializer(int64(10)), 10)


class TestModels(TestCase):
    """Tests for data model classes."""

    def test_element_data_copy(self) -> None:
        """Test the ElementData copy method creates a deep copy where appropriate."""

        cfg = ElementConfig(id=1, type=ElementType.DECENTRALIZED, num_constraints=1, num_decision_variables=1)
        orig = ElementData(
            config=cfg,
            coeffs_functional=array([1.0]),
            resource_constraints=(array([10.0]), array([1.0]), array([100.0])),
            aggregated_plan_costs=array([[2.0]]),
            delta=0.5,
            w=array([0.1, 0.2])
        )
        copied = orig.copy()

        self.assertIsNot(orig, copied)
        self.assertIs(orig.config, copied.config)
        self.assertIsNot(orig.coeffs_functional, copied.coeffs_functional)
        self.assertIsNot(orig.resource_constraints[0], copied.resource_constraints[0])
        self.assertIsNot(orig.aggregated_plan_costs, copied.aggregated_plan_costs)
        self.assertIsNot(orig.w, copied.w)

        testing.assert_array_equal(orig.coeffs_functional, copied.coeffs_functional)
        testing.assert_array_equal(orig.resource_constraints[0], copied.resource_constraints[0])
        self.assertEqual(orig.delta, copied.delta)


class TestGenerator(TestCase):
    """Tests for the data generator."""

    def test_data_generator_creates_center_data(self) -> None:
        """Test DataGenerator generates a CenterData instance with elements correctly configured."""

        generator = DataGenerator(num_elements=2, num_decision_variables=[2, 3], num_constraints=[1, 2], seed=42)
        center_data = generator.generate_center_data()

        self.assertIsInstance(center_data, CenterData)
        self.assertEqual(center_data.config.num_elements, 2)
        self.assertEqual(len(center_data.elements), 2)
        self.assertEqual(len(center_data.coeffs_functional), 2)
        self.assertEqual(center_data.elements[0].config.num_decision_variables, 2)
        self.assertEqual(center_data.elements[1].config.num_decision_variables, 3)
        self.assertEqual(center_data.elements[0].config.num_constraints, 1)
        self.assertEqual(center_data.elements[1].config.num_constraints, 2)


class TestParallelization(TestCase):
    """Tests for parallelization utilities."""

    def test_get_order_simple(self) -> None:
        """Test get_order distributes tasks."""

        sizes = [(1, 1), (5, 5), (1, 1), (1, 1)]
        threads = 2
        order = get_order(sizes, threads)

        self.assertEqual(len(order), threads)
        all_indices = set(idx for group in order for idx in group)
        self.assertEqual(all_indices, {0, 1, 2, 3})

        for group in order:
            if 1 in group:
                self.assertLessEqual(len(group), len(sizes) - len(group) + 1)


class TestSolversFactories(TestCase):
    """Tests for solver factory functions."""

    def test_new_element_solver_types(self) -> None:
        """Test new_element_solver returns correct solver types based on element configuration."""

        cfg_dec = ElementConfig(id=1, type=ElementType.DECENTRALIZED, num_constraints=1, num_decision_variables=1)
        cfg_neg = ElementConfig(id=2, type=ElementType.NEGOTIATED, num_constraints=1, num_decision_variables=1)

        data_dec = ElementData(config=cfg_dec, coeffs_functional=array(list()),
                               resource_constraints=(array([1]), array([1]), array([1])),
                               aggregated_plan_costs=array([[1]]))
        data_neg = ElementData(config=cfg_neg, coeffs_functional=array(list()),
                               resource_constraints=(array([1]), array([1]), array([1])),
                               aggregated_plan_costs=array([[1]]), delta=1, w=array([1]))

        solver_dec = new_element_solver(data_dec)
        solver_neg = new_element_solver(data_neg)

        self.assertIsInstance(solver_dec, ElementLinearFirst)
        self.assertIsInstance(solver_neg, ElementLinearSecond)


class TestSolvers(TestCase):
    """Tests for main solver classes."""

    def test_new_center_solver_types(self) -> None:
        """Test new_center_solver returns correct solver types based on center configuration."""

        base_data = DataGenerator(1, [1], [1]).generate_center_data()

        data_first = replace(base_data, config=replace(base_data.config, type=CenterType.STRICT_PRIORITY))
        data_second = replace(base_data, config=replace(base_data.config, type=CenterType.GUARANTEED_CONCESSION))
        data_third = replace(base_data, config=replace(base_data.config, type=CenterType.WEIGHTED_BALANCE))
        data_fourth = replace(base_data, config=replace(base_data.config,
                                                        type=CenterType.RESOURCE_ALLOCATION_COMPROMISE))

        solver_first = new_center_solver(data_first)
        solver_second = new_center_solver(data_second)
        solver_third = new_center_solver(data_third)
        solver_fourth = new_center_solver(data_fourth)

        self.assertIsInstance(solver_first, CenterLinearFirst)
        self.assertIsInstance(solver_second, CenterLinearSecond)
        self.assertIsInstance(solver_third, CenterLinearThird)
        self.assertIsInstance(solver_fourth, CenterLinkedFirst)


if __name__ == "__main__":
    main(argv=["first-arg-is-ignored"], exit=False)
