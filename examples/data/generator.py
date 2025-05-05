from numpy import random

from comp.models import CenterType, ElementType, ElementData, ElementConfig, CenterData, CenterConfig
from comp.utils import assert_positive


class DataGenerator:
    """Generates random test data for the optimization system."""

    def __init__(self,
                 num_elements: int = None,
                 num_decision_variables=None,
                 num_constraints=None,
                 num_schedules=None,
                 seed: int = 1810):
        """Initialize the data generator with system configuration."""

        if num_elements is None:
            num_elements = 3
        if num_decision_variables is None:
            num_decision_variables = [6, 4, 5]
        if num_constraints is None:
            num_constraints = [4, 2, 3]
        if num_schedules is None:
            num_schedules = [2, 3, 4]

        assert_positive(num_elements, "num_elements")
        for i, (ndv, nc, ns) in enumerate(zip(num_decision_variables, num_constraints, num_schedules)):
            assert_positive(ndv, f"num_decision_variables[{i}]")
            assert_positive(nc, f"num_constraints[{i}]")
            assert_positive(ns, f"num_schedules[{i}]")

        self.num_elements = num_elements
        self.num_decision_variables = num_decision_variables
        self.num_constraints = num_constraints
        self.num_schedules = num_schedules
        random.seed(seed)

    def _generate_element_data(self, element_idx: int) -> ElementData:
        """Generate random data for a single element."""

        return ElementData(
            config=ElementConfig(
                type=random.choice(list(ElementType)),
                id=element_idx,
                num_decision_variables=(n_e := self.num_decision_variables[element_idx]),
                num_constraints=(m_e := self.num_constraints[element_idx]),
                num_schedules=(n_k := self.num_schedules[element_idx]),
            ),
            coeffs_functional=random.randint(1, 10, n_e),
            resource_constraints=(
                random.randint(5, 10, m_e) * 100,
                random.randint(1, 5, n_e),
                random.randint(10, 15, n_e) * 100,
            ),
            aggregated_plan_costs=random.randint(1, 5, (m_e, n_e)),
            delta=.5,
            w=99,
            schedules=random.permutation(n_k),
            interest=random.random((m_e, n_k)),
            weight_coefficients=(weight_coeffs := random.random((m_e, n_k))) / weight_coeffs.sum(axis=1, keepdims=True),
        )

    def generate_center_data(self) -> CenterData:
        """Generate complete center data."""

        return CenterData(
            config=CenterConfig(
                type=random.choice(list(CenterType)),
                id=0,
                num_elements=self.num_elements
            ),
            coeffs_functional=[
                random.randint(1, 5, self.num_decision_variables[e])
                for e in range(self.num_elements)
            ],
            elements=[self._generate_element_data(e) for e in range(self.num_elements)],
        )


if __name__ == "__main__":
    """Test the DataGenerator class."""

    print(DataGenerator().generate_center_data())
