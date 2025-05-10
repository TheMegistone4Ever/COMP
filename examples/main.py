from dataclasses import replace

from comp.models import CenterType
from comp.solvers import new_center_solver
from examples.data import DataGenerator


def main() -> None:
    """Main function to run the solver."""

    center_data = DataGenerator().generate_center_data()
    center_linear = new_center_solver(replace(center_data, config=replace(
        center_data.config, type=CenterType.WEIGHTED_BALANCE)))
    center_linear.coordinate()
    center_linear.print_results()


if __name__ == "__main__":
    """Test the solver."""

    main()
