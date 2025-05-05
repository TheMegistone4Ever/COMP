from comp.solvers import CenterLinearThird
from examples.data import DataGenerator


def main() -> None:
    """Main function to run the solver."""

    center_data = DataGenerator().generate_center_data()
    center_linear_first = CenterLinearThird(center_data)
    center_linear_first.setup()
    center_linear_first.print_results()


if __name__ == "__main__":
    """Test the solver."""

    main()
