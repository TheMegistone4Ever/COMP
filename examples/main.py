from comp.solvers import CenterLinearThird
from examples.data import DataGenerator

if __name__ == "__main__":
    """Test the solver."""

    center_data = DataGenerator().generate_center_data()
    element_linear_first = CenterLinearThird(center_data)
    element_linear_first.setup()
    element_linear_first.print_results()
