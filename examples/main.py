from comp.solvers.element.linear.second import ElementLinearSecond
from examples.data import DataGenerator

if __name__ == "__main__":
    center_data = DataGenerator().generate_center_data()
    element_linear_first = ElementLinearSecond(center_data.elements[0])
    element_linear_first.setup()
    element_linear_first.solve()
    element_linear_first.print_results()
