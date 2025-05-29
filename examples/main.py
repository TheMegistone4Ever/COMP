from dataclasses import replace
from os.path import exists

from comp.io import load_center_data_from_json
from comp.models import CenterType
from comp.solvers import new_center_solver
from examples.data import DataGenerator


def main() -> None:
    """Main function to run the solver."""

    generated_data_filepath = "center_data_generated.json"

    if not exists(generated_data_filepath):
        print(f"Generating center data and saving to {generated_data_filepath}...")
        center_data_generated = DataGenerator().generate_center_data()
        center_data_generated.save_to_json(generated_data_filepath)
        print(f"Generated center data saved to {generated_data_filepath}")
    else:
        print(f"Center data file {generated_data_filepath} already exists. Loading existing data...")

    center_data_loaded = load_center_data_from_json(generated_data_filepath)
    print(f"Center data loaded from {generated_data_filepath}")

    center_data_for_solver = replace(center_data_loaded, config=replace(
        center_data_loaded.config, type=CenterType.RESOURCE_ALLOCATION_COMPROMISE))

    center_linear_solver = new_center_solver(center_data_for_solver)

    print("\nStarting solver coordination...")
    center_linear_solver.coordinate()
    print("Solver coordination complete.")

    print("\n--- Solver Results (Printed to Console) ---")
    center_linear_solver.print_results()
    print("--- End of Console Output ---")

    results_filepath = "center_results_output.json"
    center_linear_solver.save_results_to_json(results_filepath)
    print(f"\nSolver results saved to {results_filepath}")


if __name__ == "__main__":
    """Test the solver."""

    main()
