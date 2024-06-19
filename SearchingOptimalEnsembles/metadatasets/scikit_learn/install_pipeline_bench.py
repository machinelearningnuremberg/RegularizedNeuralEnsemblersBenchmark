import argparse
import subprocess


def install_pipeline_bench(without_data_creation):
    script_path = (
        "SearchingOptimalEnsembles/metadatasets/scikit_learn/install_pipeline_bench.sh"
    )
    # Prepare the command with or without the flag based on the input argument
    command = [script_path]
    if without_data_creation:
        command.append("--without_data_creation")

    # Run the script with the appropriate flags
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline bench installation script."
    )
    # Define a flag argument
    parser.add_argument(
        "--without_data_creation",
        action="store_true",
        help="Install pipeline bench without data creation module",
    )

    args = parser.parse_args()

    # Call the function with the parsed argument
    install_pipeline_bench(args.without_data_creation)


if __name__ == "__main__":
    main()
