"""
Script for generating plots from WandB data.

Usage:
    python -m SearchingOptimalEnsembles.plot [OPTIONS]

    For detailed options, run `python -m SearchingOptimalEnsembles.plot --help`

Note:
    We have to use the `__main__.py` construct due to the issues explained in
    https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning
"""


import argparse
import logging
from pathlib import Path

from .plot import plot

parser = argparse.ArgumentParser(
    prog="python -m SearchingOptimalEnsembles.plot",
    description="Plot incumbent from WandB",
)

# fmt: off
# User and Project Configuration
parser.add_argument("--user_name", default="relea-research", help="Wandb user name.")
parser.add_argument("--project_name", default="SearchingOptimalEnsembles", help="Wandb project name.")
parser.add_argument("--group_names", nargs="+", default=None, help="Names of the Wandb groups.")

# Output Settings
parser.add_argument("--output_dir", default="./saved_plots", help="Directory for saving plots.")

# Plot Type and Data Transformation Settings
parser.add_argument("--plot_type", choices=["regret", "rank"], default="regret", help="Type of plot to generate.")
parser.add_argument("--unnormalize", action="store_true", help="Use unnormalized data.")
parser.add_argument("--per_dataset", action="store_true", help="Plot per dataset, aggregate otherwise.")

# Axis Settings
parser.add_argument("--x_axis", default="searcher_iteration", help="Label for the x-axis.")
parser.add_argument("--y_axis", default="incumbent", help="Label for the y-axis.")

# Scale and Bound Settings
parser.add_argument("--x_range", nargs="+", type=float, default=None, help="Bound x-axis (e.g., 1 10).")
parser.add_argument("--log_x", action="store_true", help="Use logarithmic scale for the x-axis.")
parser.add_argument("--log_y", action="store_true", help="Use logarithmic scale for the y-axis.")

# Plot Appearance Settings
parser.add_argument("--plot_name", default="plot", help="Name for the plot.")
parser.add_argument("--extension", choices=["png", "pdf"], default="png", help="Plot image format.")
parser.add_argument("--dpi", type=int, default=100, help="Plot image resolution.")
# fmt: on

args = parser.parse_args()

args = parser.parse_args()

logging.basicConfig(level=logging.WARN)
if args.x_range is not None and len(args.x_range) == 2:
    args.x_range = tuple(args.x_range)
plot(
    user_name=args.user_name,
    project_name=args.project_name,
    group_names=args.group_names,
    output_dir=Path(args.output_dir),
    plot_type=args.plot_type,
    unnormalize=args.unnormalize,
    per_dataset=args.per_dataset,
    x_axis=args.x_axis,
    y_axis=args.y_axis,
    x_range=args.x_range,
    log_x=args.log_x,
    log_y=args.log_y,
    plot_name=args.plot_name,
    extension=args.extension,
    dpi=args.dpi,
)


# python -m SearchingOptimalEnsembles.plot --user_name relea-research --project_name SearchingOptimalEnsembles --group_names "G1" "G2" "G3" "G4" --plot_type regret
