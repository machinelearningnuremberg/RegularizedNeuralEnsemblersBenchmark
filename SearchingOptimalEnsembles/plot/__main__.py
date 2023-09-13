""" Plot from wandb

Usage:
    python -m SearchingOptimalEnsembles.plot [--metadatasets] [--algorithms] [--x_range] [--log_x]
    [--log_y] [--plot_name] [--extension] [--dpi] [--output_dir] [--plot_type] user_name project_name group_name

Positional arguments:
    user_name                   Wandb user name
    project_name                Wandb project name
    group_name                  Wandb group name

Optional arguments:
    -h, --help                  Show this help message and exit
    --benchmarks                List of metadatasets to plot
    --algorithms                List of algorithms to plot
    --x_range                   Bound x-axis (e.g. 1 10)
    --log_x                     If true, toggle logarithmic scale on the x-axis
    --log_y                     If true, toggle logarithmic scale on the y-axis
    --plot_name                 Plot name
    --extension                 Image format
    --dpi                       Image resolution
    --output_dir                Output directory
    --plot_type                 Type of plot to generate


Note:
    We have to use the __main__.py construct due to the issues explained in
    https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning

"""


import argparse
import logging
from pathlib import Path

from .plot import plot

parser = argparse.ArgumentParser(
    prog="python -m seo.plot", description="Plot incumbent from WandB"
)
parser.add_argument(
    "user_name",
    help="Wandb user name",
    # default="relea-research",
)
parser.add_argument(
    "project_name",
    help="Wandb project name",
    # default="SearchingOptimalEnsembles",
)
parser.add_argument(
    "group_name",
    help="Wandb group name",
    # default="debug",
)
parser.add_argument(
    "--metadatasets", nargs="+", help="List of metadatasets to plot", default=None
)
parser.add_argument(
    "--algorithms", nargs="+", help="List of algorithms to plot", default=None
)
parser.add_argument(
    "--x_range", nargs="+", type=float, help="Bound x-axis (e.g. 1 10)", default=None
)
parser.add_argument(
    "--log_x", action="store_true", help="If true, toggle logarithmic scale on the x-axis"
)
parser.add_argument(
    "--log_y", action="store_true", help="If true, toggle logarithmic scale on the y-axis"
)
parser.add_argument(
    "--plot_name",
    default="plot",
    help="Name for the plot",
)
parser.add_argument(
    "--extension",
    default="png",
    choices=["png", "pdf"],
    help="Image format",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=100,
    help="Image resolution",
)
parser.add_argument(
    "--output_dir",
    default="./saved_plots",
    help="Output directory",
)
parser.add_argument(
    "--plot_type",
    default="incumbent",
    choices=["regret", "aggregated_regret"],
    help="Type of plot to generate",
)
parser.add_argument(
    "--normalize",
    action="store_true",
    help="If true, normalize the regret",
)
args = parser.parse_args()

logging.basicConfig(level=logging.WARN)
if args.x_range is not None and len(args.x_range) == 2:
    args.x_range = tuple(args.x_range)
plot(
    user_name=args.user_name,
    project_name=args.project_name,
    group_name=args.group_name,
    metadatasets=args.metadatasets,
    algorithms=args.algorithms,
    x_range=args.x_range,
    log_x=args.log_x,
    log_y=args.log_y,
    plot_name=args.plot_name,
    extension=args.extension,
    dpi=args.dpi,
    output_dir=Path(args.output_dir),
    plot_type=args.plot_type,
    normalize=args.normalize,
)
