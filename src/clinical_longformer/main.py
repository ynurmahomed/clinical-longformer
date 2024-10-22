import argparse
import logging
import sys
import os

from clinical_longformer import __version__

from .data.processing import process_notes

__author__ = "Yassin Nurmahomed"
__copyright__ = "Yassin Nurmahomed"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----


def dataset(mimic_path, category, note_length, n_days, out_path):
    """Process dataset

    Args:
      mimic_path (str): MIMIC-III dataset location
      category (str): Clinical notes category
      note_length (int): Note length
      n_days (int): Number of days of notes to consider.

    Returns:
      str: MIMIC-III dataset location
    """
    process_notes(mimic_path, category, note_length, n_days, out_path)


# ---- CLI ----


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Clinical Longformer data processing")
    parser.add_argument(
        "--version",
        action="version",
        version="clinical-longformer {ver}".format(ver=__version__),
    )
    parser.add_argument(dest="mimic_path", help="MIMIC-III dataset path", type=str)
    parser.add_argument(
        dest="category",
        help="set notes category (ds - Discharge Summary)",
        type=str,
        choices=["ds", "all"],
    )
    parser.add_argument(
        dest="length",
        help="set note length, -1 means do not chunk text",
        type=int,
        choices=[-1, 512, 1024, 2048, 4096],
    )
    parser.add_argument(
        dest="out_path",
        help="set output path",
        type=str,
        nargs="?",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--n-days",
        dest="n_days",
        help="set number of days (only used if category is set to all)",
        type=int,
        choices=list(range(1, 31)),
        metavar='{1-30}',
        default=3,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formated message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Processing dataset...")
    dataset(args.mimic_path, args.category, args.length, args.n_days, args.out_path)
    _logger.info("Done processing.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
