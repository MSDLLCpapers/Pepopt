"""
Entry point for the FP-based Free Wilson Analysis (FP-FWA) pipeline.

Example usage
-------------
Run from the ``Pepopt_Public/`` directory:

    python local_run_fpfw.py test_data/fwa-example_1_aligned.csv pIC50 ID M-0001 \\
        --model ridge --desc_type ecfp --desc_norm False \\
        --fp_len 2048 --fp_radius 2 --fp_sel_n 500 --fp_min_freq 0 \\
        --fpbit_plot True --fpbit_top_n 5 --evaluation True \\
        --max_enum_out 1000 --enum_positions 0 1 2 3 4 5

For full argument documentation run:

    python local_run_fpfw.py -h
"""

import argparse
import logging
import os
import sys

from src.peptide_fw import free_wilson_peptides_fp

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def _str_to_bool(value):
    """Convert a CLI string to a Python bool."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")


def get_args(args=None):
    """Build and parse the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run FP-based Free Wilson Analysis on aligned peptide data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Required positional arguments                                        #
    # ------------------------------------------------------------------ #
    parser.add_argument("incsv", help="Path to the input aligned CSV file.")
    parser.add_argument("actcol", help="Name of the activity column.")
    parser.add_argument("idcol", help="Name of the compound identifier column.")
    parser.add_argument("refid", help="Identifier of the reference compound.")

    # ------------------------------------------------------------------ #
    # Alignment                                                            #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--aln",
        choices=["pairwise", "prealigned"],
        default="prealigned",
        help="Alignment mode. Only 'prealigned' is supported for public use.",
    )
    parser.add_argument("--helmcol", default="HELM", help="Name of the HELM column.")
    parser.add_argument(
        "--pos_prefix",
        default="PEPTIDE1_",
        help="Prefix for positional monomer columns (e.g. 'PEPTIDE1_').",
    )

    # ------------------------------------------------------------------ #
    # Activity transformation / filtering                                  #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--min_act_keep",
        type=float,
        default=None,
        help="Exclude compounds with activity below this threshold.",
    )
    parser.add_argument(
        "--max_act_keep",
        type=float,
        default=None,
        help="Exclude compounds with activity above this threshold.",
    )
    parser.add_argument(
        "--xform",
        default="auto",
        choices=["auto", "positive", "negative", "p"],
        help=(
            "Activity transformation. "
            "'auto': leave as-is unless range > 100 (then assume IC50 -> -log10). "
            "'positive': leave as-is. "
            "'negative': lower is better (linear scale)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Dataset grouping                                                     #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--split_series_col",
        type=str,
        default=None,
        help="Column name for compound series. If given, only fits FW on the same series as the reference.",
    )
    parser.add_argument(
        "--fixed_length",
        action="store_true",
        help="Exclude peptides whose length differs from the reference.",
    )
    parser.add_argument(
        "--aln_min_sim",
        type=float,
        default=0.8,
        help="Minimum similarity to reference during alignment.",
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--model",
        choices=["ridge", "bayesian_ridge"],
        default="ridge",
        help="Regression model type.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularisation strength for Ridge regression.",
    )

    # ------------------------------------------------------------------ #
    # Descriptor / fingerprint                                             #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--desc_type",
        choices=["ecfp", "rdk_desc"],
        default="ecfp",
        help="Molecular descriptor type.",
    )
    parser.add_argument(
        "--desc_norm",
        type=_str_to_bool,
        default=False,
        help="Normalise descriptors to zero mean and unit variance.",
    )
    parser.add_argument(
        "--fp_len",
        type=int,
        default=2048,
        help="Number of bits in the Morgan fingerprint.",
    )
    parser.add_argument(
        "--fp_radius",
        type=int,
        default=2,
        help="Radius of the Morgan fingerprint.",
    )
    parser.add_argument(
        "--fp_sel_n",
        type=int,
        default=500,
        help="Select the top N fingerprint bits by variance.",
    )
    parser.add_argument(
        "--fp_min_freq",
        type=int,
        default=0,
        help="Minimum number of peptides a fingerprint bit must appear in to be used.",
    )

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--monomer_top_n",
        type=int,
        default=10,
        help="Number of top monomers to show per position in heatmaps.",
    )
    parser.add_argument(
        "--fpbit_plot",
        type=_str_to_bool,
        default=True,
        help="Generate Morgan bit substructure figures.",
    )
    parser.add_argument(
        "--fpbit_top_n",
        type=int,
        default=5,
        help="Number of top fingerprint bits to report per monomer.",
    )
    parser.add_argument(
        "--evaluation",
        type=_str_to_bool,
        default=True,
        help="Evaluate model performance with CV and time-based splits.",
    )

    # ------------------------------------------------------------------ #
    # Enumeration                                                          #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--max_enum_out",
        type=int,
        default=1000,
        help="Maximum number of enumerated peptides to output (0 to skip).",
    )
    parser.add_argument(
        "--only_inventory",
        type=str,
        nargs="+",
        help=(
            'Restrict enumeration to monomers from these collections. '
            'Use "all" for all available. Options: "Enamine", "Enamine_48h", "MBBC".'
        ),
    )
    parser.add_argument(
        "--white_list",
        type=str,
        nargs="+",
        help="Only consider these monomers for enumeration (HELM symbols, no brackets). E.g. hS K",
    )
    parser.add_argument(
        "--black_list",
        type=str,
        nargs="+",
        help="Exclude these monomers from enumeration (HELM symbols, no brackets). E.g. hS K",
    )
    parser.add_argument(
        "--enum_positions",
        type=int,
        nargs="+",
        help="Positions to enumerate (0-based).",
    )

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory in which to save all output files.",
    )

    return parser.parse_args(args if args is not None else sys.argv[1:])


def main(args):
    """Run the FP-FWA pipeline with the given parsed arguments."""
    return free_wilson_peptides_fp(args)


if __name__ == "__main__":
    main(get_args())
