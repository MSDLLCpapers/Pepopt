"""
Free Wilson Linear Enumeration
Ruibo Zhang, 2/13/2025

Enumerates peptide combinations by combining per-position monomer coefficients
from a fitted FP-FW model.  The coefficients are linearly additive (Ridge
regression), so the top-K combinations can be found efficiently with dynamic
programming.

Can be run as a standalone script (see ``__main__`` block at the bottom) or
called programmatically via :func:`enum_one_parent_helm`.
"""

from collections.abc import Sequence
from copy import deepcopy
from logging import getLogger
from pathlib import Path

import pandas as pd

from src.api import peptide_api
from src.config import MONO_DB_OUTPUT

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Monomer filtering helpers
# ---------------------------------------------------------------------------

def limit_monos_to_enum_positions(coef: pd.DataFrame, positions=None):
    """
    Restrict enumeration to the specified positions.

    Marks every row in ``coef`` with an ``allowed_enum`` flag.  If
    ``positions`` is ``None``, all positions are allowed.

    Args:
        coef (pd.DataFrame): Coefficient DataFrame with a ``pos`` column.
        positions (list[int] | None): 0-based position indices to enumerate.
            ``None`` means enumerate all positions.

    Returns:
        pd.DataFrame: ``coef`` with an ``allowed_enum`` boolean column added.
    """
    if positions is None:
        coef["allowed_enum"] = True
    else:
        positions = positions.copy()
        positions.extend([-1, -2])  # include intercept row(s)
        coef["allowed_enum"] = coef["pos"].isin(positions)
    return coef


def limit_monos_black_white_list(coef: pd.DataFrame, black_list=None, white_list=None):
    """
    Apply white-list and black-list filters to the ``allowed_enum`` column.

    Monomers must be given as HELM symbols without brackets (e.g. ``"hS"``,
    ``"K"``).  White-list filtering only affects rows already marked
    ``allowed_enum=True``; black-list filtering unconditionally disallows the
    listed monomers.

    Args:
        coef (pd.DataFrame): Coefficient DataFrame with ``mono`` and
            ``allowed_enum`` columns.
        black_list (list[str] | None): Monomers to exclude.
        white_list (list[str] | None): If provided, only these monomers are
            allowed at enumerable positions.

    Returns:
        pd.DataFrame: Updated ``coef``.
    """
    if black_list is not None:
        coef.loc[coef["mono"].isin(black_list), "allowed_enum"] = False
    if white_list is not None:
        label = coef["mono"].isin(white_list) & coef["allowed_enum"]
        coef.loc[label, "allowed_enum"] = True
    return coef


def _get_inventory_mono_list(bb_subset=None):
    """Load monomer symbols from the configured output monomer database."""
    invn_table = pd.read_csv(MONO_DB_OUTPUT)
    return invn_table["symbol"].unique().tolist()


# ---------------------------------------------------------------------------
# Dynamic-programming enumeration
# ---------------------------------------------------------------------------

def _find_top_combinations(positions, k=1000) -> list[tuple[float, list]]:
    """
    Find the top-K monomer combinations by summed score using DP.

    Args:
        positions (list[dict]): Each dict maps monomer symbol -> score for
            one position.
        k (int): Maximum number of top combinations to retain.

    Returns:
        list[tuple[float, list]]: Top-K ``(total_score, [mono_at_pos0, ...])``
            tuples, sorted descending by score.
    """
    dp: list[list[tuple[float, list]]] = [[] for _ in range(len(positions) + 1)]
    for monomer, score in positions[0].items():
        dp[1].append((score, [monomer]))
    dp[1].sort(reverse=True, key=lambda x: x[0])
    dp[1] = dp[1][:k]

    for i in range(2, len(positions) + 1):
        candidates = []
        for prev_score, prev_combination in dp[i - 1]:
            for monomer, score in positions[i - 1].items():
                candidates.append((prev_score + score, prev_combination + [monomer]))
        candidates.sort(reverse=True, key=lambda x: x[0])
        dp[i] = candidates[:k]

    return dp[len(positions)]


def find_top_mutations_dp(
    parent_peptide: peptide_api.CycPeptide,
    coef: pd.DataFrame,
    max_prod_count: int,
):
    """
    Return the top-scoring monomer combinations using dynamic programming.

    For positions where no enumerable monomers are present, the parent
    monomer is kept fixed.

    Args:
        parent_peptide (CycPeptide): The reference/parent peptide.
        coef (pd.DataFrame): Coefficient DataFrame with ``pos``, ``mono``,
            ``coef``, and ``allowed_enum`` columns.
        max_prod_count (int): Maximum number of combinations to return.

    Returns:
        list[tuple[float, list]]: Top ``(score, [mono_at_pos0, ...])`` tuples.
    """
    positions = []
    for i in range(0, max(coef["pos"]) + 1):
        monos_at_i = coef[(coef["pos"] == i) & coef["allowed_enum"]]
        if len(monos_at_i) > 0:
            positions.append(
                dict(zip(monos_at_i["mono"], monos_at_i["coef"], strict=False))
            )
        else:
            # No enumerable monomers — fix the parent monomer at this position.
            parent_mono = parent_peptide.monomers[i]
            score = coef[(coef["pos"] == i) & (coef["mono"] == parent_mono)]["coef"].values[0]
            positions.append({parent_mono: score})

    return _find_top_combinations(positions, max_prod_count)


def _get_prediction_for_fixed_positions(
    coef_df: pd.DataFrame,
    parent_peptide: peptide_api.CycPeptide,
    enum_positions,
):
    """
    Compute the baseline prediction contribution from non-enumerated positions.

    Includes the intercept plus the coefficient for the parent monomer at
    every position that is *not* being enumerated.

    Args:
        coef_df (pd.DataFrame): Full coefficient DataFrame.
        parent_peptide (CycPeptide): Parent peptide.
        enum_positions (list[int] | None): 0-based positions being enumerated.

    Returns:
        float: Summed score for fixed positions (intercept + fixed monomers).
    """
    fixed_coef = coef_df[coef_df["pos"] < 0]["coef"].values[0]
    if enum_positions is None:
        return fixed_coef
    for i_pos, each_mono in enumerate(parent_peptide.monomers):
        if i_pos in enum_positions:
            continue
        coef_at_pos = coef_df[
            (coef_df["pos"] == i_pos) & (coef_df["mono"] == each_mono)
        ]["coef"].values[0]
        fixed_coef += coef_at_pos
    return fixed_coef


# ---------------------------------------------------------------------------
# Main enumeration API
# ---------------------------------------------------------------------------

def enum_one_parent_helm(
    parent_helm: str,
    coef_df: pd.DataFrame,
    positions=None,
    white_list=None,
    black_list=None,
    max_enum_count=100,
    pos_offset=False,
):
    """
    Enumerate the top-scoring peptide variants derived from a parent HELM.

    Args:
        parent_helm (str): HELM string of the parent peptide.
        coef_df (pd.DataFrame): Coefficient DataFrame from the FP-FW model.
        positions (list[int] | None): Positions to enumerate (0-based by
            default; use ``pos_offset=True`` if 1-based).
        white_list (list[str] | None): Only consider these monomers.
        black_list (list[str] | None): Exclude these monomers.
        max_enum_count (int): Maximum number of output peptides.
        pos_offset (bool): Set to ``True`` if ``positions`` and ``coef_df``
            use 1-based indexing (they will be shifted to 0-based internally).

    Returns:
        pd.DataFrame: Columns ``HELM`` and ``Predictions``, sorted by score
            descending.

    Raises:
        ValueError: If no enumerable monomers remain after filtering.
    """
    if 0 not in coef_df["pos"].values:
        logger.warning(
            "No 0-position found in coef_df. "
            "Pass pos_offset=True if positions are 1-based."
        )

    # Shift to 0-based indexing if needed
    if pos_offset:
        coef_df = coef_df.copy()
        coef_df["pos"] = coef_df["pos"] - 1
        if positions is not None:
            positions = [p - 1 for p in positions]

    coef_df = limit_monos_to_enum_positions(coef_df, positions)
    if coef_df["allowed_enum"].sum() == 0:
        raise ValueError(
            f"No enumerable monomers found at positions {positions}. "
            "Please check the requested positions."
        )

    coef_df = limit_monos_black_white_list(coef_df, black_list, white_list)

    parent_peptide = peptide_api.CycPeptide(parent_helm, verbose=False)
    top_mutations = find_top_mutations_dp(parent_peptide, coef_df, max_enum_count)

    baseline_prediction = _get_prediction_for_fixed_positions(
        coef_df, parent_peptide, positions
    )

    rows = []
    for enum_sum_coef, enum_monos in top_mutations:
        mut_pep = deepcopy(parent_peptide)
        for i_pos, each_mono in enumerate(enum_monos):
            mut_pep = mut_pep.mutate({i_pos: each_mono})
        rows.append({
            "HELM": mut_pep.helm,
            "Predictions": baseline_prediction + enum_sum_coef,
        })

    if pos_offset:
        coef_df["pos"] = coef_df["pos"] + 1

    return pd.DataFrame(rows)


def parse_allowed_mono_list(
    white_list: Sequence[str] | None,
    black_list: Sequence[str] | None,
    inventory_names: Sequence[str] | None,
):
    """
    Resolve the white-list for enumeration.

    Priority:
    1. If ``white_list`` is provided explicitly, use it as-is.
    2. Otherwise, load monomers from the monomer database (filtered by
       ``inventory_names`` if given; ``"all"`` loads everything).

    Args:
        white_list (Sequence[str] | None): Explicit white-list of HELM symbols.
        black_list (Sequence[str] | None): HELM symbols to exclude.
        inventory_names (Sequence[str] | None): Monomer collection names to
            restrict to.  ``None`` loads all available monomers.

    Returns:
        tuple[list[str], list[str] | None]: Resolved ``(white_list, black_list)``.
    """
    if white_list is not None:
        logger.info("white_list provided; ignoring inventory_names.")
        logger.info("white_list: %s | black_list: %s", white_list, black_list)
        return white_list, black_list

    if inventory_names is None or "all" in inventory_names:
        inventory_monos = _get_inventory_mono_list()
    else:
        inventory_monos = _get_inventory_mono_list(inventory_names)

    logger.info("white_list: %s | black_list: %s", inventory_monos, black_list)
    return inventory_monos, black_list


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate a parent HELM at the specified positions using FP-FW coefficients."
    )
    parser.add_argument("parent_helm", type=str, help="Parent HELM string.")
    parser.add_argument("coef_file", type=str, help="Path to the coefficient CSV file.")
    parser.add_argument(
        "--positions", type=int, nargs="+", help="Positions to enumerate (0-based by default)."
    )
    parser.add_argument(
        "--max_enum_count", type=int, default=100, help="Maximum number of output peptides."
    )
    parser.add_argument(
        "--pos_offset",
        action="store_true",
        help="Set this flag if positions are 1-based.",
    )
    parser.add_argument(
        "--only_inventory",
        type=str,
        nargs="+",
        help=(
            'Restrict enumeration to monomers from these collections. '
            'Use "all" for all available monomers.'
        ),
    )
    parser.add_argument(
        "--white_list",
        type=str,
        nargs="+",
        help="Only consider these monomers (HELM symbols, no brackets). E.g. hS K",
    )
    parser.add_argument(
        "--black_list",
        type=str,
        nargs="+",
        help="Exclude these monomers (HELM symbols, no brackets). E.g. hS K",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results CSV. Defaults to the same directory as the coefficient file.",
    )
    return parser


def main(args):
    coef_df = pd.read_csv(args.coef_file)
    white_list, black_list = parse_allowed_mono_list(
        args.white_list, args.black_list, args.only_inventory
    )
    enumed_df = enum_one_parent_helm(
        args.parent_helm,
        coef_df,
        args.positions,
        white_list,
        black_list,
        args.max_enum_count,
        args.pos_offset,
    )
    print(
        f"{len(enumed_df)} peptides enumerated.\n"
        f"Max prediction: {enumed_df['Predictions'].max():.4f}, "
        f"Min prediction: {enumed_df['Predictions'].min():.4f}"
    )
    if args.output_path is None:
        args.output_path = Path(args.coef_file).parent / "enumed_df.csv"
    enumed_df.to_csv(args.output_path, index=False)
    print(f"Saved enumeration results to {args.output_path}")


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(_args)

    """
    Example usage:
    python -m src.fw_lin_enum \
        'PEPTIDE1{[3AzaPhe4AcPip].W.[Phe4COOH].G.[Prot4NH2].[SbMe1Nal].D.[dNMeA].G.[Bip4CO2H].[4Pal].[Dap].[dA].[h3Pal5CN]}$PEPTIDE1,PEPTIDE1,1:R1-14:R2$$$V2.0' \
        results/fpfwa-coefficients.csv \
        --positions 1 3 --max_enum_count 10
    """
