"""
Module for peptide machine learning utilities.
"""

import re
from collections import defaultdict
from logging import getLogger
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.api import peptide_api

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional column utilities
# ---------------------------------------------------------------------------

def get_position_columns(columns, position_prefix):
    """
    Return the column names that represent aligned monomer positions.

    A valid positional column starts with ``position_prefix`` and ends with
    a pure numeric suffix (e.g. ``PEPTIDE1_3``).  Columns such as
    ``PEPTIDE1_SEQID`` or ``PEPTIDE1_QSAR_5`` are excluded.

    Args:
        columns (iterable): All column names of the aligned DataFrame.
        position_prefix (str): Prefix used for positional columns
            (e.g. ``"PEPTIDE1_"``).

    Returns:
        list[str]: Ordered list of positional column names.

    Raises:
        RuntimeError: If fewer than 3 positional columns are found.
    """
    position_columns = [c for c in columns if c.startswith(position_prefix)]
    position_columns = [
        c for c in position_columns if c.split(position_prefix)[1].isdigit()
    ]
    if len(position_columns) < 3:
        raise RuntimeError(
            f"Too few positional columns exist to model: {position_columns} "
            f"(from {columns})\nThis prefix was specified: {position_prefix}"
        )
    return position_columns


# ---------------------------------------------------------------------------
# HELM parsing helpers
# ---------------------------------------------------------------------------

def find_peptide_sections(helm_string):
    """
    Extract the monomer sequences from all PEPTIDE{} sections of a HELM string.

    Args:
        helm_string (str): A HELM notation string.

    Returns:
        list[str]: Each element is the raw monomer sequence inside one
            ``PEPTIDE{...}`` block (brackets not included).
    """
    pattern = r"(?<=PEPTIDE\d\{)[^{}]+(?=\})"
    return re.findall(pattern, helm_string)


# ---------------------------------------------------------------------------
# Reference peptide helpers
# ---------------------------------------------------------------------------

def find_ref_pept(df, refid, idcol, actcol, helmcol):
    """
    Locate the reference peptide row and return a CycPeptide object.

    Args:
        df (pd.DataFrame): Activity DataFrame.
        refid (str): Identifier of the reference compound.
        idcol (str): Column name for compound identifiers.
        actcol (str): Column name for the activity values.
        helmcol (str): Column name for HELM strings.

    Returns:
        tuple[CycPeptide, float]: Parsed reference peptide and its activity.

    Raises:
        ValueError: If ``refid`` is not found in ``df``.
    """
    ref_df = df[df[idcol] == refid]
    if len(ref_df) == 0:
        raise ValueError(
            f"Dataset does not contain reference ID '{refid}' after dropping NA values"
        )
    if len(ref_df) > 1:
        logger.warning(
            f"Reference ID '{refid}' found more than once. "
            f"Picking first occurrence: {ref_df.to_dict('records')[0]}"
        )

    ref_dict = ref_df.to_dict("records")[0]
    print(f"Reference info:\n Activity ({actcol}) : {ref_dict[actcol]}")
    print(f"HELM: {ref_dict[helmcol]}")

    ref_pept = peptide_api.CycPeptide(ref_dict[helmcol])
    for i, sym in enumerate(ref_pept.monomers):
        print(f"{sym}: {i}", end=",")
    print()

    return ref_pept, ref_dict[actcol]


# ---------------------------------------------------------------------------
# DataFrame filtering helpers
# ---------------------------------------------------------------------------

def filter_df_by_series(act_df, refid, idcol, split_series_col=None):
    """
    Keep only rows that belong to the same series as the reference compound.

    Series values are compared case-insensitively.  The ``split_series_col``
    column must be pre-processed to remove typos and mismatches.

    Args:
        act_df (pd.DataFrame): Activity DataFrame.
        refid (str): Identifier of the reference compound.
        idcol (str): Column name for compound identifiers.
        split_series_col (str | None): Column name for the series label.
            If ``None`` the DataFrame is returned unchanged.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if split_series_col:
        ref_series = act_df[act_df[idcol] == refid][split_series_col].values[0]
        rows_to_keep = act_df[split_series_col].apply(
            lambda x: str(x).lower() == str(ref_series).lower()
        )
        act_df = act_df[rows_to_keep]
    return act_df


def _find_length(helm_str):
    """
    Return the number of monomers in the (single) PEPTIDE section of a HELM string.

    Returns 0 if no PEPTIDE section is found or if multiple sections are present
    (multi-chain peptides are not yet supported).
    """
    pept_sections = find_peptide_sections(helm_str)
    if not pept_sections:
        return 0
    if len(pept_sections) > 1:
        print(f"Warning: multiple PEPTIDE sections found in {helm_str}, skipping.")
        return 0
    return len(pept_sections[0].split("."))


def filter_df_by_length(act_df, ref_pept, helm_col):
    """
    Keep only peptides whose length matches the reference peptide.

    Args:
        act_df (pd.DataFrame): Activity DataFrame with a HELM column.
        ref_pept (CycPeptide): Reference peptide whose length is the target.
        helm_col (str): Column name containing HELM strings.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    rows_to_keep = act_df[helm_col].apply(
        lambda x: _find_length(x) == len(ref_pept)
    )
    return act_df[rows_to_keep]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_top_monomer_scores(
    coef_df, pos_cols, monomer_top_n, position_prefix, fig_save_dir=None, model_str=""
):
    """
    Heatmap of top-scoring monomers by position and rank.

    Each cell shows the monomer name and its FP-FW coefficient score.

    Args:
        coef_df (pd.DataFrame): Coefficient DataFrame with columns
            ``pos`` (int), ``mono`` (str), ``coef`` (float).
        pos_cols (list[str]): Ordered position column names.
        monomer_top_n (int): Number of top monomers to show per position.
        position_prefix (str): Prefix to strip when converting column names
            to numeric position indices.
        fig_save_dir (str | None): File path to save the figure; not saved if
            ``None``.
        model_str (str): Extra string appended to the plot title.
    """
    scores_by_position_rank: dict[int, dict[str, float]] = defaultdict(dict)
    monomers_by_position_rank: dict[int, dict[str, str]] = defaultdict(dict)

    for pos in pos_cols:
        pos_num = (
            int(pos.replace(position_prefix, ""))
            if pos.startswith(position_prefix)
            else int(pos)
        )
        pos_df = coef_df[(coef_df["pos"] == pos_num) & (coef_df["mono"] != "Baseline")]
        pos_df = pos_df.sort_values("coef", ascending=False)
        monomer_scores = list(zip(pos_df["mono"], pos_df["coef"], strict=False))

        for i in range(monomer_top_n):
            if i < len(monomer_scores):
                monomer, score = monomer_scores[i]
                scores_by_position_rank[i][pos] = score
                monomers_by_position_rank[i][pos] = monomer
            else:
                scores_by_position_rank[i][pos] = np.nan
                monomers_by_position_rank[i][pos] = ""

    scores_df = pd.DataFrame(scores_by_position_rank).T
    monomers_df = pd.DataFrame(monomers_by_position_rank).T

    annot = np.empty(scores_df.shape, dtype=object)
    for i in range(scores_df.shape[0]):
        for j in range(scores_df.shape[1]):
            if not np.isnan(scores_df.iloc[i, j]):  # type: ignore[arg-type]
                monomer = cast(str, monomers_df.iloc[i, j])
                if len(monomer) > 14:
                    monomer = monomer[:12] + "..."
                score = cast(float, scores_df.iloc[i, j])
                annot[i, j] = f"{monomer}\n({score:.2f})"
            else:
                annot[i, j] = ""

    plt.figure(figsize=(len(pos_cols) * 1.5, monomer_top_n * 0.8))
    ax = sns.heatmap(
        scores_df,
        annot=annot,
        fmt="",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Score"},
    )
    ax.tick_params(axis="y", labelsize=12, rotation=0)
    plt.title(f"Top Monomers by Position{model_str}", fontsize=14, fontweight="bold")
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Rank", fontsize=12)
    plt.yticks(
        np.arange(monomer_top_n) + 0.5,
        [f"#{i + 1}" for i in range(monomer_top_n)],
        rotation=90,
        fontsize=10,
    )
    plt.tight_layout()
    if fig_save_dir is not None:
        plt.savefig(f"{fig_save_dir}", dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_scoring_monomer_frequencies(
    aligned_df,
    pos_cols,
    coef_df,
    monomer_top_n,
    position_prefix,
    fig_save_dir=None,
    model_str="",
):
    """
    Heatmap showing how often each top-scoring monomer appears in the training set.

    Each cell shows the monomer name, its raw count and percentage in the
    dataset, and its FP-FW coefficient score.

    Args:
        aligned_df (pd.DataFrame): Aligned peptide DataFrame.
        pos_cols (list[str]): Ordered position column names.
        coef_df (pd.DataFrame): Coefficient DataFrame with columns
            ``pos``, ``mono``, ``coef``.
        monomer_top_n (int): Number of top monomers per position.
        position_prefix (str): Prefix to strip for numeric index extraction.
        fig_save_dir (str | None): File path to save the figure; not saved if
            ``None``.
        model_str (str): Extra string appended to the plot title.
    """
    freq_by_position = {pos: aligned_df[pos].value_counts() for pos in pos_cols}

    freqs_by_position_rank: dict[int, dict[str, float]] = defaultdict(dict)
    monomers_by_position_rank: dict[int, dict[str, str]] = defaultdict(dict)
    scores_by_position_rank: dict[int, dict[str, float]] = defaultdict(dict)

    for pos_col in pos_cols:
        pos_num = int(pos_col.replace(position_prefix, ""))
        pos_coef_data = coef_df[
            (coef_df["pos"] == pos_num) & (coef_df["mono"] != "Baseline")
        ].copy().sort_values("coef", ascending=False)

        for i in range(monomer_top_n):
            if i < len(pos_coef_data):
                row = pos_coef_data.iloc[i]
                monomer = row["mono"]
                score = row["coef"]
                freq = freq_by_position[pos_col].get(monomer, 0)
                freqs_by_position_rank[i][pos_col] = freq
                monomers_by_position_rank[i][pos_col] = monomer
                scores_by_position_rank[i][pos_col] = score
            else:
                freqs_by_position_rank[i][pos_col] = np.nan
                monomers_by_position_rank[i][pos_col] = ""
                scores_by_position_rank[i][pos_col] = np.nan

    freqs_df = pd.DataFrame(freqs_by_position_rank).T
    monomers_df = pd.DataFrame(monomers_by_position_rank).T
    scores_df = pd.DataFrame(scores_by_position_rank).T

    annot = np.empty(freqs_df.shape, dtype=object)
    for i in range(freqs_df.shape[0]):
        for j in range(freqs_df.shape[1]):
            if not np.isnan(freqs_df.iloc[i, j]):  # type: ignore[arg-type]
                monomer = cast(str, monomers_df.iloc[i, j])
                if len(monomer) > 14:
                    monomer = monomer[:12] + "..."
                freq = cast(float, freqs_df.iloc[i, j])
                score = cast(float, scores_df.iloc[i, j])
                percent = (freq / aligned_df.shape[0]) * 100
                annot[i, j] = f"{monomer}\n({freq:.0f}, {percent:.1f}%)\nScore: {score:.2f}"
            else:
                annot[i, j] = ""

    plt.figure(figsize=(len(pos_cols) * 1.5, monomer_top_n * 0.8))
    ax = sns.heatmap(
        freqs_df,
        annot=annot,
        fmt="",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Frequency"},
    )
    plt.title(
        f"Frequencies of Top-Scoring Monomers by Position{model_str}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Score Rank", fontsize=12)
    plt.yticks(
        np.arange(monomer_top_n) + 0.5,
        [f"#{i + 1}" for i in range(monomer_top_n)],
        rotation=0,
    )
    plt.tight_layout()
    if fig_save_dir is not None:
        plt.savefig(f"{fig_save_dir}", dpi=300, bbox_inches="tight")
    plt.show()
