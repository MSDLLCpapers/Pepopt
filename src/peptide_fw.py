"""
Fingerprint-based Free Wilson Analysis (FP-FWA) for peptides.

Main entry point: :func:`free_wilson_peptides_fp`.
"""

import io
import math
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src import config
from src.fw_lin_enum import enum_one_parent_helm, parse_allowed_mono_list
from src.peptide_ml import (
    filter_df_by_length,
    filter_df_by_series,
    find_ref_pept,
    get_position_columns,
    plot_top_monomer_scores,
    plot_top_scoring_monomer_frequencies,
)
from src.PyHELM import HelmObj

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------------

def substitute_rgroup(smiles_string):
    """
    Replace R-group symbols (R1–R5) with the wildcard atom ``[*]``.

    Handles both bare ``R1`` and bracketed ``[R1]`` forms.

    Args:
        smiles_string (str): Input SMILES string.

    Returns:
        str: SMILES with all R-group tokens replaced by ``[*]``.
    """
    # Add brackets to bare R1/R2/… tokens
    modified = re.sub(r"(?<!\[)R([1-5])(?!\])", r"[R\1]", smiles_string)
    # Replace [R1], [R2], … with [*]
    modified = re.sub(r"\[R[1-5]\]", "[*]", modified)
    return modified


def fix_smiles_brackets(smiles):
    """
    Balance unmatched square brackets in a SMILES string.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        str: SMILES with balanced ``[`` / ``]`` brackets.
    """
    bracket_seq = "".join(c for c in smiles if c in "[]")
    if bracket_seq and bracket_seq.count("[") != bracket_seq.count("]"):
        if not bracket_seq.startswith("["):
            smiles = "[" + smiles
        if not bracket_seq.endswith("]"):
            smiles = smiles + "]"
    return smiles


def is_valid_smiles(smiles_string):
    """
    Return ``True`` if *smiles_string* can be parsed by RDKit.

    Args:
        smiles_string (str): SMILES to validate.

    Returns:
        bool: Whether RDKit can parse the (pre-processed) SMILES.
    """
    mol = Chem.MolFromSmiles(substitute_rgroup(fix_smiles_brackets(smiles_string)))
    return mol is not None


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def data_preproc(indf, args):
    """
    Validate and lightly preprocess the input DataFrame.

    Currently checks that the required columns are present and returns a copy
    of the DataFrame unchanged.  Extend this function to add activity
    transformations or duplicate aggregation as needed.

    Args:
        indf (pd.DataFrame): Raw input DataFrame.
        args: Parsed argument namespace with at least ``actcol`` and
            ``helmcol`` attributes.

    Returns:
        tuple[pd.DataFrame, str]: Preprocessed DataFrame and the (possibly
            renamed) activity column name.

    Raises:
        AssertionError: If ``actcol`` or ``helmcol`` are not present in the
            DataFrame.
    """
    assert args.actcol in indf.columns, (
        f"Activity column '{args.actcol}' not found in input DataFrame"
    )
    assert args.helmcol in indf.columns, (
        f"HELM column '{args.helmcol}' not found in input DataFrame"
    )
    return indf.copy(), args.actcol


# ---------------------------------------------------------------------------
# HELM parsing and alignment
# ---------------------------------------------------------------------------

def parse_helm_string(helm_string):
    """
    Parse a HELM string into a :class:`HelmObj` instance.

    Args:
        helm_string (str): HELM notation string.

    Returns:
        HelmObj: Parsed HELM object.
    """
    helm_obj = HelmObj()
    helm_obj.parse_helm(helm_string)
    return helm_obj


def align_peptides(act_df, args):
    """
    Align peptide sequences.

    Currently only the ``"prealigned"`` mode is supported for public use.
    Pairwise alignment requires internal services.

    Args:
        act_df (pd.DataFrame): DataFrame containing peptide data.
        args: Parsed argument namespace with an ``aln`` attribute.

    Returns:
        pd.DataFrame: Aligned DataFrame (unchanged for ``"prealigned"``).

    Raises:
        NotImplementedError: If ``args.aln == "pairwise"``.
        ValueError: If an unknown alignment mode is requested.
    """
    if args.aln == "pairwise":
        raise NotImplementedError(
            "Pairwise alignment requires internal services and is not available "
            "for public use. Please supply pre-aligned data."
        )
    if args.aln == "prealigned":
        return act_df.copy()
    raise ValueError(f"Unsupported alignment method: '{args.aln}'")


# ---------------------------------------------------------------------------
# Monomer extraction
# ---------------------------------------------------------------------------

def get_monomers(helm_obj):
    """
    Collect all monomers from every polymer chain in a HelmObj.

    Args:
        helm_obj (HelmObj): Parsed HELM object.

    Returns:
        list[str]: All monomer symbols (in order, across all chains).
    """
    monomers = []
    for polymer in helm_obj.get_polymer_names():
        monomers.extend(helm_obj.get_monomers_from_polymer(polymer))
    return monomers


def get_distinct_monomers(act_df):
    """
    Return the set of unique monomers observed across all peptides.

    Relies on the ``HelmObj`` column created by :func:`parse_helm_string`.

    Args:
        act_df (pd.DataFrame): DataFrame with a ``HelmObj`` column.

    Returns:
        list[str]: Unique monomer symbols (brackets stripped).
    """
    all_monomers = act_df["HelmObj"].apply(get_monomers)
    flat = [m.strip("[]") for sublist in all_monomers for m in sublist]
    return list(set(flat))


# ---------------------------------------------------------------------------
# Monomer SMILES lookup
# ---------------------------------------------------------------------------

def get_monomer_smiles_local(distinct_monomers_list, monomer_db_path):
    """
    Look up SMILES for a list of monomers from a local CSV database.

    Matching is attempted first by ``symbol``, then by ``aliases`` (comma-
    separated).  Unmatched monomers that are themselves valid SMILES are
    treated as their own structure.

    Args:
        distinct_monomers_list (list[str]): Monomer symbols to look up.
        monomer_db_path (str | Path): Path to the monomer database CSV.
            Must contain at least ``symbol`` and ``Partial_SMILES`` columns.

    Returns:
        pd.DataFrame: Columns ``monomer`` and ``smiles`` for all resolved
            monomers.  Unresolvable monomers are silently dropped.
    """
    monomer_db = pd.read_csv(monomer_db_path)
    monomer_smiles = {}
    unmatched = []

    for monomer in distinct_monomers_list:
        # Exact match on symbol
        symbol_match = monomer_db[monomer_db["symbol"] == monomer]
        if not symbol_match.empty:
            monomer_smiles[monomer] = symbol_match.iloc[0]["Partial_SMILES"]
            continue

        # Match via aliases column
        matched_via_alias = False
        if "aliases" in monomer_db.columns:
            for _, row in monomer_db.iterrows():
                if pd.notna(row["aliases"]):
                    aliases = [a.strip() for a in row["aliases"].split(",")]
                    if monomer in aliases:
                        monomer_smiles[monomer] = row["Partial_SMILES"]
                        matched_via_alias = True
                        break

        if not matched_via_alias:
            unmatched.append(monomer)

    monomer_smiles_df = pd.DataFrame(
        list(monomer_smiles.items()), columns=["monomer", "smiles"]
    )

    # Treat unmatched monomers that are valid SMILES as their own structure
    if unmatched:
        valid_smiles_extra = [m for m in unmatched if is_valid_smiles(m)]
        unmatched_remaining = [m for m in unmatched if not is_valid_smiles(m)]
        if valid_smiles_extra:
            extra_df = pd.DataFrame(
                {"monomer": valid_smiles_extra, "smiles": valid_smiles_extra}
            )
            monomer_smiles_df = pd.concat([monomer_smiles_df, extra_df], ignore_index=True)
        unmatched = unmatched_remaining

    print(f"Total distinct monomers: {len(distinct_monomers_list)}")
    print(f"Matched monomers (incl. valid SMILES): {len(monomer_smiles_df)}")
    print(f"Unmatched monomers: {len(unmatched)}")
    if unmatched:
        print(f"  {unmatched}")

    return monomer_smiles_df


# ---------------------------------------------------------------------------
# Descriptor calculation
# ---------------------------------------------------------------------------

def calc_monomer_desc(smiles, feat_type, fp_radius=2, fp_nbits=2048, use_chirality=True):
    """
    Compute molecular descriptors or fingerprints for a single SMILES string.

    Args:
        smiles (str): SMILES string for the monomer fragment.
        feat_type (str): Feature type — one of ``"ecfp"``, ``"ecfp_counts"``,
            or ``"rdk_desc"``.
        fp_radius (int): Morgan fingerprint radius (default 2).
        fp_nbits (int): Number of fingerprint bits (default 2048).
        use_chirality (bool): Include chirality in fingerprints (default True).

    Returns:
        dict | None: ``{feature_name: value}`` mapping, or ``None`` if the
            molecule could not be parsed or an error occurred.
    """
    smiles = substitute_rgroup(fix_smiles_brackets(smiles))
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        result = {}
        if feat_type == "rdk_desc":
            for name, func in Descriptors._descList:
                try:
                    result[name] = func(mol)
                except Exception:
                    result[name] = None

        elif feat_type == "ecfp":
            gen = GetMorganGenerator(
                radius=fp_radius, fpSize=fp_nbits, includeChirality=use_chirality
            )
            fp = gen.GetFingerprint(mol)
            result = {f"FP_{i}": bit for i, bit in enumerate(fp)}

        elif feat_type == "ecfp_counts":
            gen = GetMorganGenerator(
                radius=fp_radius,
                fpSize=fp_nbits,
                includeChirality=use_chirality,
            )
            arr = gen.GetCountFingerprintAsNumPy(mol)
            result = {f"FP_{i}": int(count) for i, count in enumerate(arr)}

        return result

    except Exception:
        return None


def integrate_monomer_desc(
    monomer_smiles_df,
    feat_type,
    fp_radius=2,
    fp_nbits=2048,
    use_chirality=True,
    norm=False,
):
    """
    Compute descriptors for every monomer and join them into one DataFrame.

    Args:
        monomer_smiles_df (pd.DataFrame): DataFrame with a ``smiles`` column.
        feat_type (str): Feature type (see :func:`calc_monomer_desc`).
        fp_radius (int): Morgan fingerprint radius (default 2).
        fp_nbits (int): Number of fingerprint bits (default 2048).
        use_chirality (bool): Include chirality (default True).
        norm (bool): Z-score normalise numeric descriptor columns (default False).

    Returns:
        pd.DataFrame: Original monomer columns plus computed descriptor columns.
            Rows where descriptor calculation failed are dropped.
    """
    descriptors_series = monomer_smiles_df["smiles"].apply(
        lambda smi: calc_monomer_desc(
            smi,
            feat_type=feat_type,
            fp_radius=fp_radius,
            fp_nbits=fp_nbits,
            use_chirality=use_chirality,
        )
    )
    valid_mask = descriptors_series.notna()
    descriptors_df = pd.DataFrame(descriptors_series[valid_mask].tolist())
    valid_monomers_df = monomer_smiles_df[valid_mask]

    if norm:
        numeric_cols = descriptors_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            std_val = descriptors_df[col].std()
            if std_val > 0:
                descriptors_df[col] = (
                    (descriptors_df[col] - descriptors_df[col].mean()) / std_val
                )

    return pd.concat([valid_monomers_df, descriptors_df], axis=1)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_monomer_desc(monomer_desc_df, top_n=500, exclude_cols=None):
    """
    Select the top descriptor columns by variance.

    Args:
        monomer_desc_df (pd.DataFrame): DataFrame of monomer descriptors.
        top_n (int | None): Number of highest-variance features to keep.
            Pass ``None`` to keep all valid features.
        exclude_cols (list[str] | None): Columns to skip (default:
            ``["monomer", "smiles"]``).

    Returns:
        list[str]: Selected feature column names.
    """
    if exclude_cols is None:
        exclude_cols = ["monomer", "smiles"]

    descriptor_cols = [c for c in monomer_desc_df.columns if c not in exclude_cols]

    variances = {}
    for col in descriptor_cols:
        if monomer_desc_df[col].isna().all():
            continue
        if not pd.api.types.is_numeric_dtype(monomer_desc_df[col]):
            continue
        var = monomer_desc_df[col].var(skipna=True)
        if pd.isna(var) or var == 0:
            continue
        variances[col] = round(var, 10)

    if top_n is None:
        return list(variances.keys())

    sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    if not sorted_features:
        return []
    cutoff = sorted_features[min(top_n, len(sorted_features)) - 1][1]
    return [f for f, v in sorted_features if v >= cutoff]


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def concat_desc(aligned_df, monomer_desc_df, pos_cols, idcol, feature_cols=None):
    """
    Build a position-specific feature matrix by concatenating monomer descriptors.

    For each peptide, the descriptor vector for the monomer at each position is
    concatenated in order.  Gap positions (``"-"``) are represented by zeros.
    Peptides with any unresolvable monomer are silently dropped.

    Args:
        aligned_df (pd.DataFrame): Aligned peptide DataFrame.
        monomer_desc_df (pd.DataFrame): Monomer descriptor table with a
            ``monomer`` column.
        pos_cols (list[str]): Ordered positional column names.
        idcol (str): Column name for peptide identifiers.
        feature_cols (list[str] | None): Descriptor columns to use.  If
            ``None``, all non-``monomer``/non-``smiles`` columns are used.

    Returns:
        pd.DataFrame: Feature matrix with one row per valid peptide, plus an
            ``idcol`` column for the peptide identifier.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in monomer_desc_df.columns if c not in ("monomer", "smiles")
        ]

    print(f"Using {len(feature_cols)} features for concatenation")

    all_features = []
    peptide_ids = []
    missing_monomers: set[str] = set()

    for _, row in aligned_df.iterrows():
        row_features = []
        valid = True

        for col in pos_cols:
            monomer = row[col]
            if monomer == "-":
                row_features.extend([0] * len(feature_cols))
            else:
                mono_row = monomer_desc_df[monomer_desc_df["monomer"] == monomer]
                if mono_row.empty:
                    missing_monomers.add(monomer)
                    valid = False
                    break
                vals = np.nan_to_num(mono_row[feature_cols].values[0], nan=0.0)
                row_features.extend(vals)

        if valid:
            all_features.append(row_features)
            peptide_ids.append(row[idcol])

    if missing_monomers:
        print(f"Warning: {len(missing_monomers)} monomers not found in descriptor table:")
        for m in sorted(missing_monomers):
            print(f"  - {m}")

    concat_columns = [
        f"pos{i}_{feat}" for i, _ in enumerate(pos_cols) for feat in feature_cols
    ]
    result_df = pd.DataFrame(all_features, columns=concat_columns)
    result_df[idcol] = peptide_ids
    return result_df


def filter_fp_features(x, fp_min_freq=5):
    """
    Drop fingerprint columns that are set to 1 in fewer than ``fp_min_freq`` peptides.

    Args:
        x (pd.DataFrame): Binary fingerprint feature matrix.  Column names
            must follow the ``"posN_FP_K"`` convention.
        fp_min_freq (int): Minimum number of peptides in which a bit must be
            set to 1 to be retained (default 5).

    Returns:
        tuple[dict[str, list[str]], dict[str, int]]: A mapping
            ``{pos_key: [kept_col, ...]}`` and a summary
            ``{pos_key: n_kept_cols}``.
    """
    kept_by_pos: dict[str, list[str]] = defaultdict(list)
    for col in x.columns:
        position = col.split("_FP_")[0]
        if (x[col] == 1).sum() >= fp_min_freq:
            kept_by_pos[position].append(col)
    summary = {pos: len(cols) for pos, cols in kept_by_pos.items()}
    return kept_by_pos, summary


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(args):
    """
    Instantiate the regression model specified in ``args.model``.

    Args:
        args: Namespace with a ``model`` attribute (``"ridge"`` or
            ``"bayesian_ridge"``) and an ``alpha`` attribute for Ridge.

    Returns:
        Fitted-ready sklearn estimator.
    """
    if args.model == "ridge":
        return Ridge(alpha=args.alpha)
    if args.model == "bayesian_ridge":
        return BayesianRidge()
    raise ValueError(f"Unknown model type: '{args.model}'")


# ---------------------------------------------------------------------------
# Coefficient calculation
# ---------------------------------------------------------------------------

def calculate_coef_df(
    aligned_df,
    pos_cols,
    monomer_desc_df,
    kept_columns_by_pos,
    model,
    act_col,
    idcol,
    position_prefix,
):
    """
    Translate model coefficients into per-monomer scores for each position.

    Uses the dot product of per-monomer fingerprint bits and the corresponding
    model weights to assign a scalar score to every (position, monomer) pair.

    Args:
        aligned_df (pd.DataFrame): Aligned peptide DataFrame (used for
            activity annotations).
        pos_cols (list[str]): Ordered positional column names.
        monomer_desc_df (pd.DataFrame): Monomer descriptor table.
        kept_columns_by_pos (dict[str, list[str]]): Mapping from ``"posN"``
            to the list of retained feature column names.
        model: Trained sklearn model with ``coef_`` and ``intercept_``
            attributes.
        act_col (str | None): Activity column name, used for annotations.
            Pass ``None`` to skip annotation.
        idcol (str): Identifier column name.
        position_prefix (str): Prefix of positional columns (e.g.
            ``"PEPTIDE1_"``).

    Returns:
        pd.DataFrame: Coefficient DataFrame with columns ``coef``, ``pos``,
            ``mono``, plus optional annotation columns.
    """
    intercept = model.intercept_
    coef_value = float(intercept) if np.ndim(intercept) == 0 else float(intercept[0])
    coef_data = [
        {
            "coef": coef_value,
            "pos": -1,
            "mono": "Baseline",
            "num": "",
            "min_act": "",
            "max_act": "",
            "ids": "",
            "acts": "",
            "HELMs": "",
        }
    ]

    # Determine 0-based offset for position indexing
    pattern = re.compile(rf"^{re.escape(position_prefix)}(\d+)$")
    suffixes = [int(pattern.match(col).group(1)) for col in pos_cols]
    offset = min(suffixes)

    # Build mapping: pos_num (0-based) → (coef_start, coef_end)
    sorted_pos_keys = sorted(
        kept_columns_by_pos.keys(), key=lambda k: int(k.replace("pos", ""))
    )
    cum = np.cumsum([0] + [len(kept_columns_by_pos[k]) for k in sorted_pos_keys])
    pos_to_coef_range = {
        int(k.replace("pos", "")): (int(cum[i]), int(cum[i + 1]))
        for i, k in enumerate(sorted_pos_keys)
    }

    for pos_col in pos_cols:
        pos = int(pos_col.replace(position_prefix, ""))
        pos_key = f"pos{pos - offset}"
        if pos_key not in kept_columns_by_pos:
            continue

        start_idx, end_idx = pos_to_coef_range[pos - offset]
        weights = model.coef_[start_idx:end_idx]

        # Map kept x_fp columns (e.g. "pos1_FP_7") → monomer-space names ("FP_7")
        kept_fp_cols_monomer = [
            f"FP_{col.split('_FP_')[1]}"
            for col in kept_columns_by_pos[pos_key]
        ]

        for monomer in aligned_df[pos_col].unique():
            if monomer == "-":
                fp = [0] * len(kept_fp_cols_monomer)
            else:
                mono_row = monomer_desc_df[monomer_desc_df["monomer"] == monomer]
                fp = mono_row[kept_fp_cols_monomer].values[0].tolist() if not mono_row.empty else None

            if fp is None:
                continue

            score = float(np.dot(fp, weights))

            # Annotation
            if act_col is not None:
                example_df = aligned_df[aligned_df[pos_col] == monomer].sort_values(
                    act_col, ascending=False
                )
                top_n = 10
                idstr = ",".join(example_df.head(top_n)[idcol].astype(str))
                actstr = ",".join(
                    str(v) for v in example_df.head(top_n)[act_col]
                )
                helm_cols = [c for c in example_df.columns if c.startswith("HELM")]
                helmstr = (
                    ",".join(example_df[helm_cols[0]].astype(str))
                    if helm_cols else ""
                )
                coef_data.append({
                    "coef": score,
                    "pos": pos,
                    "mono": str(monomer),
                    "num": int(len(example_df)),
                    "min_act": example_df[act_col].min() if not example_df.empty else None,
                    "max_act": example_df[act_col].max() if not example_df.empty else None,
                    "ids": idstr,
                    "acts": actstr,
                    "HELMs": helmstr,
                })
            else:
                coef_data.append({"coef": score, "pos": pos, "mono": str(monomer)})

    coef_df = pd.DataFrame(coef_data)
    coef_df["pos"] = coef_df["pos"].astype(int)
    coef_df["coef"] = pd.to_numeric(coef_df["coef"], errors="coerce")
    return coef_df.sort_values(["pos", "coef"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Substructure visualisation
# ---------------------------------------------------------------------------

def plot_top_morgan_bits(
    coef_df,
    pos_cols,
    model,
    kept_columns_by_pos,
    monomer_desc_df,
    x_fp,
    position_prefix="PEPTIDE_",
    position_idx=1,
    top_n_monomers=3,
    top_n_bits=3,
    n_cols=5,
    fig_save_dir=None,
    fp_radius=2,
    fp_nbits=2048,
    use_chirality=True,
    pos_idx_offset=True,
):
    """
    Visualise the most important Morgan fingerprint bits for top monomers.

    For the requested position, the function:
    1. Identifies the top-scoring monomers.
    2. Draws the monomer structure.
    3. For each monomer, ranks its active bits by weighted contribution and
       draws the top-N bit environments.

    Args:
        coef_df (pd.DataFrame): Coefficient DataFrame.
        pos_cols (list[str]): Ordered positional column names.
        model: Trained sklearn model with ``coef_`` attribute.
        kept_columns_by_pos (dict[str, list[str]]): Kept FP columns per
            position (``"posN"`` → column list).
        monomer_desc_df (pd.DataFrame): Monomer descriptor table.
        x_fp (pd.DataFrame): Full binary fingerprint feature matrix.
        position_prefix (str): Positional column prefix (default ``"PEPTIDE_"``).
        position_idx (int): Position to analyse (1-based if
            ``pos_idx_offset=True``, else 0-based).
        top_n_monomers (int): Number of top monomers to visualise.
        top_n_bits (int): Number of top bits to draw per monomer.
        n_cols (int): Subplot grid columns.
        fig_save_dir (str | Path | None): Directory for saving figures.
        fp_radius (int): Morgan fingerprint radius.
        fp_nbits (int): Fingerprint bit count.
        use_chirality (bool): Include chirality.
        pos_idx_offset (bool): Treat ``position_idx`` as 1-based.
    """
    if pos_idx_offset:
        position_idx = position_idx - 1
    if not (0 <= position_idx < len(pos_cols)):
        raise IndexError(
            f"position_idx {position_idx} out of range for "
            f"pos_cols of length {len(pos_cols)}"
        )

    position = pos_cols[position_idx]
    pos_num = (
        int(position.replace(position_prefix, ""))
        if position.startswith(position_prefix)
        else int(position)
    )
    pos_key = f"pos{pos_num}"

    if pos_key not in kept_columns_by_pos or not kept_columns_by_pos[pos_key]:
        print(f"No kept fingerprint columns for {pos_key}; skipping.")
        return

    pos_df = coef_df[(coef_df["pos"] == pos_num) & (coef_df["mono"] != "Baseline")].copy()
    if pos_df.empty:
        print(f"No monomer coefficients for position {pos_num}.")
        return
    pos_df = pos_df.sort_values("coef", ascending=False)
    top_monomers = list(zip(pos_df["mono"], pos_df["coef"]))[:top_n_monomers]

    kept_fp_cols_with_pos = kept_columns_by_pos[pos_key]
    kept_fp_names = [f"FP_{col.split('_FP_')[1]}" for col in kept_fp_cols_with_pos]

    # Locate weights for this position in the flat coefficient array
    all_pos_keys_sorted = sorted(
        kept_columns_by_pos.keys(), key=lambda k: int(k.replace("pos", ""))
    )
    counts = [len(kept_columns_by_pos[k]) for k in all_pos_keys_sorted]
    cum = np.cumsum([0] + counts)
    pos_idx_in_sorted = all_pos_keys_sorted.index(pos_key)
    start_idx = int(cum[pos_idx_in_sorted])
    end_idx = int(cum[pos_idx_in_sorted + 1])

    coef_arr = model.coef_
    weights = coef_arr[start_idx:end_idx] if np.ndim(coef_arr) == 1 else coef_arr[0][start_idx:end_idx]

    if len(weights) != len(kept_fp_names):
        raise ValueError(
            f"Weight count {len(weights)} ≠ kept FP count {len(kept_fp_names)} "
            f"for {pos_key}."
        )

    bit_weights = dict(zip(kept_fp_names, weights))

    # Bit frequencies across the dataset
    if x_fp is not None:
        bit_counts = x_fp[kept_fp_cols_with_pos].values.sum(axis=0)
        total_samples = x_fp.shape[0]
        bit_freq_map = dict(
            zip(kept_fp_names, (bit_counts / total_samples) * 100)
        )
    else:
        bit_freq_map = None

    for monomer, score in top_monomers:
        if score <= 0:
            print(f"\nSkipping {monomer} (score={score:.4f} ≤ 0)")
            continue

        print(f"\n=== Monomer: {monomer}  Score: {score:.4f} ===")
        mono_row = monomer_desc_df[monomer_desc_df["monomer"] == monomer]
        if mono_row.empty:
            print(f"  No SMILES found for {monomer}")
            continue

        smiles = substitute_rgroup(mono_row["smiles"].values[0])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Could not parse SMILES for {monomer}: {smiles}")
            continue

        # Draw monomer structure
        plt.figure(figsize=(6, 4))
        plt.imshow(Draw.MolToImage(mol, size=(400, 300)))
        plt.title(f"Monomer: {monomer}  (Score: {score:.4f})")
        plt.axis("off")
        plt.tight_layout()
        if fig_save_dir is not None:
            plt.savefig(
                Path(fig_save_dir) / f"{position}_{monomer}_structure.png",
                dpi=300, bbox_inches="tight",
            )
        plt.show()

        # Compute per-bit weighted contributions
        ao = AdditionalOutput()
        ao.CollectBitInfoMap()
        gen = GetMorganGenerator(
            radius=fp_radius, fpSize=fp_nbits, includeChirality=use_chirality
        )
        fp = gen.GetFingerprint(mol, additionalOutput=ao)
        bit_info = ao.GetBitInfoMap()
        fp_df = pd.DataFrame([{f"FP_{i}": b for i, b in enumerate(fp)}])

        weighted_fp = {
            bit: (fp_df[bit].values if bit in fp_df.columns else np.array([0.0]))
                 * bit_weights[bit]
            for bit in kept_fp_names
        }

        score_check = float(sum(v[0] for v in weighted_fp.values()))
        if abs(score_check - score) > 1e-3:
            raise ValueError(
                "Re-computed score does not match stored score — "
                "verify FP configuration (radius, nBits, chirality)."
            )

        sorted_bits = sorted(weighted_fp.items(), key=lambda x: x[1], reverse=True)
        top_bits = [(b, float(v)) for b, v in sorted_bits if float(v) != 0.0][:top_n_bits]

        # Draw top-bit substructure images
        n_rows = math.ceil(max(1, len(top_bits)) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, (bit_name, weighted_value) in enumerate(top_bits):
            bit_idx = int(bit_name.split("_")[1])
            weight = bit_weights[bit_name]
            frequency = bit_freq_map[bit_name] if bit_freq_map is not None else None
            ax = axes[i // n_cols, i % n_cols]

            if bit_idx in bit_info:
                raw_img = Draw.DrawMorganBit(mol, bit_idx, bit_info, useSVG=False)
                bit_img = (
                    raw_img if isinstance(raw_img, PILImage.Image)
                    else PILImage.open(io.BytesIO(raw_img))
                )
                ax.imshow(bit_img)
                title = f"Bit {bit_name}\nWeight: {weight:.4f}"
                if frequency is not None:
                    title += f"\nFrequency: {frequency:.1f}%"
                ax.set_title(title)
            else:
                msg = f"Bit {bit_name} not found\nWeight: {weight:.4f}"
                if frequency is not None:
                    msg += f"\nFrequency: {frequency:.1f}%"
                ax.text(0.5, 0.5, msg, ha="center", va="center")
            ax.axis("off")

        # Turn off unused subplots
        for i in range(len(top_bits), n_rows * n_cols):
            axes[i // n_cols, i % n_cols].axis("off")

        plt.suptitle(f"Top {top_n_bits} Morgan Bits for {monomer}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.7)
        if fig_save_dir is not None:
            fig_path = Path(fig_save_dir) / f"{position}_{monomer}_top{top_n_bits}_bits.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def _gen_train_test_ids(aligned_df, n_split, index_col):
    """
    Generate cross-validation and time-based train/test splits.

    Returns three pairs of (train_id_lists, test_id_lists):
    - ``cv``: shuffled K-fold
    - ``time``: contiguous K-fold (circular)
    - ``time_order``: expanding-window time-ordered (for reference, unused in
      the evaluation loop but kept for potential future use)
    """
    n_samples = len(aligned_df)
    fold_size = n_samples // n_split
    dataset_idx = aligned_df[index_col].tolist()

    # Circular time-based split
    train_ids_time, test_ids_time = [], []
    for fold in range(n_split):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_split - 1 else n_samples
        test_ids = aligned_df.iloc[test_start:test_end][index_col].values
        train_ids = aligned_df.iloc[
            list(range(0, test_start)) + list(range(test_end, n_samples))
        ][index_col].values
        train_ids_time.append(train_ids)
        test_ids_time.append(test_ids)

    # Shuffled K-fold
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
    train_ids_cv, test_ids_cv = [], []
    for train_idx, test_idx in kf.split(dataset_idx):
        train_ids_cv.append([dataset_idx[i] for i in train_idx])
        test_ids_cv.append([dataset_idx[i] for i in test_idx])

    return train_ids_time, test_ids_time, train_ids_cv, test_ids_cv


def _calc_metrics(y_test, y_pred):
    """Compute NRMSE, R², Spearman, and Pearson metrics."""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse = rmse / (np.std(y_test) + 1e-6)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    return nrmse, r2, spearman_corr, pearson_corr


def evaluate_fpfw(x, y, aligned_df, args, fig_save_dir="./mdl_eval"):
    """
    Evaluate the FP-FW model with cross-validation and time-based splits.

    For each fold and each split type, a fresh model is trained on the
    training set and evaluated on the held-out test set.  Scatter plots and a
    summary bar chart are saved to ``fig_save_dir``.

    Args:
        x (pd.DataFrame): Feature matrix indexed by compound ID.
        y (pd.DataFrame): Target values indexed by compound ID.
        aligned_df (pd.DataFrame): Aligned peptide DataFrame (used for
            time-ordering by numeric ID component).
        args: Argument namespace (``idcol``, ``model``, ``alpha``).
        fig_save_dir (str): Directory to save evaluation figures.

    Returns:
        pd.DataFrame: Performance metrics with columns
            ``split_type``, ``fold``, ``nrmse``, ``r2``,
            ``spearman_corr``, ``pearson_corr``.
    """
    Path(fig_save_dir).mkdir(parents=True, exist_ok=True)

    # Sort by numeric component of compound IDs for time-based splits
    aligned_df = aligned_df.copy()
    aligned_df["_ID_Int"] = (
        aligned_df[args.idcol].str.extract(r"^[A-Za-z]+[-_]0*(\d+)$").astype(float)
    )
    aligned_df = aligned_df.sort_values("_ID_Int")

    n_split = 5
    train_ids_time, test_ids_time, train_ids_cv, test_ids_cv = _gen_train_test_ids(
        aligned_df, n_split, args.idcol
    )

    dataset_idx = x.index.tolist()
    metrics_rows = []

    for split_type, train_indices, test_indices in [
        ("cv", train_ids_cv, test_ids_cv),
        ("time", train_ids_time, test_ids_time),
    ]:
        for fold in range(n_split):
            train_idx = [i for i in train_indices[fold] if i in dataset_idx]
            test_idx = [i for i in test_indices[fold] if i in dataset_idx]

            x_train, y_train = x.loc[train_idx].values, y.loc[train_idx].values.flatten()
            x_test, y_test = x.loc[test_idx].values, y.loc[test_idx].values.flatten()

            mdl = build_model(args)
            mdl.fit(x_train, y_train)
            y_pred = mdl.predict(x_test).flatten()

            nrmse, r2, sp, pe = _calc_metrics(y_test, y_pred)
            metrics_rows.append({
                "split_type": split_type, "fold": fold,
                "nrmse": nrmse, "r2": r2,
                "spearman_corr": sp, "pearson_corr": pe,
            })

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, s=50)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{split_type.upper()} — Fold {fold + 1}: Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
            ax.text(
                0.05, 0.95,
                f"R² = {r2:.3f}\nSpearman = {sp:.3f}\nPearson = {pe:.3f}",
                transform=ax.transAxes, verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(
                Path(fig_save_dir) / f"scatter_{split_type}_fold_{fold + 1}.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

    performance_metrics = pd.DataFrame(metrics_rows)

    # Summary bar chart
    metrics_to_plot = ["r2", "spearman_corr", "pearson_corr"]
    cv_means = performance_metrics[performance_metrics["split_type"] == "cv"][metrics_to_plot].mean()
    time_means = performance_metrics[performance_metrics["split_type"] == "time"][metrics_to_plot].mean()
    cv_stds = performance_metrics[performance_metrics["split_type"] == "cv"][metrics_to_plot].std()
    time_stds = performance_metrics[performance_metrics["split_type"] == "time"][metrics_to_plot].std()

    x_pos = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x_pos - width / 2, cv_means, width, label="CV Split",
                   yerr=cv_stds, capsize=5, alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, time_means, width, label="Time Split",
                   yerr=time_stds, capsize=5, alpha=0.8)
    ax.set_xlabel("Performance Metrics")
    ax.set_ylabel("Score")
    ax.set_title(f"Average model performance over {n_split}-fold CV and time split")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["R²", "Spearman", "Pearson"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    def _add_labels(bars, values):
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    _add_labels(bars1, cv_means)
    _add_labels(bars2, time_means)
    plt.tight_layout()
    plt.savefig(Path(fig_save_dir) / "mdl_perform_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    return performance_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def free_wilson_peptides_fp(args, out_prefix: str = "fpfwa"):
    """
    Run the full FP-based Free Wilson Analysis pipeline.

    Steps:
    1. Load and preprocess the input CSV.
    2. (Optionally) filter by series and/or fixed peptide length.
    3. Parse HELM strings; look up monomer SMILES from the local database.
    4. Compute and select molecular fingerprint descriptors.
    5. Build the feature matrix and (optionally) filter low-frequency bits.
    6. (Optionally) evaluate model with CV and time-based splits.
    7. Train the final model on all data.
    8. Calculate per-monomer coefficients.
    9. Enumerate top peptides from the training-set monomers.
    10. Repeat steps 8–9 for the full monomer output database.
    11. Generate all visualisations.

    Args:
        args: Parsed argument namespace (see ``local_run_fpfw.py``).
        out_prefix (str): Prefix for output file names (default ``"fpfwa"``).
    """
    results_save_dir = Path(args.results_dir)
    results_save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    indf = pd.read_csv(args.incsv)
    act_df, actcol = data_preproc(indf, args)
    args.actcol = actcol
    act_df = act_df.dropna()

    # ------------------------------------------------------------------
    # 2. Reference peptide and optional filtering
    # ------------------------------------------------------------------
    ref_pept, ref_act = find_ref_pept(
        act_df, args.refid, args.idcol, args.actcol, args.helmcol
    )

    if args.split_series_col:
        before = len(act_df)
        act_df = filter_df_by_series(act_df, args.refid, args.idcol, args.split_series_col)
        args.aln_min_sim = min(0.3, args.aln_min_sim)
        print(f"Series filter: {before} → {len(act_df)} compounds.")

    if args.fixed_length:
        before = len(act_df)
        act_df = filter_df_by_length(act_df, ref_pept, args.helmcol)
        print(f"Length filter: {before} → {len(act_df)} compounds.")

    # ------------------------------------------------------------------
    # 3. Parse HELM and align
    # ------------------------------------------------------------------
    act_df["HelmObj"] = act_df[args.helmcol].apply(parse_helm_string)
    aligned_df = align_peptides(act_df, args)
    aligned_df[args.actcol] = pd.to_numeric(aligned_df[args.actcol], errors="coerce")
    pos_cols = get_position_columns(aligned_df.columns.values, args.pos_prefix)

    # ------------------------------------------------------------------
    # 4. Monomer SMILES lookup and descriptor calculation
    # ------------------------------------------------------------------
    distinct_monomers = get_distinct_monomers(act_df)
    monomer_smiles_df = get_monomer_smiles_local(distinct_monomers, config.MONOMER_DB_INPUT)

    monomer_desc_df = integrate_monomer_desc(
        monomer_smiles_df,
        args.desc_type,
        fp_radius=args.fp_radius,
        fp_nbits=args.fp_len,
        use_chirality=True,
        norm=args.desc_norm,
    )
    selected_desc = select_monomer_desc(monomer_desc_df, args.fp_sel_n)

    # ------------------------------------------------------------------
    # 5. Feature matrix
    # ------------------------------------------------------------------
    x = concat_desc(aligned_df, monomer_desc_df, pos_cols, args.idcol, selected_desc)
    x.set_index(args.idcol, inplace=True)
    print(f"FP feature matrix shape: {x.shape}")

    kept_columns_by_pos, summary = filter_fp_features(x, fp_min_freq=args.fp_min_freq)
    if args.fp_min_freq > 0:
        print(f"Minimum frequency filter ({args.fp_min_freq}):")
        for pos_key in sorted(summary, key=lambda k: int(k.replace("pos", ""))):
            print(f"  {pos_key}: {summary[pos_key]} columns kept")
        all_kept = [c for cols in kept_columns_by_pos.values() for c in cols]
        print(f"  Total kept: {len(all_kept)} / {len(x.columns)} "
              f"({len(all_kept) / len(x.columns) * 100:.1f}%)")
        x = x[all_kept]
        print(f"  Filtered feature matrix shape: {x.shape}")

    dataset_idx = x.index.tolist()
    y = aligned_df.set_index(args.idcol)[[args.actcol]]

    # ------------------------------------------------------------------
    # 6. Optional evaluation
    # ------------------------------------------------------------------
    if args.evaluation:
        print("Starting model evaluation …")
        perf = evaluate_fpfw(
            x, y, aligned_df, args,
            fig_save_dir=str(results_save_dir / "mdl_eval"),
        )
        perf.to_csv(results_save_dir / f"{out_prefix}_mdl_performance.csv", index=False)

    # ------------------------------------------------------------------
    # 7–8. Train final model and compute coefficients
    # ------------------------------------------------------------------
    model = build_model(args)
    model.fit(x, y.loc[dataset_idx])
    coef_df = calculate_coef_df(
        aligned_df=aligned_df,
        pos_cols=pos_cols,
        monomer_desc_df=monomer_desc_df,
        kept_columns_by_pos=kept_columns_by_pos,
        model=model,
        act_col=args.actcol,
        idcol=args.idcol,
        position_prefix=args.pos_prefix,
    )
    monomer_smiles_df = monomer_smiles_df.rename(columns={"monomer": "mono"})
    coef_df = coef_df.merge(monomer_smiles_df, on="mono", how="left")
    coef_df = coef_df.drop(columns=["HELMs"])
    coef_df.to_csv(results_save_dir / f"{out_prefix}-coefficients.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Enumerate from training-set monomers
    # ------------------------------------------------------------------
    pos_offset = 0 if 0 in coef_df["pos"].values else 1
    white_list, black_list = parse_allowed_mono_list(
        args.white_list, args.black_list, args.only_inventory
    )

    if args.max_enum_out > 0:
        enumed_df = enum_one_parent_helm(
            ref_pept.helm,
            coef_df,
            args.enum_positions,
            white_list,
            black_list,
            args.max_enum_out,
            pos_offset=pos_offset,
        )
        print(
            f"{len(enumed_df)} peptides enumerated (training monomers). "
            f"Pred range: [{enumed_df['Predictions'].min():.4f}, "
            f"{enumed_df['Predictions'].max():.4f}]"
        )
        # Add per-position monomer columns
        n_elements = enumed_df["HELM"].values[0].count(".")
        for iseq in range(n_elements):
            enumed_df[f"PEPTIDE1_{iseq}"] = enumed_df["HELM"].apply(
                lambda s, idx=iseq: s.split("PEPTIDE1{")[1].split("}")[0].split(".")[idx]
            )
        enumed_df.to_csv(
            results_save_dir / f"{out_prefix}_best_{args.max_enum_out}.csv", index=False
        )

    # ------------------------------------------------------------------
    # 10. Coefficients and enumeration for the full monomer database
    # ------------------------------------------------------------------
    monomer_db_filt = pd.read_csv(config.MONO_DB_OUTPUT)
    monomer_db_filt_smiles_df = get_monomer_smiles_local(
        monomer_db_filt["symbol"].tolist(), config.MONO_DB_OUTPUT
    )
    monomer_db_filt_desc_df = integrate_monomer_desc(
        monomer_db_filt_smiles_df,
        args.desc_type,
        fp_radius=args.fp_radius,
        fp_nbits=args.fp_len,
        use_chirality=True,
        norm=args.desc_norm,
    )

    # Build a dummy aligned_df covering all DB monomers (one row per monomer)
    aligned_df_all = pd.DataFrame(columns=pos_cols)
    for col in pos_cols:
        aligned_df_all[col] = monomer_db_filt_smiles_df["monomer"]
    aligned_df_all[args.idcol] = aligned_df_all.index

    coef_db_df = calculate_coef_df(
        aligned_df=aligned_df_all,
        pos_cols=pos_cols,
        monomer_desc_df=monomer_db_filt_desc_df,
        kept_columns_by_pos=kept_columns_by_pos,
        model=model,
        act_col=None,
        idcol=args.idcol,
        position_prefix=args.pos_prefix,
    )

    # Annotate with training-set statistics for monomers that appeared in training
    coef_db_df = coef_db_df[["mono", "pos", "coef"]].merge(
        coef_df[["pos", "mono", "num", "min_act", "max_act", "ids", "acts"]],
        on=["mono", "pos"],
        how="left",
    )

    # Attach SMILES
    monomer_db_filt_smiles_df = monomer_db_filt_smiles_df.rename(columns={"monomer": "mono"})
    coef_db_df = coef_db_df.merge(monomer_db_filt_smiles_df, on="mono", how="left")

    # Positions where all DB monomer scores are zero (only one monomer in training):
    # replace with the training-set rows for those positions so enumeration still works.
    all_zero_pos = coef_db_df.groupby("pos")["coef"].apply(lambda s: (s == 0).all())
    zero_positions = all_zero_pos[all_zero_pos].index.tolist()
    coef_db_df = coef_db_df[~coef_db_df["pos"].isin(zero_positions)]
    for pos in zero_positions:
        coef_db_df = pd.concat([coef_db_df, coef_df[coef_df["pos"] == pos]], ignore_index=True)

    coef_db_df = coef_db_df.sort_values(["pos", "coef"], ascending=[True, False])
    coef_db_df.to_csv(
        results_save_dir / f"{out_prefix}-monomer_db_coefficients.csv", index=False
    )

    if args.max_enum_out > 0:
        enumed_db_df = enum_one_parent_helm(
            ref_pept.helm,
            coef_db_df,
            args.enum_positions,
            white_list,
            black_list,
            args.max_enum_out,
            pos_offset=pos_offset,
        )
        print(
            f"{len(enumed_db_df)} peptides enumerated (monomer DB). "
            f"Pred range: [{enumed_db_df['Predictions'].min():.4f}, "
            f"{enumed_db_df['Predictions'].max():.4f}]"
        )
        n_elements = enumed_db_df["HELM"].values[0].count(".")
        for iseq in range(n_elements):
            enumed_db_df[f"PEPTIDE1_{iseq}"] = enumed_db_df["HELM"].apply(
                lambda s, idx=iseq: s.split("PEPTIDE1{")[1].split("}")[0].split(".")[idx]
            )
        enumed_db_df.to_csv(
            results_save_dir / f"{out_prefix}_best_{args.max_enum_out}-monomer_db.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # 11. Visualisations
    # ------------------------------------------------------------------
    model_str = f" (FP FW {args.model} model)"
    plot_top_monomer_scores(
        coef_df, pos_cols, args.monomer_top_n, args.pos_prefix,
        str(results_save_dir / f"{out_prefix}_top{args.monomer_top_n}_monomers.png"),
        model_str,
    )
    plot_top_scoring_monomer_frequencies(
        aligned_df, pos_cols, coef_df, args.monomer_top_n, args.pos_prefix,
        str(results_save_dir / f"{out_prefix}_top{args.monomer_top_n}_monomers_freq.png"),
        model_str,
    )
    plot_top_monomer_scores(
        coef_db_df, pos_cols, args.monomer_top_n, args.pos_prefix,
        str(results_save_dir / f"{out_prefix}_top{args.monomer_top_n}_monomer_db.png"),
        f" from output DB (FP FW {args.model} model)",
    )

    if args.fpbit_plot:
        for pos_col in pos_cols:
            pos = int(pos_col.replace(args.pos_prefix, ""))
            sub_dir = results_save_dir / "substructure_analysis" / pos_col
            sub_dir.mkdir(parents=True, exist_ok=True)
            plot_top_morgan_bits(
                coef_df,
                pos_cols,
                model,
                kept_columns_by_pos,
                monomer_desc_df,
                x,
                args.pos_prefix,
                position_idx=pos,
                top_n_monomers=args.monomer_top_n,
                top_n_bits=args.fpbit_top_n,
                n_cols=5,
                fig_save_dir=sub_dir,
                fp_radius=args.fp_radius,
                fp_nbits=args.fp_len,
                use_chirality=True,
                pos_idx_offset=pos_offset,
            )
