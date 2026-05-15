# PepOpt — Peptide Lead Optimization

> **PepOpt** is an interpretable machine learning framework for peptide lead optimization in ultralarge design spaces.

**Authors:** Yixiang Mao (MSD) · Ruibo Zhang (MSD)  
**Contact:** [ruibo.zhang@msd.com](mailto:ruibo.zhang@msd.com)

---

## Table of Contents

- [PepOpt — Peptide Lead Optimization](#pepopt--peptide-lead-optimization)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Input Data](#input-data)
    - [1. Peptide Dataset](#1-peptide-dataset)
    - [2. Monomer Lookup Tables](#2-monomer-lookup-tables)
  - [Quick Start](#quick-start)
  - [Full Argument Reference](#full-argument-reference)
  - [Output Files](#output-files)

---

## Overview

PepOpt implements an **FP-based Free Wilson Analysis (FP-FWA)** pipeline that:

- Learns interpretable per-monomer contributions to a target activity using ridge regression on molecular fingerprints
- Visualizes the most important monomers and their structural substructures
- Enumerates and ranks novel peptide candidates from an ultralarge design space

---

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Input Data

Two types of input files are required to run PepOpt.

### 1. Peptide Dataset

The peptide input must be **pre-aligned**, with each position represented as a separate column following the naming convention:

```
PEPTIDE1_0, PEPTIDE1_1, PEPTIDE1_2, ...
```

Each cell should contain the **HELM monomer name** for that position.

The CSV must also include:

| Column | Description | Example |
|--------|-------------|---------|
| **ID column** | Unique compound identifier in the format `letters + ("-" or "_") + numbers` | `PEP_00123`, `ABC-45` |
| **Target column** | The numerical property to be predicted (e.g., pIC50) | `pIC50` |
| **HELM column**   |  HELM of peptides. HELMs are required for peptide enumeration even though all monomers are provided in prealigned data table  |   | 

> 📄 **Example file:** `test_data/example_data_aligned.csv`

---

### 2. Monomer Lookup Tables

Monomer lookup tables map HELM monomer names to their structural SMILES fragments. You must provide **at least two tables**:

- One for monomers present in the **input dataset**
- One for monomers available for **enumeration / prediction**

Table paths are configured in `src/config.py`.

Each table must contain the following columns:

| Column | Description |
|--------|-------------|
| `symbol` | HELM monomer name |
| `Partial_SMILES` | Clipped SMILES fragment (reaction handles removed) |

> 📄 **Example file:** `test_data/example_monomer_db.csv`

---

## Quick Start

Run the FP-FWA pipeline from the `Pepopt_Public/` directory:

```bash
python local_run_fpfw.py test_data/example_data_aligned.csv pIC50 ID M-0001 \
    --model ridge \
    --desc_type ecfp --desc_norm False \
    --fp_len 2048 --fp_radius 2 --fp_sel_n 500 --fp_min_freq 0 \
    --fpbit_plot True --fpbit_top_n 5 \
    --evaluation True \
    --max_enum_out 1000 \
    --enum_positions 0 1 2 3 4 5
```

The four **positional arguments** are:

| Position | Argument | Description |
|----------|----------|-------------|
| 1 | `incsv` | Path to the aligned input CSV file |
| 2 | `actcol` | Name of the activity/target column |
| 3 | `idcol` | Name of the compound ID column |
| 4 | `refid` | ID of the reference peptide |

---

## Full Argument Reference

For a complete list of options, run:

```bash
python local_run_fpfw.py -h
```

---

## Output Files

All outputs are saved to `./results/` (or the directory specified by `--results_dir`):

```
results/
├── fpfwa-coefficients.csv               # Per-monomer coefficients from the model
├── fpfwa-monomer_db_coefficients.csv    # Extrapolated coefficients for monomers in the DB
├── fpfwa_top10_monomers.png             # Heatmap of top monomer coefficients (training set)
├── fpfwa_top10_monomers_freq.png        # Same heatmap, weighted by monomer frequency
├── fpfwa_top10_monomer_db.png           # Heatmap of extrapolated DB monomer coefficients
├── fpfwa_best_1000.csv                  # Top enumerated peptides (training set monomers)
├── fpfwa_best_1000-monomer_db.csv       # Top enumerated peptides (monomer DB)
└── substructure_analysis/              # Per-position Morgan bit substructure figures
```
