# Local peptide apis
import os
import functools
import ast
import pickle
from logging import getLogger
from pathlib import Path
from typing import Callable
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, RDKFingerprint
from rdkit.Chem.AllChem import GetMorganGenerator
from mordred import Calculator, descriptors
from PIL import Image as PILImage
import re

import numpy as np
import pandas as pd
import math
from src.args import get_args
from src.peptide_fw import *
from benchmark_pepfw import *
from src.api import peptide_api
from src.sequence.alignment import align_helm

import matplotlib.pyplot as plt
import seaborn as sns

import json
import random
import time
import optuna
from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from paper_dataset_config import DATASET_SETTINGS_UPDATE, refid_all


output_dir = './results/paper/'
os.makedirs(output_dir, exist_ok=True)
output_dir_align = f'{output_dir}/align/'
os.makedirs(output_dir_align, exist_ok=True)
output_dir_split = f'{output_dir}/split/'
os.makedirs(output_dir_split, exist_ok=True)
output_dir_desc = f'{output_dir}/desc/'
os.makedirs(output_dir_desc, exist_ok=True)
output_dir_opt_para = f'{output_dir}/opt_para/'
os.makedirs(output_dir_opt_para, exist_ok=True)

def calc_monomer_desc(smiles, feat_type, fp_radius=2, fp_nBits=100, useChirality=True):
    """
    Calculate molecular descriptors or fingerprints for a given SMILES string.
    
    Parameters:
    -----------
    smiles : str
        SMILES string of the molecule.
    feat_type : str
        The type of features to calculate. Options:
        - 'rdk_desc': 200+ RDKit constitutional descriptors.
        - 'ecfp': ECFP fingerprint bit vector.
        - 'ecfp_counts': ECFP fingerprint count vector.
        - 'rdk_fp': RDKit path-based fingerprint bit vector.
        - 'mordred': All 2D Mordred descriptors (Note: inefficient for batch processing).
    fp_radius : int, optional (default=2)
        Radius for Morgan fingerprint (ECFP).
    fp_nBits : int, optional (default=100)
        Number of bits for all fingerprint types.
    useChirality : bool, optional (default=True)
        Whether to include chirality in fingerprints.
        
    Returns:
    --------
    dict or None
        Dictionary containing features, or None if processing fails.
    """
    smiles = substitute_rgroup(smiles)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        result = {}
        
        if feat_type == 'rdk_desc':
            for name, func in Descriptors._descList:
                try:
                    result[name] = func(mol)
                except Exception:
                    result[name] = np.nan # Use NaN for failed descriptors
        
        elif feat_type == 'ecfp':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_nBits, useChirality=useChirality)
            for i, bit in enumerate(fp):
                result[f"FP_{i}"] = bit
        
        elif feat_type == 'ecfp_counts':
            gen = GetMorganGenerator(radius=fp_radius, fpSize=fp_nBits, includeChirality=useChirality)
            arr = gen.GetCountFingerprintAsNumPy(mol)
            for i, count in enumerate(arr):
                result[f"FP_{i}"] = int(count)

        elif feat_type == 'rdk_fp':
            # RDKit path-based fingerprint
            fp = RDKFingerprint(mol, fpSize=fp_nBits)
            for i, bit in enumerate(fp):
                result[f"FP_{i}"] = bit

        else:
            raise ValueError(f"Unknown feat_type: '{feat_type}'")
            
        return result
        
    except Exception as e:
        # print(f"Error processing SMILES '{smiles}': {e}") # Uncomment for debugging
        return None

def integrate_monomer_desc(monomer_smiles_df, feat_type, fp_radius=2, fp_nBits=100, useChirality=True, norm=False):
    """
    Integrates molecular descriptors/fingerprints into a DataFrame.
    Includes a highly efficient batch mode for Mordred descriptors.
    """
    # --- Special, high-performance case for Mordred ---
    if feat_type == 'mordred':
        print("Using efficient batch mode for Mordred descriptors...")
        # 1. Create Mol objects, filtering out invalid SMILES
        mols = [Chem.MolFromSmiles(substitute_rgroup(smi)) for smi in monomer_smiles_df['smiles']]
        valid_mask = [mol is not None for mol in mols]
        valid_mols = [m for m, v in zip(mols, valid_mask) if v]
        valid_monomer_smiles_df = monomer_smiles_df[valid_mask].reset_index(drop=True)

        # 2. Calculate descriptors in a single batch operation
        calc = Calculator(descriptors, ignore_3D=True)
        descriptors_df = calc.pandas(valid_mols)
        
        # Convert descriptor values to numeric, coercing errors to NaN
        descriptors_df = descriptors_df.apply(pd.to_numeric, errors='coerce')
        
        # drop columns with nan values
        descriptors_df.dropna(axis=1, inplace=True)

        # Fill any remaining NaN values (from calculation errors) with 0 or column mean
        # Using 0 is simple and often effective.
        # descriptors_df.fillna(0, inplace=True)

    # --- General case for all other feature types ---
    else:
        # Apply the function to each molecule
        descriptors_list = monomer_smiles_df['smiles'].apply(
            lambda x: calc_monomer_desc(x, feat_type=feat_type, fp_radius=fp_radius, fp_nBits=fp_nBits, useChirality=useChirality)
        )
        # Filter out the None values from failed calculations
        valid_mask = descriptors_list.notnull()
        valid_desc_list = descriptors_list[valid_mask]
        valid_monomer_smiles_df = monomer_smiles_df[valid_mask].reset_index(drop=True)
        
        if valid_desc_list.empty:
            print("Warning: No valid molecules could be processed.")
            return pd.DataFrame() # Return empty DataFrame
            
        # Convert the list of dictionaries to a DataFrame
        descriptors_df = pd.DataFrame(valid_desc_list.tolist())
        descriptors_df.fillna(0, inplace=True) # Fill NaNs from rdk_desc
    
    # Combine with the original valid DataFrame
    monomer_desc_df = pd.concat([valid_monomer_smiles_df, descriptors_df], axis=1)
    return monomer_desc_df


# calculate descriptors
act_col = 'potency'
idcol = 'CompoundID'
smile_col = 'COMPOUND_SMILES'
helm_col = 'HELM'
fp_radius = 2
fp_len = 2048
decs_norm = False

# Initialize list to collect all monomer smiles dataframes
all_monomer_smiles_dfs = []

for target_name in ['Target_A', 'Target_B', 'Target_C', 'Target_D', 'Target_E']:
    settings = DATASET_SETTINGS_UPDATE[target_name]
    series_sel_all = settings['series_sel']
    for series_sel in series_sel_all:
        # load the aligned results
        align_file_path = os.path.join(output_dir_align, f"{target_name}_{series_sel}_align.csv")
        aligned_df = pd.read_csv(align_file_path)
        # parse HELM Obj
        aligned_df['HelmObj'] = aligned_df[helm_col].apply(parse_helm_string)
        # get position cols
        pos_cols = get_position_columns(aligned_df.columns.values, 'PEPTIDE1_')
        # get distinct monomers from all peptides
        distinct_monomers_list = get_distinct_monomers(aligned_df)
        # get monomer smiles from local db
        monomer_db_path = 'regmol_db_update.csv' # regmol_monomers_processed.csv, peptide_monomer_db.csv
        #TODO: was peptide_monomer_db.csv. Should be regmol_db_update.csv?
        monomer_smiles_df = get_monomer_smiles_local(distinct_monomers_list, monomer_db_path)
        # collect monomer smiles dataframes
        all_monomer_smiles_dfs.append(monomer_smiles_df)

# combine the monomer smiles across targets and series
monomer_smiles_all_df = pd.concat(all_monomer_smiles_dfs, ignore_index=True)

# drop the duplicates by the 'monomer' column
monomer_smiles_all_df = monomer_smiles_all_df.drop_duplicates(subset=['monomer'], keep='first')

feature_types = ['ecfp', 'rdk_fp','rdk_desc', 'mordred'] # 'ecfp', 'rdk_fp','rdk_desc', 
for desc_type in feature_types:
    print(f"Processing descriptor type: {desc_type}")
    
    monomer_desc_df = integrate_monomer_desc(
        monomer_smiles_all_df, 
        desc_type, 
        fp_radius=fp_radius, 
        fp_nBits=fp_len, 
        useChirality=True, 
        norm=decs_norm
    )
    
    """ # set symbol column monomer_desc_df to index and drop the smiles column in monomer_desc_df
    monomer_desc_df = monomer_desc_df.set_index('monomer')

    if 'smiles' in monomer_desc_df.columns:
        monomer_desc_df = monomer_desc_df.drop('smiles', axis=1)
    
    # Check for problematic values
    print("Checking for infinite and extreme values...")
    monomer_desc_df = monomer_desc_df.apply(pd.to_numeric, errors='coerce')
    inf_mask = np.isinf(monomer_desc_df.values)
    if inf_mask.any():
        print(f"Found {inf_mask.sum()} infinite values")
    
    # Replace infinite values with NaN
    monomer_desc_df = monomer_desc_df.replace([np.inf, -np.inf], np.nan)
    
    # Optional: Replace extremely large values with NaN (adjust threshold as needed)
    # This helps with values that are technically finite but too large
    large_threshold = 1e10
    monomer_desc_df = monomer_desc_df.where(np.abs(monomer_desc_df) < large_threshold, np.nan)
    
    # Drop rows with missing values (including the converted inf values)
    monomer_desc_df = monomer_desc_df.dropna(axis=1)
    print(f"Shape after cleaning: {monomer_desc_df.shape}") """

    # save the monomer descriptors
    desc_file_path = os.path.join(output_dir_desc, f"{desc_type}_all.pkl")
    monomer_desc_df.to_pickle(desc_file_path)
    