# Only example settings could be provided due to data regulations. 
# For illustrating the format only. 

DATASET_SETTINGS_UPDATE = {
    'Target_A': {
        'incsv': './test_data/Target_A/Target_A_processed_data.csv',
        'act_col': 'pEC50',
        'idcol': 'COMPOUND_ID',
        'split_series_col': 'CHEM_SERIES',
        'smile_col': 'COMPOUND_SMILES',
        'fixed_length': True,
        'aln_min_sim': 0.3,
        'series_sel': ['Series-1', 'Series-2', 'Series-3'],
        'refid': 'CPD-1'
    },
}


refid_all = {
    'Target_A': {

    },
    'Target_B': {

    },
    'Target_C': {

    },
    
    'Target_D': {
    },

    'Target_E':{
    }
}