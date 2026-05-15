
import os
import pandas as pd
from paper_dataset_config import DATASET_SETTINGS_UPDATE


if __name__ == "__main__":

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



    # get data splits
    act_col = 'potency'
    idcol = 'COMPOUND_ID'
    smile_col = 'COMPOUND_SMILES'
    helm_col = 'HELM'

    for target_name in ['Target_A', 'Target_B', 'Target_C', 'Target_D', 'Target_E']:
        settings = DATASET_SETTINGS_UPDATE[target_name]
        series_sel_all = settings['series_sel']
        for series_sel in series_sel_all:
            # load the aligned results
            align_file_path = os.path.join(output_dir_align, f"{target_name}_{series_sel}_align.csv")
            aligned_df = pd.read_csv(align_file_path)
            # sort by time (COMPOUND_ID)
            aligned_df['COMPOUND_ID-Int'] = aligned_df[idcol].str.extract(r'B-0*(\d+)').astype(float)
            aligned_df = aligned_df.sort_values('COMPOUND_ID-Int')
            # generate cv folds
            n_split = 5
            train_indices_time, test_indices_time, train_indices_time_order, test_indices_time_order, train_indices_cv, test_indices_cv = gen_train_test_ids(aligned_df, n_split, idcol)
            for split_type in ['cv', 'time']:
                if split_type == 'cv':
                    train_indices = train_indices_cv
                    test_indices = test_indices_cv
                elif split_type == 'time':
                    train_indices = train_indices_time
                    test_indices = test_indices_time
                elif split_type == 'walkforward':
                    train_indices = train_indices_time_order
                    test_indices = test_indices_time_order
                for fold in range(len(train_indices)):
                    train_idx = train_indices[fold]
                    test_idx = test_indices[fold]
                    fold_str = f'{split_type}_fold{fold}'
                    
                    train_data = aligned_df[aligned_df[idcol].isin(train_idx)]
                    train_file = os.path.join(output_dir_split, f'{target_name}_{series_sel}_{fold_str}_train.csv')
                    train_data.to_csv(train_file, index=False)
                    
                    test_data = aligned_df[aligned_df[idcol].isin(test_idx)]
                    test_file = os.path.join(output_dir_split, f'{target_name}_{series_sel}_{fold_str}_test.csv')
                    test_data.to_csv(test_file, index=False)
                    
                    print(f"Saved fold {fold}: {len(train_data)} train samples, {len(test_data)} test samples")

