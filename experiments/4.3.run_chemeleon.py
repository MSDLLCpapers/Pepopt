#!/usr/bin/env python3
import os
import pandas as pd
import torch
import time
from chemprop import featurizers, nn, data, models

from pathlib import Path
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from paper_dataset_config import DATASET_SETTINGS_UPDATE

def main():
    overwrite = True
    # Base directories
    curr_dir_path = "~/free_wilson_peptides"
    data_set_path = '~/free_wilson_peptides/test_data/'
    output_dir = f'{curr_dir_path}/results/paper/'
    output_dir_split = f'{output_dir}/split/'
    # 
    act_col = 'potency'
    idcol = 'CompoundID'
    smile_col = 'COMPOUND_SMILES'
    # load chemleon model
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    agg = nn.MeanAggregation()
    chemeleon_mp = torch.load(f"{curr_dir_path}/chemeleon_mp.pt", weights_only=True)
    
    # Set parameters
    epochs = 30
    num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
    num_splits = 5
    
    # Set the split type and target data
    for target_name in ['Target_A', 'Target_B', 'Target_C', 'Target_D', 'Target_E']:
        settings = DATASET_SETTINGS_UPDATE[target_name]
        series_sel_all = settings['series_sel']
        for series_sel in series_sel_all:
            print(f"Start evaluating {target_name}: {series_sel}")
            for split_type in ['cv', 'time', 'walkforward']: # 'cv', 'time'
                if split_type == 'walkforward':
                    n_fold = num_splits-1
                else:
                    n_fold = num_splits
                for db_fold_num in range(n_fold):
                    fold_result_dir_path = f"{output_dir}/chemeleon/{target_name}_{series_sel}/{split_type}_fold{db_fold_num}/"
                    if os.path.exists(f"{fold_result_dir_path}/predictions.csv") and overwrite is False:
                        # Skip if the predictions exits
                        print(f"Skip {target_name}_{series_sel}_{split_type}_fold{db_fold_num} as the predictions exist")
                        continue
                    os.makedirs(fold_result_dir_path, exist_ok=True)
                    
                    # load and prepare data
                    train_path = f"{output_dir_split}/{target_name}_{series_sel}_{split_type}_fold{db_fold_num}_train.csv"
                    test_path = f"{output_dir_split}/{target_name}_{series_sel}_{split_type}_fold{db_fold_num}_test.csv"
                    train_df = pd.read_csv(train_path)
                    test_df = pd.read_csv(test_path)

                    # Create MoleculeDatapoint objects
                    train_smis = train_df.loc[:, smile_col].values
                    train_ys = train_df.loc[:, act_col].values.reshape(-1, 1)
                    train_data = []
                    for smi, y in zip(train_smis, train_ys):
                        try:
                            dp = data.MoleculeDatapoint.from_smi(smi, y)
                            train_data.append(dp)
                        except Exception as e:
                            print(f"Skipping invalid SMILES: {smi} ({e})")

                    test_smis = test_df.loc[:, smile_col].values
                    test_ys = test_df.loc[:, act_col].values.reshape(-1, 1)
                    test_data = []
                    for smi, y in zip(test_smis, test_ys):
                        try:
                            dp = data.MoleculeDatapoint.from_smi(smi, y)
                            test_data.append(dp)
                        except Exception as e:
                            print(f"Skipping invalid SMILES: {smi} ({e})")
                    # exclude the rows in test_df whose smile is not valid
                    valid_mask = []
                    for smi in test_df[smile_col]:
                        try:
                            _ = data.MoleculeDatapoint.from_smi(smi, 0)
                            valid_mask.append(True)
                        except Exception:
                            valid_mask.append(False)
                    test_df = test_df[valid_mask].reset_index(drop=True)

                    # Split training data into train and validation (80/20 split of the training data)
                    train_mols = [d.mol for d in train_data]  # RDkit Mol objects for structure based splits
                    train_indices, val_indices = data.make_split_indices(train_mols, "random", (0.8, 0.2, 0))[:2]  # Only need train/val split
                    final_train_data, val_data = data.split_data_by_indices(
                        train_data, train_indices, val_indices, []  # Empty list for test since we already have test_data
                    )[:2]

                    # Create datasets and normalize targets
                    train_dset = data.MoleculeDataset(final_train_data[0], featurizer)
                    scaler = train_dset.normalize_targets()  # Normalize training targets

                    val_dset = data.MoleculeDataset(val_data[0], featurizer)
                    val_dset.normalize_targets(scaler)  # Apply same normalization to validation

                    test_dset = data.MoleculeDataset(test_data, featurizer)
                    # Note: We don't normalize test targets as we want to predict in original scale

                    # Create data loaders
                    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
                    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
                    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

                    # Create model components with output transform for inverse scaling
                    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
                    mp.load_state_dict(chemeleon_mp['state_dict'])
                    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)  # This will inverse transform predictions
                    ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=mp.output_dim)
                    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
                    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

                    # 
                    checkpointing = ModelCheckpoint(
                        f"{fold_result_dir_path}/checkpoints",  # Directory where model checkpoints will be saved
                        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
                        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
                        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
                        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
                    )
                    trainer = pl.Trainer(
                        logger=False,
                        enable_checkpointing=True,
                        enable_progress_bar=True,
                        accelerator="auto",
                        devices=1,
                        max_epochs=epochs, # number of epochs to train for
                        callbacks=[checkpointing], # Use the configured checkpoint callback
                    )

                    # Start timing for training
                    train_start = time.time()
                    
                    # train the model (on normalized targets)
                    trainer.fit(mpnn, train_loader, val_loader)
                    
                    train_time = time.time() - train_start
                    
                    # Start timing for testing/prediction
                    test_start = time.time()
                    
                    # get the predictions (automatically inverse transformed to original scale)
                    predictions = trainer.predict(mpnn, test_loader)
                    all_predictions = torch.cat(predictions, dim=0)
                    predictions_array = all_predictions.numpy()
                    
                    test_time = time.time() - test_start
                    
                    # save the predictions with the targets (predictions are now in original scale)
                    test_df['predictions'] = predictions_array.flatten()
                    test_df.to_csv(f"{fold_result_dir_path}/predictions.csv", index=False)
                    
                    # # calculate the performance metrics
                    # nrmse, r2, spearman_corr, pearson_corr = calc_metrics(test_df[act_col].values.flatten(), test_df['predictions'].values.flatten())
                    # print(f'Model Performance for testing:')
                    # print(f'r2:{r2}; spearman:{spearman_corr}; pearson:{pearson_corr}')
                    # Save timing information
                    times_df = pd.DataFrame({
                        'train_time_seconds': [train_time],
                        'test_time_seconds': [test_time]
                    })
                    times_df.to_csv(f"{fold_result_dir_path}/times.csv", index=False)
                    
                    print(f"Completed {target_name}_{series_sel}_{split_type}_fold{db_fold_num}")
                    print(f"  Training: {train_time:.2f}s, Testing: {test_time:.2f}s")


if __name__ == "__main__":
    main()
