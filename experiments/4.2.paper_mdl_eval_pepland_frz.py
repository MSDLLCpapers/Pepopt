
import os
import numpy as np
import pandas as pd
import pickle
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
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb
from src.peptide_fw import *
from paper_dataset_config import DATASET_SETTINGS_UPDATE, refid_all
from sklearn.neural_network import MLPRegressor

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


# evaluate the models using the optimal parameters
def build_model_with_optimal_params(model_type, optimal_params, X_train=None, random_state=10):
    """
    Create a model instance with optimal parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of model
    optimal_params : dict or tuple
        Optimal parameters (handles both dict and tuple formats)
    X_train : array-like, optional
        Training data needed for PLS/PCR to calculate n_components from perc_comp
    random_state : int
        Random state for reproducible results
        
    Returns:
    --------
    model : sklearn model instance
    actual_params : dict
        Dictionary containing actual parameters used (including computed n_components for PLS/PCR)
    """
    
    # Handle case where optimal_params is a tuple (params, time)
    if isinstance(optimal_params, tuple):
        params = optimal_params[0]
    else:
        params = optimal_params
    
    actual_params = {}
    
    if model_type == 'ridge':
        model = Ridge(alpha=params['alpha'])
        actual_params = params.copy()
        
    elif model_type == 'lasso':
        model = Lasso(alpha=params['alpha'])
        actual_params = params.copy()
    
    elif model_type == 'ridge_default':
        model = Ridge()
        actual_params = {'alpha': 1.0}  # Default value
        
    elif model_type == 'lasso_default':
        model = Lasso()
        actual_params = {'alpha': 1.0}  # Default value
    
    elif model_type == 'elastic_net':
        model = ElasticNet(
            alpha=params['alpha'],
            l1_ratio=params['l1_ratio']
        )
        actual_params = params.copy()
    
    elif model_type == 'elastic_net_default':
        model = ElasticNet()
        actual_params = {'alpha': 1.0, 'l1_ratio': 0.5}  # Default values
    
    elif model_type == 'pls':
        if 'perc_comp' in params:
            # New percentage-based approach
            if X_train is None:
                raise ValueError("X_train is required for PLS with perc_comp parameter")
            
            perc_comp = params['perc_comp']
            num_samples = X_train.shape[0]
            n_components = max(1, min(round(perc_comp * num_samples), 
                                    min(X_train.shape[0], X_train.shape[1])))
            
            model = PLSRegression(n_components=n_components)
            actual_params = {
                'perc_comp': perc_comp,
                'n_components': n_components,
                'num_samples': num_samples
            }
        else:
            # Legacy approach with direct n_components
            n_components = params.get('n_components', 2)
            model = PLSRegression(n_components=n_components)
            actual_params = {'n_components': n_components}
    
    elif model_type == 'pls_default':
        model = PLSRegression()
        actual_params = {'n_components': 2}  # Default value
    
    elif model_type == 'pcr':
        if 'perc_comp' in params:
            # New percentage-based approach
            if X_train is None:
                raise ValueError("X_train is required for PCR with perc_comp parameter")
            
            perc_comp = params['perc_comp']
            num_samples = X_train.shape[0]
            n_components = max(1, min(round(perc_comp * num_samples), 
                                    min(X_train.shape[0], X_train.shape[1])))
            
            model = Pipeline([
                ('pca', PCA(n_components=n_components)),
                ('regressor', LinearRegression())
            ])
            actual_params = {
                'perc_comp': perc_comp,
                'pca__n_components': n_components,
                'num_samples': num_samples
            }
        else:
            # Legacy approach with direct pca__n_components
            n_components = params.get('pca__n_components', 2)
            model = Pipeline([
                ('pca', PCA(n_components=n_components)),
                ('regressor', LinearRegression())
            ])
            actual_params = {'pca__n_components': n_components}
        
    elif model_type == 'bayesian_ridge':
        model = BayesianRidge()
        actual_params = {}  # Uses default parameters
    
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=random_state,
            n_jobs=-1
        )
        actual_params = params.copy()
        actual_params['random_state'] = random_state
    
    elif model_type == 'rf_default':
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        actual_params = {'random_state': random_state}
    
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        actual_params = params.copy()
        actual_params['random_state'] = random_state
    
    elif model_type == 'xgboost_default':
        model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0)
        actual_params = {'random_state': random_state}
    
    elif model_type == 'svr':
        model = SVR(
            C=params['C'],
            epsilon=params['epsilon'],
            gamma=params['gamma']
        )
        actual_params = params.copy()

    elif model_type == 'svr':
        model = LinearSVR(
            C=params['C'],
            epsilon=params['epsilon'],
            max_iter = 5000
        )
        actual_params = params.copy()

    elif model_type == 'svr_default':
        model = SVR()
        actual_params = {}  # Uses default parameters
    
    elif model_type == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=(300, params['hidden_layer2']),
            learning_rate_init=params['learning_rate_init'],
            max_iter=50,
            random_state=random_state
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, actual_params

def calc_metrics(y_test, y_pred):
    # Calculate performance metrics
    # NRMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse = rmse / (np.std(y_test)+1e-6)
    
    # R2
    r2 = r2_score(y_test, y_pred)
    
    # Spearman Correlation
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)
    
    # Pearson Correlation
    pearson_corr, pearson_p = pearsonr(y_test, y_pred)

    return nrmse, r2, spearman_corr, pearson_corr


# Updated main evaluation loop
n_split = 5
act_col = 'potency'
idcol = 'CompoundID'

all_models = ['bayesian_ridge', 'ridge', 'mlp']



# evaluate different models and descriptors
metrics_columns = ['series', 'num_peptides', 'split_type', 'fold', 'descriptor', 'model', 
                    'nrmse', 'r2', 'spearman_corr', 'pearson_corr', 'fine_tune_time', 
                    'train_time', 'test_time', 'actual_params']
performance_metrics = pd.DataFrame(columns=metrics_columns)

for target_name in ['Target_B', 'Target_A', 'Target_C', 'Target_D', 'Target_E']:
# for target_name in ['Target_A',]:
    settings = DATASET_SETTINGS_UPDATE[target_name]
    series_sel_all = settings['series_sel']
    for series_sel in series_sel_all:
        # load the aligned results
        align_file_path = os.path.join(output_dir_align, f"{target_name}_{series_sel}_align.csv")
        aligned_df = pd.read_csv(align_file_path)

        # get the target
        y = aligned_df.loc[:, [idcol, act_col]]
        y.set_index(idcol, inplace=True)

        for model_type in all_models:
            desc_types = ["pepland_orig_ckpt", "pepland_0130",]
            desc_types = ["pepland_0130",]

            for desc_type in desc_types:
                pkl_path = os.path.join(output_dir_desc, f"all_pepts_for_{desc_type}.pkl")
                with open(pkl_path, "rb") as f:
                    X = pickle.load(f)["pepland_frz"]  # fixed dataframe key

                dataset_idx = X.index.to_list()
                print(f"Start evaluating {target_name}, {series_sel}, {model_type}, and {desc_type}")

                # scale the data for numerical features
                if desc_type in ['mono_rdk_desc', 'mono_mordred', 'pep_mordred', 'pep_ecfp_counts',]:
                    print('Normalize the data')
                    # exclude the columns in X_train and X_test that have very small variance in X_train_val
                    variances = X.var(axis=0)
                    non_zero_var_mask = variances > 0.1
                    # Filter both datasets
                    X = X.loc[:, non_zero_var_mask]
                    # Create scaler
                    scaler = StandardScaler()
                    # Normalize and create new DataFrame
                    X_norm = pd.DataFrame(
                        scaler.fit_transform(X), 
                        columns=X.columns, 
                        index=X.index
                    )

                    # After standardization, values beyond ±N standard deviations are outliers
                    X_norm = X_norm.clip(lower=-3, upper=3)
                elif desc_type in ['pepland_orig_ckpt', 'pepland_0130',]:
                    scaler = StandardScaler()
                    X_norm = pd.DataFrame(
                        scaler.fit_transform(X), 
                        columns=X.columns, 
                        index=X.index
                    )
                    # also scale the Y
                    y_scaled = scaler.fit_transform(y).flatten()
                    y = pd.DataFrame(y_scaled, columns=[act_col], index=y.index)

                else: # do not normalize the data
                    X_norm = X

                # load the optimal parameters  # TODO: how was the avg got? 
                if model_type != 'bayesian_ridge':
                    avg_params_path = os.path.join(output_dir_opt_para, 
                                            f"{desc_type}_{model_type}_optimal_params_avg.json")
                    print(avg_params_path)
                    with open(avg_params_path, 'r') as f:
                        data = json.load(f)
                    optimal_params = data['optimal_params']
                    fine_tune_time = data.get('fine_tune_time', None)
                    print("Loaded optimal parameters:")
                    print(json.dumps(optimal_params, indent=2))
                else:
                    optimal_params = None
                    fine_tune_time = 0

                for split_type in ['cv', 'time', 'walkforward']:
                    # Loop over folds
                    if split_type == 'walkforward':
                        n_fold = n_split-1
                    else:
                        n_fold = n_split
                        
                    for fold in range(n_fold):
                        train_path = f"{output_dir_split}/{target_name}_{series_sel}_{split_type}_fold{fold}_train.csv"
                        test_path = f"{output_dir_split}/{target_name}_{series_sel}_{split_type}_fold{fold}_test.csv"
                        train_df = pd.read_csv(train_path)
                        test_df = pd.read_csv(test_path)
                        train_idx = train_df[idcol].values
                        test_idx = test_df[idcol].values

                        # make sure that train and test idx are in dataset_idx
                        train_idx = [idx for idx in train_idx if idx in dataset_idx]
                        test_idx = [idx for idx in test_idx if idx in dataset_idx]

                        # Extract data
                        X_train = X_norm.loc[train_idx].values
                        y_train = y.loc[train_idx].values.flatten()
                        X_test = X_norm.loc[test_idx].values
                        y_test = y.loc[test_idx].values.flatten()

                        if model_type in ['pls', 'pcr']:
                            # exclude the columns in X_train and X_test that have very small variance in X_train_val
                            variances = np.var(X_train, axis=0)
                            # Find columns with non-zero variance
                            if desc_type in ['mono_rdk_desc', 'mono_mordred']:
                                # Find columns with non-zero variance
                                non_zero_var_mask = variances > 0.5
                            else:
                                non_zero_var_mask = variances > 0
                            # Filter both datasets
                            X_train = X_train[:, non_zero_var_mask]  # For numpy arrays
                            X_test = X_test[:, non_zero_var_mask]

                        # get the model with the fine-tuned model parameters        
                        model, actual_params = build_model_with_optimal_params(
                            model_type, optimal_params, X_train=X_train
                        )

                        # Print component info for PLS/PCR
                        if model_type in ['pls', 'pcr'] and 'n_components' in actual_params:
                            print(f"Using {actual_params['n_components']} components for {model_type}")
                            if 'perc_comp' in actual_params:
                                print(f"  (perc_comp: {actual_params['perc_comp']:.3f}, "
                                      f"num_samples: {actual_params['num_samples']})")

                        # train
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_time = time.time() - start_time

                        # test
                        start_time = time.time()
                        y_pred = model.predict(X_test)
                        _y_pred_debug = y_pred.copy()

                        # # Force removal of outliers, for pepland results
                        # y_pred[y_pred > 15] = np.nan
                        # y_pred[y_pred < 0] = np.nan
                        # np.nan_to_num(y_pred, copy=False, nan=np.nanmean(y_pred))
                        if np.isnan(y_pred).all():
                            print(_y_pred_debug)

                        # #TODO DEBUG:
                        # print("Mean & std:", y_pred.mean(), y_pred.std())
                        # outlier_pred_i = np.where((y_pred > 15) | (y_pred < 0))[0]
                        # outlier_pred_idx = np.array(test_idx)[outlier_pred_i]
                        # if len(outlier_pred_idx) > 0:
                        #     print(f"Outlier predictions at indices: {outlier_pred_idx}, "
                        #           f"predictions: {y_pred[outlier_pred_i]}",
                        #           f"Split Type: {split_type}, Fold: {fold}, Model: {model_type}, Descriptor: {desc_type}")

                        testing_time = time.time() - start_time
                        
                        nrmse, r2, spearman_corr, pearson_corr = calc_metrics(y_test, y_pred.flatten())
                        print(f'Performance for model:{model_type}, {split_type}, {fold}')
                        print(f'r2:{r2}; spearman:{spearman_corr}; pearson:{pearson_corr}')

                        # Add metrics to performance DataFrame
                        new_row = {
                            'series': f'{target_name}_{series_sel}',
                            'num_peptides': len(train_idx)+len(test_idx),
                            'split_type': split_type,
                            'fold': fold,
                            'descriptor': desc_type, 
                            'model': model_type,
                            'nrmse': nrmse,
                            'r2': r2,
                            'spearman_corr': spearman_corr,
                            'pearson_corr': pearson_corr,
                            'fine_tune_time': fine_tune_time,
                            'train_time': training_time, 
                            'test_time': testing_time,
                            'actual_params': json.dumps(actual_params)  # Store as JSON string
                        }
                        performance_metrics.loc[len(performance_metrics)] = new_row
                        
    performance_metrics.to_csv(f"{output_dir}/performance_metrics_pepland_scaledY.csv", index=False)
    print(performance_metrics)

