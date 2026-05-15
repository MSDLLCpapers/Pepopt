# note: for sklearn models. better rewrite another script for other models. 
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
from sklearn.neural_network import MLPRegressor
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


def fine_tune_hyperparameters_optuna(train_indices, test_indices, X_full, y_full, model_type, params_path, desc_type, n_trials=100, random_state=42, verbose=True):
    # Check if optimal parameters already exist
    if os.path.exists(params_path):
        if verbose:
            print(f"Loading existing optimal parameters from {params_path}")
        with open(params_path, 'r') as f:
            data = json.load(f)
        optimal_params = data['optimal_params']
        fine_tune_time = data.get('fine_tune_time', None)
        return optimal_params, fine_tune_time
    
    if verbose:
        print(f"Starting inner CV hyperparameter optimization for {model_type}...")
    
    start_time = time.time()
    # normalize the data for numerical features
    if desc_type in ['mono_rdk_desc', 'mono_mordred', 'pep_ecfp_counts', 'pep_mordred']:
        print(f'Normalize feature: {desc_type}')
        # Calculate variances using pandas method
        variances = X_full.var(axis=0)
        non_zero_var_mask = variances > 0.1
        # Filter columns with sufficient variance
        X_full_filtered = X_full.loc[:, non_zero_var_mask]
        # Apply scaling
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X_full_filtered)
        # Convert back to DataFrame with original index and filtered column names
        X_full = pd.DataFrame(
            X_full_scaled,
            index=X_full_filtered.index,
            columns=X_full_filtered.columns
        )

    # Define objective function
    def objective(trial):
        # Suggest hyperparameters based on model type
        if model_type == 'ridge':
            alpha = trial.suggest_float('alpha', 1e-4, 1e3, log=True)
            model_class = Ridge
            model_params = {'alpha': alpha}
            
        elif model_type == 'lasso':
            alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
            model_class = Lasso
            model_params = {'alpha': alpha, 'max_iter': 2000}
            
        elif model_type == 'elastic_net':
            alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            model_class = ElasticNet
            model_params = {'alpha': alpha, 'l1_ratio': l1_ratio, 'max_iter': 2000}
            
        elif model_type == 'pls':
            #
            n_components = trial.suggest_int('n_components', 1, 50)
            model_class = PLSRegression
            model_params = {'n_components': n_components} 
            
        elif model_type == 'pcr':
            # Use percentage of components instead of exact number
            perc_comp = trial.suggest_float('perc_comp', 0.01, 0.3)
            # Store percentage for later conversion in pipeline

        elif model_type == 'pca_pls':
            pca_n_components = trial.suggest_int('pca_n_components', 1, 50)
            pls_n_components = trial.suggest_int('pls_n_components', 1, 10)
            # Store parameters for pipeline creation
            model_params = {
                'pca_n_components': pca_n_components,
                'pls_n_components': pls_n_components
            }

        elif model_type == 'rf':
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
            model_class = RandomForestRegressor
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': random_state,
                'n_jobs': -1
            }
            
        elif model_type == 'xgboost':
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            model_class = xgb.XGBRegressor
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'random_state': random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            
        elif model_type == 'svr':
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            epsilon = trial.suggest_float('epsilon', 1e-4, 1.0, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model_class = SVR
            model_params = {'C': C, 'epsilon': epsilon, 'gamma': gamma}
        
        elif model_type == 'linear_svr':
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            epsilon = trial.suggest_float('epsilon', 1e-4, 1.0, log=True)
            model_class = LinearSVR
            model_params = {'C': C, 'epsilon': epsilon, 'max_iter': 5000}

        elif model_type == 'mlp':
            # as suggested, two layers, hidden dim = 300, relu
            hidden_layer_sizes = (300,
                                  trial.suggest_int('hidden_layer2', 100, 500))
            activation = 'relu'
            learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
            model_class = MLPRegressor
            es_patience = 10
            model_params = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'learning_rate_init': learning_rate_init,
                'max_iter': 50,
                'random_state': random_state,
                'early_stopping': True,
                'n_iter_no_change': es_patience,
                'validation_fraction': 0.2,
            }

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Evaluate using existing train_folds for inner CV
        cv_scores = []
        for fold in range(len(train_indices)):
            # Use one fold as validation, others as training
            train_idx = train_indices[fold]
            test_idx = test_indices[fold]
            
            # Get inner training and validation data
            X_train = X_full.loc[train_idx].values
            y_train = y_full.loc[train_idx].values.flatten()
            X_test = X_full.loc[test_idx].values
            y_test = y_full.loc[test_idx].values.flatten()
            if model_type in ['pls', 'pcr', 'pca_pls']:
                # exclude the columns in X_train and X_test that have very small variance in X_train_val
                variances = np.var(X_train, axis=0)
                if desc_type in ['mono_rdk_desc', 'mono_mordred', 'pep_mordred', 'pep_ecfp_counts']:
                    # Find columns with non-zero variance
                    non_zero_var_mask = variances > 0.1
                else:
                    non_zero_var_mask = variances > 0
                # Filter both datasets
                X_train = X_train[:, non_zero_var_mask]  # For numpy arrays
                X_test = X_test[:, non_zero_var_mask]

            # Create and train model with component calculation    
            # Create and train model with component calculation    
            if model_type == 'pcr':
                # Calculate actual number of components based on training data size
                num_samples = X_train.shape[0]
                n_components = max(1, min(round(perc_comp * num_samples), 
                                        min(X_train.shape[0], X_train.shape[1])))
                model = Pipeline([
                    ('pca', PCA(n_components=n_components)),
                    ('regression', LinearRegression())
                ])
            elif model_type == 'pca_pls':
                # Create PCA-PLS pipeline
                pca_n_comp = min(model_params['pca_n_components'], 
                                min(X_train.shape[0], X_train.shape[1]))
                pls_n_comp = min(model_params['pls_n_components'], 
                                min(X_train.shape[0], pca_n_comp))  # PLS components <= PCA components
                
                model = Pipeline([
                    ('pca', PCA(n_components=pca_n_comp)),
                    ('pls', PLSRegression(n_components=pls_n_comp))
                ])
            else:
                model = model_class(**model_params)

            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate MSE for this inner fold
            mse = mean_squared_error(y_test, y_pred)
            cv_scores.append(mse)
        
        # Return mean CV score
        return np.mean(cv_scores)
            
    
    # Create study and optimize
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials)
    
    fine_tune_time = time.time() - start_time
    best_params = study.best_params
    best_score = study.best_value
    
    if verbose:
        print(f"Best parameters: {best_params}")
        print(f"Best inner CV MSE: {best_score:.4f}")
        print(f"Fine-tuning time: {fine_tune_time:.2f} seconds")
    
    # Save results
    save_data = {
        'optimal_params': best_params,
        'fine_tune_time': fine_tune_time,
        'best_score': best_score,
        'n_trials': len(study.trials),
        'model_type': model_type,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(params_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    return best_params, fine_tune_time

# First, collect all pairs
pairs = [
    (target_name, series_sel)
    for target_name in ['Target_A', 'Target_B', 'Target_C', 'Target_D', 'Target_E']
    for series_sel in DATASET_SETTINGS_UPDATE[target_name]['series_sel']
]

# Randomly select 3 pairs
random.seed(10)
selected_pairs = random.sample(pairs, 3)
n_split = 5
act_col = 'potency'
idcol = 'CompoundID'

linear_models = ['ridge', 'lasso', 'elastic_net', 'pls', 'pcr', 'linear_svr']
non_linear_mdls = ['rf', 'xgboost', 'svr', 'mlp']

all_models = linear_models + non_linear_mdls

# #TODO: DEBUG: 
# all_moldes = ['mlp']

for target_name, series_sel in selected_pairs:
    print(f"Start {target_name}: {series_sel}")
    # load the aligned results
    align_file_path = os.path.join(output_dir_align, f"{target_name}_{series_sel}_align.csv")
    aligned_df = pd.read_csv(align_file_path)
    # load monomer-based descriptors
    desc_file_path = os.path.join(output_dir_desc, f"{target_name}_{series_sel}_X_all.pkl")
    with open(desc_file_path, "rb") as f:
        X_all = pickle.load(f)
    # get the target
    y = aligned_df.loc[:, [idcol, act_col]]
    y.set_index(idcol, inplace=True)

    # fine-tune for the linear models
    for model_type in all_models:
        if model_type in linear_models:
            desc_types = ['mono_ecfp', 'mono_rdk_fp', 'mono_rdk_desc', 'mono_mordred', 'mono_ohe',
                           'pepland_orig_ckpt', 'pepland_0130'] # 'mono_ecfp', 'mono_rdk_fp','mono_rdk_desc', 'mono_mordred', 'mono_ohe'
            n_trials = 50

        else:
            desc_types = ['pep_mordred', 'mono_ecfp', 'mono_mordred', 'pep_ecfp', 'pep_ecfp_counts',
                           'pepland_orig_ckpt', 'pepland_0130']
            n_trials = 100

        # #TODO: DEBUG:
        # desc_types = ['pepland_orig_ckpt', 'pepland_0130']
        # n_trails = 20


        for desc_type in desc_types:
            try:
                X = X_all[desc_type]
            except KeyError:
                pkl_path = os.path.join(output_dir_desc, f"all_pepts_for_{desc_type}.pkl")
                with open(pkl_path, "rb") as f:
                    X = pickle.load(f)["pepland_frz"]  # fixed dataframe key
            
            dataset_idx = X.index.to_list()
            for split_type in ['cv', 'time']:
                train_indices = []
                test_indices = []
                # Loop over 5 folds
                for fold in range(n_split):
                    train_path = f"{output_dir_split}/{target_name}_{series_sel}_{split_type}_fold{fold}_train.csv"
                    test_path = f"{output_dir_split}/{target_name}_{series_sel}_{split_type}_fold{fold}_test.csv"
                    train_df = pd.read_csv(train_path)
                    test_df = pd.read_csv(test_path)
                    train_idx = train_df[idcol].values
                    test_idx = test_df[idcol].values

                    # make sure that train and test idx are in dataset_idx
                    train_idx = [idx for idx in train_idx if idx in dataset_idx]
                    test_idx = [idx for idx in test_idx if idx in dataset_idx]
                    
                    train_indices.append(train_idx)
                    test_indices.append(test_idx)


                # Hyperparameter tuning using inner CV on training folds
                params_path = os.path.join(output_dir_opt_para, 
                                        f"{target_name}_{series_sel}_{desc_type}_{model_type}_{split_type}_optimal_params.json")
                        
                optimal_params, fine_tune_time = fine_tune_hyperparameters_optuna(
                    train_indices, test_indices, X, y, model_type, params_path, desc_type, n_trials=n_trials
                )

