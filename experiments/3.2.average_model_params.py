import random
import os
import json
import numpy as np

from paper_dataset_config import DATASET_SETTINGS_UPDATE, refid_all



output_dir = './results/paper/'
output_dir_opt_para = f'{output_dir}/opt_para/'


if __name__ == "__main__":
    # average the optimal parameters over selected series
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

    # #TODO:DEBUG
    # all_models = ['mlp']

    # average the optimal hyperparameters across series and time splits
    # Define which parameters should be integers for each model type
    INT_PARAMS = {
        'pls': ['n_components'],
        'pcr': ['pca__n_components'],
        'rf': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'xgboost': ['n_estimators', 'max_depth'],
        'ridge': [],
        'lasso': [],
        'elastic_net': [],
        'svr': [],
        'mlp': ['hidden_layer2'],
    }
    for model_type in all_models:
        if model_type in linear_models:
            desc_types = ['mono_ecfp', 'mono_rdk_fp', 'mono_rdk_desc', 'mono_mordred', 'mono_ohe',
                           'pepland_orig_ckpt', 'pepland_0130'] # 'mono_ecfp', 'mono_rdk_fp','mono_rdk_desc', 'mono_mordred', 'mono_ohe'
        else:
            desc_types = ['pep_mordred', 'mono_ecfp', 'mono_mordred', 'pep_ecfp', 'pep_ecfp_counts',
                           'pepland_orig_ckpt', 'pepland_0130']        
        for desc_type in desc_types:
            # Collect parameters from all selected pairs
            all_params = []
            all_fine_tune_times = []
            for split_type in ['cv', 'time']:
                for target_name, series_sel in selected_pairs:
                    
                    # Load optimal parameters for each pair
                    params_path = os.path.join(output_dir_opt_para,
                                            f"{target_name}_{series_sel}_{desc_type}_{model_type}_{split_type}_optimal_params.json")

                    try:
                        with open(params_path, 'r') as f:
                            data = json.load(f)
                        optimal_params = data['optimal_params']
                        fine_tune_time = data.get('fine_tune_time', None)

                        all_params.append(optimal_params)
                        if fine_tune_time is not None:
                            all_fine_tune_times.append(fine_tune_time)

                    except FileNotFoundError:
                        print(f"Warning: File not found - {params_path}")
                        continue
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in file - {params_path}")
                        continue

            # Calculate averaged parameters
            if all_params:
                averaged_params = {}
                
                # Get all unique parameter names
                all_param_names = set()
                for params in all_params:
                    all_param_names.update(params.keys())
                
                # Average each parameter
                for param_name in all_param_names:
                    param_values = []
                    for params in all_params:
                        if param_name in params:
                            param_values.append(params[param_name])
                    
                    if param_values:
                        # Handle different parameter types
                        if isinstance(param_values[0], (int, float)):
                            # Numeric parameters - calculate mean
                            avg_value = np.mean(param_values)
                            
                            # Convert to int if this parameter should be an integer
                            if param_name in INT_PARAMS.get(model_type, []):
                                averaged_params[param_name] = int(round(avg_value))
                            else:
                                averaged_params[param_name] = avg_value
                                
                        elif isinstance(param_values[0], bool):
                            # Boolean parameters - use majority vote
                            averaged_params[param_name] = sum(param_values) > len(param_values) / 2
                        elif isinstance(param_values[0], str):
                            # String parameters - use most common value
                            from collections import Counter
                            counter = Counter(param_values)
                            averaged_params[param_name] = counter.most_common(1)[0][0]
                        else:
                            # For other types, use the first value as default
                            averaged_params[param_name] = param_values[0]

                
                # Calculate average fine-tune time if available
                avg_fine_tune_time = np.mean(all_fine_tune_times) if all_fine_tune_times else None
                
                # Prepare data to save
                averaged_data = {
                    'optimal_params': averaged_params,
                    'fine_tune_time': avg_fine_tune_time,
                    'n_pairs_averaged': len(all_params),
                    'pairs_used': selected_pairs[:len(all_params)]  # Only pairs that were successfully loaded
                }
                
                # Save the averaged optimal parameters
                avg_params_path = os.path.join(output_dir_opt_para, 
                                            f"{desc_type}_{model_type}_optimal_params_avg.json")
                
                with open(avg_params_path, 'w') as f:
                    json.dump(averaged_data, f, indent=2, default=str)  # default=str handles numpy types
                
                print(f"\nAveraged over {len(all_params)} pairs")
                print(f"Saved averaged parameters for {desc_type}_{model_type} to {avg_params_path}")

                if "xgb" in model_type and "ecfp" in desc_type:
                    print(f"Averaged parameters for {desc_type}_{model_type}:")
                    for param_name, avg_value in averaged_params.items():
                        print(f"  {param_name}: {avg_value:.3f}" if isinstance(avg_value, (float)) \
                            else f"  {param_name}: {avg_value}")
            else:
                print(f"Warning: No valid parameters found for {desc_type}_{model_type}")
