#   Calculate the time for Genetic Alg. search, and test if the same best peptide could be found by GA
# this is the 12h version

import time
import random
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, Ridge
from src.PyHELM import HelmObj


MONO_DESC_TAB_PATH = "test_data/pred_time/4000_monomers_desc.csv"; TAG='4k'  # for lib of 4000
MONO_DESC_TAB_PATH = "test_data/pred_time/380used_monomers_desc.csv"; TAG='380'  # for appeared monomers

TRAINING_X_PATH = "test_data/pred_time/training_x_fp.csv"
TRAINING_Y_PATH = "test_data/pred_time/training_y_tgt.csv"
KEPT_COLS_PATH = "test_data/pred_time/kept_columns_by_pos.json"
TRAINING_DATA_PATH = "test_data/pred_time/training_data.csv"

REF_ID = "B-100009096-001E"
REF_ID = "B-100004028-001C"

SET_RND_SEED = False


def get_fps_from_seq(seq, monos_desc_table, kept_cols=None):
    # seq is a list of monomer names. 
    # monos_desc_table is a dataframe with monomer descriptors, the col "monomer" is for monomer name, and "FP_xx" is for fingerprints
    # kept cols is the dict of kept cols after feature selection. Key is 'posXX' and items are like "posXX_FP_xx"

    # find the concat FPs for each given seq, return as np.array  (1xN)
    fps = []
    for i, monomer in enumerate(seq):
        mono_fp = monos_desc_table[monos_desc_table['monomer'] == monomer]
        if kept_cols is not None:
            pos_key = f'pos{i}'
            pos_kept_cols = [x.replace(f"pos{i}_", "") for x in kept_cols[pos_key]]  # strip each prefix
            pos_kept_fps = [
                mono_fp[col].values[0] for col in pos_kept_cols if col.startswith('FP_')
            ]
            fps.append(pos_kept_fps)
        else:
            fp_cols = [col for col in monos_desc_table.columns if col.startswith('FP_')]
            fps.append(mono_fp[fp_cols].values[0])
    fps_array = np.concatenate(fps).reshape(1, -1)
    return fps_array

def get_pred_score(fp_array, mdl):
    # fp_array is (1xN)
    pred = mdl.predict(fp_array)  # (1,)
    return pred[0]

def get_mutations(seq, monos_names: list, n_pos, n_muts, rnd_state=42):
    # seq is a list of monomer names
    # monos_names is a list of all possible monomer names
    # n_pos is the number of positions to be mutated
    # n_muts is the number of mutations to generate
    if SET_RND_SEED:
        random.seed(rnd_state)
    mutations = []
    for _ in range(n_muts):
        mut_seq = seq.copy()
        pos_indices = random.sample(range(len(seq)), n_pos)
        for pos in pos_indices:
            new_mono = random.choice(monos_names)
            mut_seq[pos] = new_mono
        mutations.append(mut_seq)
    return mutations


def ga_search(init_seq, mdl, monos_desc_table, kept_cols,
              n_generations=50, n_mutations=40, n_pos_mutated=2, n_kept_each_round=5):
    # init_seq is a list of monomer names
    # mdl is the trained model for prediction
    # monos_desc_table is the dataframe with monomer descriptors
    current_population = [init_seq]
    best_seq = init_seq
    mono_names = monos_desc_table['monomer'].tolist()
    best_score = get_pred_score(
        get_fps_from_seq(best_seq, monos_desc_table, kept_cols), mdl)
    print(f"Initial best score: {best_score}")

    history_best_scores = [best_score]
    history_best_seqs = [best_seq]
    for gen in range(n_generations):
        all_mutations = []
        for seq in current_population:
            muts = get_mutations(seq, mono_names,
                                 n_pos=n_pos_mutated, n_muts=n_mutations)
            all_mutations.extend(muts)

        scored_mutations = []
        for mut_seq in all_mutations:
            mut_fp = get_fps_from_seq(mut_seq, monos_desc_table, kept_cols)
            mut_score = get_pred_score(mut_fp, mdl)
            scored_mutations.append((mut_seq, mut_score))

        scored_mutations.sort(key=lambda x: x[1], reverse=True)
        current_population = [seq for seq, score in scored_mutations[:n_kept_each_round]]

        if scored_mutations[0][1] > best_score:
            best_seq = scored_mutations[0][0]
            best_score = scored_mutations[0][1]
        
        history_best_scores.append(best_score)
        history_best_seqs.append(best_seq)

        print(f"Generation {gen+1}: Best score: {best_score}")
    return best_seq, best_score, history_best_seqs, history_best_scores



if __name__ == "__main__":
    # Load data
    monos_desc_table = pd.read_csv(MONO_DESC_TAB_PATH)
    training_x = pd.read_csv(TRAINING_X_PATH, index_col=0)
    training_y = pd.read_csv(TRAINING_Y_PATH, index_col=0)
    with open(KEPT_COLS_PATH, 'r') as f:
        kept_cols = json.load(f)

    # Note: flattened kept_cols should match with training_x's cols
    _flat = []
    for pos_key, cols in kept_cols.items():
        _flat.extend(cols)
    assert set(_flat) == set(training_x.columns.tolist()), "Kept cols do not match training_x columns."

    training_data = pd.read_csv(TRAINING_DATA_PATH)
    # Prepare model
    start_time = time.time()
    mdl = BayesianRidge()
    mdl.fit(training_x.values, training_y.reindex(training_x.index).values.ravel())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model trained in {elapsed_time:.2f} sec")

    # Get reference sequence
    ref_row = training_data[training_data['ID_y'] == REF_ID]
    ref_pep = HelmObj(ref_row.iloc[0]["HELM_x"])
    ref_seq_raw = ref_pep.get_monomers_from_polymer("PEPTIDE1")
    # must take off the '[]' of each monomer
    ref_seq = []
    for mono in ref_seq_raw:
        if mono.startswith('[') and mono.endswith(']'):
            ref_seq.append(mono[1:-1])
        else:
            ref_seq.append(mono)
    print("Reference sequence:", '-'.join(ref_seq))

    # GA search
    start_time = time.time()
    best_seq, best_score, history_seqs, history_scores = ga_search(
        init_seq=ref_seq,
        mdl=mdl,
        monos_desc_table=monos_desc_table,
        kept_cols=kept_cols,
        n_generations=50,
        n_mutations=100,
        n_pos_mutated=2,
        n_kept_each_round=20
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    array_id = os.getenv("PBS_ARRAY_INDEX")
    message = array_id + ',' + ','.join([
        str(x) for x in history_scores
    ])
    with open(f"results/pred_time/traject_{TAG}/trj{array_id}.csv", "a") as f:
        f.write(message + "\n")


    print(f"GA Search completed in {elapsed_time:.2f} seconds.")
    print(f"Best sequence found: {'-'.join(best_seq)} with score: {best_score}")
    print("===" * 20)
