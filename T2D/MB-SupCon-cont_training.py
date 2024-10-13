# %% [markdown]
# # Build and train MB-SupCon-cont

# %%
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
import re
import sys

from scipy.stats import spearmanr
#import skimage

import time
import torch
import torch.hub
import torch.nn

import random
import pickle

# import self-defined MB-SupCon module
from mbsupcon_cont import MbSupConContModel

# %%
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# %%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Training script for MB-SupCon with embedding and weighting options.")

# Add arguments for embedding dimension
parser.add_argument(
    '-e', '--embedding_dim', type=int, required=True,
    help="Set embedding dimension (e.g., 10, 20, 40, or any other positive integer)"
)

# Add arguments for weighting method
parser.add_argument(
    '-w', '--weighting_method', type=str, choices=['linear', 'exponential', 'negative-log'], required=True,
    help="Select weighting method: 'linear', 'exponential', or 'negative-log'"
)

# Parse the arguments
args = parser.parse_args()

# Set embedding dimension and weighting method based on user input
EMBEDDING_DIM = args.embedding_dim
WEIGHTING_METHOD = args.weighting_method

print("========================================================")
print('EMBEDDING_DIM:', EMBEDDING_DIM)
print('WEIGHTING_METHOD:', WEIGHTING_METHOD)
print("========================================================")

SAVE_FOLDER = f'./emb_dim-{EMBEDDING_DIM}/{WEIGHTING_METHOD}_weights'
os.makedirs(SAVE_FOLDER, exist_ok=True)

# %% [markdown]
# ## 1 Load data

# %%
gut_16s_df = pd.read_table("./data/gut_16s_abundance.txt", index_col=0)
metabolome_df = pd.read_table("./data/metabolome_abundance.txt", index_col=0)
metabolome_df = metabolome_df.drop(columns=['SubjectID', 'CollectionDate', 'CL1', 'CL2', 'CL3', 'CL4'])

# %%
subjects = pd.read_csv("./data/subjects.csv", na_values="NA")
copy_gut_16s_df = gut_16s_df.copy()
copy_gut_16s_df['SubjectID'] = [copy_gut_16s_df.index[i][0:7] for i in range(len(copy_gut_16s_df.index))]
gut_16s_df_subj = pd.merge(left=copy_gut_16s_df, right=subjects, how='left', on='SubjectID')
gut_16s_df_subj.index = copy_gut_16s_df.index

# %%
# Get indexes for shared patient ids
indexes = list(set(gut_16s_df.index.values) & set(metabolome_df.index.values))
indexes.sort()
print("# of samples: {}".format(len(indexes)))

# %% [markdown]
# ## 2 Train MB-SupCon-cont with different continuous covariates and store embeddings

# %%
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_EPOCH = 1000

random_seed_list = range(1,13)
covariate_list = ['Age', 'BMI', 'SSPG']

## From tuning
max_idx = {'Age': (0, 3, 0), 'BMI': (0, 3, 4), 'SSPG': (1, 4, 4)}

# %%
start_time = time.time()

multi_omics_dict = {'gut_16s': gut_16s_df, 'metabolome': metabolome_df}
net_dict = {"gut_16s": [gut_16s_df.shape[1], 64, 32], 
            "metabolome": [metabolome_df.shape[1], 256, 64]}
predict_methods_list = ['elasticnet', 'svr', 'rf', 'xgboost']

dropout_list = [0.2, 0.4, 0.6, 0.8]
weight_decay_list = 0.01*2.**np.linspace(-2,5,8)
temp_list = 0.5*2.**np.linspace(-2,2,5)

microbiome_mse_embedding_dict, microbiome_mse_original_dict = dict(), dict()
metabolome_mse_embedding_dict, metabolome_mse_original_dict = dict(), dict()
import copy
for covariate in covariate_list:
    
    dropout = dropout_list[max_idx[covariate][0]]
    weight_decay = weight_decay_list[max_idx[covariate][1]]
    temperature = temp_list[max_idx[covariate][2]]

    g_valtest_mse_embedding, g_valtest_mse_original, m_valtest_mse_embedding, m_valtest_mse_original = \
        [dict(list(zip(predict_methods_list,
            np.zeros(shape=(len(predict_methods_list), len(random_seed_list), 2))))) 
            for i in range(4)]
    for s, seed in enumerate(random_seed_list):
        MB_SupCon_cont = MbSupConContModel(covariate=covariate, indexes=indexes, multi_omics_dict=multi_omics_dict,
                                 df_with_covariates=gut_16s_df_subj, device=DEVICE,
                                 root_folder=SAVE_FOLDER, random_seed=seed)
        MB_SupCon_cont.initialize(net_dict=net_dict, weight_method_name = WEIGHTING_METHOD, n_out_features=EMBEDDING_DIM,
                                  temperature=temperature, dropout_rate=dropout, weight_decay=weight_decay)
        MB_SupCon_cont.train_model(n_epoch=N_EPOCH, print_hist=False)
        MB_SupCon_cont.plot_training(save=True)
        MB_SupCon_cont.save_training()
        MB_SupCon_cont.save_embedding()

        ### Predict based on embeddings by using different predictors
        predict_embedding_by_method_dict, predict_original_by_method_dict = dict(), dict()
        mse_embedding_dict, mse_original_dict = [dict.fromkeys(MB_SupCon_cont.keys) for i in range(2)]
        for method in predict_methods_list:
            predict_embedding_by_method_dict[method] = MB_SupCon_cont.predict_embedding(predict_method=method)
            g_valtest_mse_embedding[method][s] = \
                [predict_embedding_by_method_dict[method]['gut_16s']['val'][0],
                predict_embedding_by_method_dict[method]['gut_16s']['test'][0]]
            m_valtest_mse_embedding[method][s] = \
                [predict_embedding_by_method_dict[method]['metabolome']['val'][0],
                predict_embedding_by_method_dict[method]['metabolome']['test'][0]]
            
            
            predict_original_by_method_dict[method] = MB_SupCon_cont.predict_original(predict_method=method)
            g_valtest_mse_original[method][s] = \
                [predict_original_by_method_dict[method]['gut_16s']['val'][0], 
                predict_original_by_method_dict[method]['gut_16s']['test'][0]]
            m_valtest_mse_original[method][s] = \
                [predict_original_by_method_dict[method]['metabolome']['val'][0], 
                predict_original_by_method_dict[method]['metabolome']['test'][0]]

        for gm in MB_SupCon_cont.keys:
            mse_embedding_dict[gm], mse_original_dict[gm] =\
                [pd.DataFrame(index=predict_methods_list, 
                              columns=['Validation', 'Testing']) for i in range(2)]
            for ii, method in enumerate(predict_methods_list):
                for jj, which_dataset in enumerate(['val', 'test']):
                    mse_embedding_dict[gm].iloc[ii, jj] = \
                        predict_embedding_by_method_dict[method][gm][which_dataset][0]
                    mse_original_dict[gm].iloc[ii, jj] = \
                        predict_original_by_method_dict[method][gm][which_dataset][0]

        print('Prediction based on microbiome embeddings of MB-SupCon:')
        # print(mse_embedding_dict['gut_16s'].style.format(formatter="{:.4f}"))
        print(mse_embedding_dict['gut_16s'])
        print('Prediction based on original microbiome data:')
        print(mse_original_dict['gut_16s'])
        print('------------------------------------------------------------------')
        print('Prediction based on metabolome embeddings of MB-SupCon:')
        print(mse_embedding_dict['metabolome'])
        print('Prediction based on original metabolome data:')
        print(mse_original_dict['metabolome'])
        print('------------------------------------------------------------------')
            
    microbiome_mse_embedding_dict[covariate] = g_valtest_mse_embedding
    microbiome_mse_original_dict[covariate] = g_valtest_mse_original
    
    metabolome_mse_embedding_dict[covariate] = m_valtest_mse_embedding 
    metabolome_mse_original_dict[covariate] = m_valtest_mse_original

import pickle
with open(os.path.join(SAVE_FOLDER, 'microbiome_mse_embedding_dict.pkl'), 'wb') as f:
    pickle.dump(microbiome_mse_embedding_dict, f)

with open(os.path.join(SAVE_FOLDER, 'microbiome_mse_original_dict.pkl'), 'wb') as f:
    pickle.dump(microbiome_mse_original_dict, f)
    
with open(os.path.join(SAVE_FOLDER, 'metabolome_mse_embedding_dict.pkl'), 'wb') as f:
    pickle.dump(metabolome_mse_embedding_dict, f)

with open(os.path.join(SAVE_FOLDER, 'metabolome_mse_original_dict.pkl'), 'wb') as f:
    pickle.dump(metabolome_mse_original_dict, f)
    
end_time = time.time()
print(end_time - start_time)

# %% [markdown]
# ---
# 
# ## MSE $\rightarrow$ RMSE

# %%
def mse_to_rmse_dict(save_folder, omics_mse_dict_filename, 
                covariate_list=['Age'], 
                predict_methods_list=['elasticnet', 'svr', 'rf', 'xgboost']):
    import pickle
    with open(os.path.join(SAVE_FOLDER, omics_mse_dict_filename), 'rb') as f:
        omics_mse_dict = pickle.load(f)
    
    omics_rmse_dict = {}
    for covariate in covariate_list:
        omics_rmse_dict[covariate] = {}
        for method in predict_methods_list:
            omics_rmse_dict[covariate][method] = np.sqrt(omics_mse_dict[covariate][method])
            
    omics_rmse_dict_filename = omics_mse_dict_filename.replace('mse', 'rmse')
    with open(os.path.join(SAVE_FOLDER, omics_rmse_dict_filename), 'wb') as f:
        pickle.dump(omics_rmse_dict, f)
    return omics_rmse_dict

# %%
microbiome_rmse_embedding_dict = mse_to_rmse_dict(SAVE_FOLDER, 'microbiome_mse_embedding_dict.pkl', covariate_list=covariate_list)
microbiome_rmse_original_dict = mse_to_rmse_dict(SAVE_FOLDER, 'microbiome_mse_original_dict.pkl', covariate_list=covariate_list)
metabolome_rmse_embedding_dict = mse_to_rmse_dict(SAVE_FOLDER, 'metabolome_mse_embedding_dict.pkl', covariate_list=covariate_list)
metabolome_rmse_original_dict = mse_to_rmse_dict(SAVE_FOLDER, 'metabolome_mse_original_dict.pkl', covariate_list=covariate_list)

# %% [markdown]
# ## 3 Scatterplots on lower-dimensional space for random seed 1

# %%
dim_reduction_list = ['pca']
random_seed_list = [1]

# %%
start_time = time.time()

for covariate in covariate_list:
    dropout = dropout_list[max_idx[covariate][0]]
    weight_decay = weight_decay_list[max_idx[covariate][1]]
    temperature = temp_list[max_idx[covariate][2]]

    for s, seed in enumerate(random_seed_list):
        ### load models, not necessary to train them again
        model_save_folder = os.path.join(SAVE_FOLDER, 'models/{}'.format(covariate))
        model_path = os.path.join(model_save_folder, 
                                  'MB-SupCon-cont_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pth'.\
                         format(covariate, N_EPOCH, temperature, dropout, weight_decay, seed))
        
        embedding_save_folder = os.path.join(SAVE_FOLDER, 'embeddings/{}'.format(covariate))
        embedding_path = os.path.join(embedding_save_folder, 
                                      'embeddings_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pkl').\
              format(covariate, N_EPOCH, temperature, dropout, weight_decay, seed)
        
        
        MB_SupCon_cont = MbSupConContModel(covariate=covariate, indexes=indexes, multi_omics_dict=multi_omics_dict,
                                 df_with_covariates=gut_16s_df_subj, device=DEVICE,
                                 root_folder=SAVE_FOLDER, random_seed=seed)
        MB_SupCon_cont.initialize(net_dict=net_dict, weight_method_name = WEIGHTING_METHOD, n_out_features=EMBEDDING_DIM,
                                  temperature=temperature, dropout_rate=dropout, weight_decay=weight_decay)
        MB_SupCon_cont.load_model(model_path, embedding_path)
        
        for dim_reduction in dim_reduction_list:
            MB_SupCon_cont.dim_reduction_embedding(dim_reduction_method = dim_reduction, fontsize=40)
            MB_SupCon_cont.dim_reduction_original(dim_reduction_method = dim_reduction, fontsize=40)
        
        ## Save index split 
        indexes_train, indexes_val, indexes_test = MB_SupCon_cont.indexes_split
            
        os.makedirs(f'{SAVE_FOLDER}/data/index/{covariate}', exist_ok=True)
        np.savetxt(f'{SAVE_FOLDER}/data/index/{covariate}/indexes_noukn.txt', MB_SupCon_cont.indexes_no_ukn, fmt='%s')
        np.savetxt(f'{SAVE_FOLDER}/data/index/{covariate}/indexes_train.txt', indexes_train, fmt='%s')
        np.savetxt(f'{SAVE_FOLDER}/data/index/{covariate}/indexes_val.txt', indexes_val, fmt='%s')
        np.savetxt(f'{SAVE_FOLDER}/data/index/{covariate}/indexes_test.txt', indexes_test, fmt='%s')
    
end_time = time.time()
print(end_time - start_time)

# %% [markdown]
# ### Environment

# %%
import platform
print(platform.system())
print(platform.release())

# %%
import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# %%
from platform import python_version

print(python_version())

# %%
os.system("conda list -p /work/PCDC/s198665/conda_envs/envir_MB-SupCon")