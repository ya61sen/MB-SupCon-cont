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
# ---

# %% [markdown]
# ## 1 Simulate data

# %% [markdown]
# ### Simulate embeddings

# %%
import numpy as np
from scipy.stats import spearmanr

def generate_cov_matrix(correlation, corr_sd):
    while True:
        temp_rn1 = np.random.normal(loc=correlation, scale=corr_sd)
        if abs(temp_rn1) <= 1:
            cov_mat = np.array([[1, temp_rn1], [temp_rn1, 1]])
            if np.all(np.linalg.eigvals(cov_mat) >= 0):
                return cov_mat

def generate_embeddings(N=1000, correlation=0.6, corr_sd=0.1, dim=10, random_state=123):
    np.random.seed(random_state)

    g_embed_mat = np.zeros((N, dim))
    m_embed_mat = np.zeros((N, dim))
    cor_col = np.zeros(dim)

    for i in range(dim):
        cov_mat = generate_cov_matrix(correlation, corr_sd)
        mu_g, mu_m = np.random.normal(size=2)
        X = np.random.multivariate_normal([mu_g, mu_m], cov_mat, size=N)
        g_embed_mat[:, i], m_embed_mat[:, i] = X.T   # X[:, 0], X[:, 1]
        cor_col[i] = spearmanr(X[:, 0], X[:, 1])[0]
    return pd.DataFrame(g_embed_mat, index=range(N)), pd.DataFrame(m_embed_mat, index=range(N)), cor_col

# %% [markdown]
# ### Generate original omics data

# %% [markdown]
# a. Generate loading matrices

# %% 
def generate_explained_variance_ratio(dim, scale=1, exaggeration_factor=3., random_state=123):
    np.random.seed(random_state)
    random_numbers = np.random.exponential(scale=scale, size=dim)
    transformed_numbers = random_numbers ** exaggeration_factor
    percent_variance = (transformed_numbers / np.sum(transformed_numbers)) * dim
    sorted_variance = np.sort(percent_variance)[::-1]
    return sorted_variance

def generate_loadings(dim_dict, random_state=123):
    np.random.seed(random_state)
    loadings_dict = {}
    for key in dim_dict.keys():
        dim = dim_dict[key]
        A = np.random.randn(dim, dim)
        # Perform QR Decomposition to get an orthogonal matrix Q
        Q, _ = np.linalg.qr(A)
        eigenvalues = generate_explained_variance_ratio(dim)
        print(eigenvalues)
        loadings_dict[key] = Q * np.sqrt(eigenvalues).reshape(-1,1) # (n_pc, n_data_col)
    return loadings_dict

# %% 
dim_dict = {'gut_16s': 100,
            'metabolome': 500}
loadings_dict = generate_loadings(dim_dict)

# %% [markdown]
# b. Generate original omics data

# %%
def add_nonlinearity(data_df, func):
    newdata = func(torch.tensor(data_df.values))
    return pd.DataFrame(newdata.numpy(), index=range(newdata.shape[0]))

# %%
def reconstruct_data_by_reverse_PCA(loadings_dict, embedding_dict, means_dict):
    embedding_dim = list(embedding_dict.values())[0].shape
    sim_original_dict = dict()
    for key in loadings_dict.keys():
        sim_data = np.matmul(embedding_dict[key].values, loadings_dict[key][:embedding_dim[1]]) + means_dict[key]
        sim_original_dict[key] = pd.DataFrame(sim_data, index=range(embedding_dim[0]))
    return sim_original_dict

# %% [markdown]
# ### Generate response

# %% [markdown]
# $$\beta \sim N(0,5^2)\\
# \tilde{y}=[Z_{\text{microbiome}};Z_{\text{metabolome}}]\beta\\
# y_{\text{Age}}=\frac{\tilde{y}-\min{\tilde{y}}}{\max{\tilde{y}}-\min{\tilde{y}}}\times 100$$

# %%

def generate_response(embedding_dict, embedding_dim=10, random_state=123):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    embedding_g = embedding_dict['gut_16s'].values
    embedding_m = embedding_dict['metabolome'].values

    beta_g = np.random.randn(embedding_dim) * 5
    beta_m = np.random.randn(embedding_dim) * 5

    resp = np.sum(embedding_g*beta_g, axis=1) + np.sum(embedding_g**2*beta_g, axis=1) +\
            np.sum(embedding_m*beta_m, axis=1) + np.sum(embedding_m**2*beta_m, axis=1) +\
            np.sum((embedding_g*embedding_m)*(beta_g*beta_m), axis=1) + np.random.randn(embedding_g.shape[0])

    resp_norm = (resp-resp.min())/(resp.max()-resp.min())*100
    return resp_norm, (beta_g, beta_m)

# %%
def generate_net_structure(input_dim, output_dim, n_linear_layers=4):
    net_structure=[input_dim]
    for i in range(n_linear_layers-1):
        temp_dim = int((net_structure[-1] + output_dim)/2)
        net_structure.append(temp_dim)
    return net_structure

# %% [markdown]
# ## 2 Train MB-SupCon-cont with different continuous covariates and store embeddings

# %% [markdown]
# ### MB-SupCon-cont

# %%
DEVICE = "cuda:0"
N_EPOCH = 1000

covariate_list = ['Age']

## From tuning
max_idx = {'Age': (0, 2, 3)}

CORR = 0.4
random_seed_list = range(1,13)
BATCH_SIZE = 35

# %%
start_time = time.time()
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
        print("========================================================")
        print(f"Generate simulation data with random seed {seed}:")
        print("========================================================")
        # generate embeddings
        gut_16s_embeding_sim, metabolome_embeding_sim, _ = generate_embeddings(correlation=CORR, random_state=seed, dim=EMBEDDING_DIM)
        embedding_dict = {'gut_16s': gut_16s_embeding_sim,
                          'metabolome': metabolome_embeding_sim}
        # generate original omics
        sim_data_dict = reconstruct_data_by_reverse_PCA(loadings_dict, embedding_dict, 
                                                        {'gut_16s':0, 'metabolome':0}) 
                                                        # Do NOT add mean back = normalized original omcis data
        # generate response
        resp_norm, beta = generate_response(embedding_dict, random_state=seed, embedding_dim=EMBEDDING_DIM)
        print('beta\n', beta)
        
        # MB-SupCon-cont model
        multi_omics_dict = sim_data_dict
        gut_16s_sim = sim_data_dict['gut_16s']
        gut_16s_df_subj = gut_16s_sim.copy()
        gut_16s_df_subj['Age'] = resp_norm ###
        net_dict = {"gut_16s": generate_net_structure(multi_omics_dict['gut_16s'].shape[1], EMBEDDING_DIM, n_linear_layers=2), 
                    "metabolome": generate_net_structure(multi_omics_dict['metabolome'].shape[1], EMBEDDING_DIM, n_linear_layers=2)}


        MB_SupCon_cont = MbSupConContModel(covariate=covariate, indexes=range(gut_16s_df_subj.shape[0]), 
                                           multi_omics_dict=multi_omics_dict, 
                                           df_with_covariates=gut_16s_df_subj, device=DEVICE,
                                           root_folder=SAVE_FOLDER, random_seed=seed) 
        MB_SupCon_cont.initialize(net_dict=net_dict, weight_method_name = WEIGHTING_METHOD, batch_size=BATCH_SIZE,
                                  n_out_features=EMBEDDING_DIM,
                                  temperature=temperature, dropout_rate=dropout, weight_decay=weight_decay)
        
        MB_SupCon_cont.train_model(n_epoch=N_EPOCH, print_hist=False)
        MB_SupCon_cont.plot_training(save=True)
        MB_SupCon_cont.save_training()
        MB_SupCon_cont.save_embedding()

        ### Predict based on embeddings by using different predictors
        predict_embedding_by_method_dict, predict_original_by_method_dict = dict(), dict()
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
            
        ### display results
        mse_e_dict = dict(zip(MB_SupCon_cont.keys,
                              [g_valtest_mse_embedding, m_valtest_mse_embedding]))
        mse_o_dict = dict(zip(MB_SupCon_cont.keys,
                              [g_valtest_mse_original, m_valtest_mse_original]))
        
        mse_embedding_dict, mse_original_dict = [dict.fromkeys(MB_SupCon_cont.keys) for i in range(2)]
        for gm in MB_SupCon_cont.keys:
            mse_embedding_dict[gm], mse_original_dict[gm] =\
                [pd.DataFrame(index=predict_methods_list, 
                              columns=['Validation', 'Testing']) for i in range(2)]
            for ii, method in enumerate(predict_methods_list):
                for jj, which_dataset in enumerate(['val', 'test']):
                    mse_embedding_dict[gm].iloc[ii, jj] = \
                        mse_e_dict[gm][method][s,jj]
                    mse_original_dict[gm].iloc[ii, jj] = \
                        mse_o_dict[gm][method][s,jj]

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
microbiome_rmse_embedding_dict = mse_to_rmse_dict(SAVE_FOLDER, 'microbiome_mse_embedding_dict.pkl', covariate_list=['Age'])
microbiome_rmse_original_dict = mse_to_rmse_dict(SAVE_FOLDER, 'microbiome_mse_original_dict.pkl', covariate_list=['Age'])
metabolome_rmse_embedding_dict = mse_to_rmse_dict(SAVE_FOLDER, 'metabolome_mse_embedding_dict.pkl', covariate_list=['Age'])
metabolome_rmse_original_dict = mse_to_rmse_dict(SAVE_FOLDER, 'metabolome_mse_original_dict.pkl', covariate_list=['Age'])


# %% [markdown]
# ## 3 Scatterplots on lower-dimensional space for random seed 1

# %%
dim_reduction_list = ['pca']
random_seed_list = [1]

DEVICE = "cuda:0"
N_EPOCH = 1000
covariate_list = ['Age']
## From tuning
max_idx = {'Age': (0, 2, 3)}

dropout_list = [0.2, 0.4, 0.6, 0.8]
weight_decay_list = 0.01*2.**np.linspace(-2,5,8)
temp_list = 0.5*2.**np.linspace(-2,2,5)

# %%
start_time = time.time()

for covariate in covariate_list:
    dropout = dropout_list[max_idx[covariate][0]]
    weight_decay = weight_decay_list[max_idx[covariate][1]]
    temperature = temp_list[max_idx[covariate][2]]

    for s, seed in enumerate(random_seed_list):
        # generate embeddings
        gut_16s_embeding_sim, metabolome_embeding_sim, _ = generate_embeddings(correlation=CORR, random_state=seed, dim=EMBEDDING_DIM)
        embedding_dict = {'gut_16s': gut_16s_embeding_sim,
                          'metabolome': metabolome_embeding_sim}
        # generate original omics
        sim_data_dict = reconstruct_data_by_reverse_PCA(loadings_dict, embedding_dict, 
                                                        {'gut_16s':0, 'metabolome':0}) # Do NOT add mean back due to metabolome data
        # generate response
        resp_norm, beta = generate_response(embedding_dict, random_state=seed, embedding_dim=EMBEDDING_DIM)
        
        # MB-SupCon-cont model
        multi_omics_dict = sim_data_dict
        gut_16s_sim = sim_data_dict['gut_16s']
        gut_16s_df_subj = gut_16s_sim.copy()
        gut_16s_df_subj['Age'] = resp_norm ###
        net_dict = {"gut_16s": generate_net_structure(multi_omics_dict['gut_16s'].shape[1], EMBEDDING_DIM, n_linear_layers=2), 
                    "metabolome": generate_net_structure(multi_omics_dict['metabolome'].shape[1], EMBEDDING_DIM, n_linear_layers=2)}
        
        ### load models, not necessary to train them again
        model_save_folder = os.path.join(SAVE_FOLDER, 'models/{}'.format(covariate))
        model_path = os.path.join(model_save_folder, 
                                  'MB-SupCon-cont_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pth'.\
                         format(covariate, N_EPOCH, temperature, dropout, weight_decay, seed))
        
        embedding_save_folder = os.path.join(SAVE_FOLDER, 'embeddings/{}'.format(covariate))
        embedding_path = os.path.join(embedding_save_folder, 
                                      'embeddings_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pkl').\
              format(covariate, N_EPOCH, temperature, dropout, weight_decay, seed)
        
        
        MB_SupCon_cont = MbSupConContModel(covariate=covariate, indexes=range(gut_16s_df_subj.shape[0]), 
                                           multi_omics_dict=multi_omics_dict,
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


