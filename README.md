# MB-SupCon-cont  
### A Generalized Supervised Contrastive Learning Framework for Integrative Multi-omics Prediction Models

<p align="center">
  <img src="./framework.png" width="750"/>
</p>

<p align="center">
Figure 1. Overview of the MB-SupCon-cont framework for integrative multi-omics prediction.
</p>

---

## Overview

Advancements in multi-omics research have demonstrated the importance of integrating multiple biological data modalities, such as microbiome and metabolomics, to better understand physiological processes and improve predictive modeling in biomedical studies.

Traditional single-omics models often fail to capture the complexity of biological systems. Recent supervised contrastive learning frameworks have improved predictive performance for categorical outcomes, but extending these approaches to continuous covariates remains challenging.

MB-SupCon-cont is a supervised contrastive learning framework designed for both categorical and continuous covariates in multi-omics data. The model improves prediction accuracy by incorporating a generalized contrastive loss function that defines similarity and dissimilarity relationships for continuous covariates.

Using simulation studies and two real-world datasets (Type 2 Diabetes and High-Fat Diet), MB-SupCon-cont consistently achieves lower prediction errors than conventional models while providing improved representation learning for downstream visualization.

---

## Repository Structure

The repository contains three main components.

```
MB-SupCon-cont/
│
├── T2D/              # Type 2 Diabetes study
├── HFD/              # High-Fat Diet study
└── Simulation/       # Simulation experiments
```

### T2D

Scripts and notebooks used to:

- train MB-SupCon-cont models  
- generate embeddings  
- produce PCA visualizations  
- summarize prediction results

### HFD

Contains the same analysis pipeline applied to the High-Fat Diet dataset.

### Simulation

Simulation experiments evaluating model performance under different correlation structures.  
Each subfolder corresponds to an average correlation coefficient

$$
\mu_{\rho} \in \{0.4, 0.6, 0.8\}
$$

---

## Data

### Real-world studies

For the T2D and HFD datasets, all data required for each study are located in

```
./{STUDY}/data
```

where

```
STUDY ∈ {T2D, HFD}
```

### Simulation data

Simulation datasets are generated automatically by the provided scripts.

---

## Code Description

### Real-world studies (T2D and HFD)

Key scripts include:

`MB-SupCon-cont_training.py`

Main script used to

- train MB-SupCon-cont models for different covariates  
- output feature embeddings in the representation space  
- make predictions using embeddings and original data  
- compute average prediction RMSE across train/validation/test splits  
- generate PCA scatter plots for visualization

Other dimensionality reduction methods such as t-SNE and UMAP can also be applied.

`results_visualization_3.ipynb`

Notebook used to summarize and visualize prediction results.

`supervised_loss.py`

Implementation of the supervised contrastive loss function.

`mbsupcon_cont.py`

Model architecture for the MB-SupCon-cont framework.

`utils_eval.py`

Utility functions for evaluation and result processing.

---

### Simulation studies

Each simulation folder corresponds to a specific average correlation coefficient.

The main script

```
MB-SupCon-cont_simulation.py
```

is used to

- generate simulated datasets  
- train MB-SupCon-cont models  
- evaluate prediction performance

Other supporting scripts are shared with those used in the real-data analyses.

---

## Installation

A conda environment is provided for reproducibility.  
All deep learning models are implemented using PyTorch.

### System configuration used in experiments

| Component | Version |
|---|---|
| OS | Linux |
| Python | 3.8.5 |
| PyTorch | 1.7.1 |
| GPU | Tesla V100-SXM2-32GB |

---

## Quick Start

### Clone the repository

```bash
git clone https://github.com/ya61sen/MB-SupCon-cont.git
cd MB-SupCon-cont
```

### Create the conda environment

```bash
conda env create -f environment.yml
conda activate envir_MB-SupCon-cont
```

---

## Training the model

### Example: T2D study

Train the model with embedding dimension 10 and linear weighting.

```bash
cd T2D
python MB-SupCon-cont_training.py \
    --embedding_dim 10 \
    --weighting_method linear
```

### Example: simulation study

For simulation experiments with

- embedding dimension 10  
- linear weighting  
- average correlation coefficient 0.4

```bash
cd Simulation
python MB-SupCon-cont_simulation.py \
    --embedding_dim 10 \
    --weighting_method linear \
    --correlation_coefficient 0.4
```

---

## Contact

Sen Yang  
Department of Public Health Sciences  
Penn State College of Medicine  
Hershey, PA, USA

Email:

- sky5218@psu.edu  
- syang4@pennstatehealth.psu.edu  

---

## Citation

If you use this repository in your research, please cite the associated manuscript once it becomes publicly available.