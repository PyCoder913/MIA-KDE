# Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators

[![Paper](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2603.10937) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and datasets for the paper **"Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators"**. 

## Overview
The use of synthetic data has become increasingly popular as a privacy-preserving alternative to sharing real datasets, especially in sensitive domains such as healthcare, finance, and demography. [cite_start]However, the privacy assurances of synthetic data are not absolute, and remain susceptible to membership inference attacks (MIAs), where adversaries aim to determine whether a specific individual was present in the dataset used to train the generator. In this work, we propose a practical and effective method to quantify membership disclosure risk in tabular synthetic datasets using Kernel Density Estimators (KDEs). Our KDE-based approach models the distribution of nearest-neighbour distances between synthetic data and the training records, allowing probabilistic inference of membership and enabling robust evaluation via ROC curves.

## Methodology
Our framework introduces two distinct attack models to evaluate risk:
* **True Distribution Attack:** Assumes privileged access to training data, serving as a robust assessment tool for data custodians.
* **Realistic Attack:** A more implementable attack that uses auxiliary data without true membership labels, simulating a real-world adversary.

## Repository Structure
The repository is organized as follows:

* 📁 `Datasets/` - Directory containing the processed datasets used for evaluation.
* 📁 `Experiments/` - Contains Jupyter notebooks for the experiments.
* 📁 `src/` - Core source code for accelerated distance computations and MIA risk evaluation.
* 📄 `LICENSE` - The repository's open-source license.

## Datasets and Generators
Empirical evaluations in this repository span four real-world datasets and six synthetic data generators:
* **Datasets:** MIMIC-IV, UK Census, Texas-100X, and Nexoid COVID-19.
* **Generators:** CTGAN, ADS-GAN, DPGAN, TabDDPM, TVAE, and Bayesian Network. The generators are implemented using the SynthCity framework.

## Key Results
Our method consistently achieves higher F1 scores and sharper risk characterization than prior baseline approaches (data-partitioning methods). Crucially, it provides a practical framework for post-generation risk assessment without requiring computationally expensive shadow models. 

## Citation
If you find this code or research useful, please consider citing our paper:
```bibtex
@article{pathak2026quantifying,
  title={Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators},
  author={Pathak, Rajdeep and Jana, Sayantee},
  year={2026}
}
