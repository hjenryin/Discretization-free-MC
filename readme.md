# Official implementation for "Discretization-free Multicalibration through Loss Minimization over Tree Ensembles"

## Environment setup

We use conda for environment setup. Python 3.11.11 was used for our experiments. All used library was specified in `environment.yml`, and the required libraries can be installed with the following code:

```
conda env create -f environment.yml
conda activate DFMC
pip install -e .
```

## Code structure

### `src`

- `alg`: This is the place where our proposed algorithms and the baseline algorithms are implemented.
- `utils`: Functions that help split the dataset, calculate the metrics, run the experiments, and and display the results.
- `data`: This folder contains the code to process the datasets.

### `dataset`

Raw datasets (which need to be downloaded separately) and the results of baseline model (neural networks, linear regression, etc.) are stored here. The processing code in `src/data` transforms these into experiment-ready formats.

### `expr`

This folder contains the `jupyter` notebooks of the experiment, as well as the experiment results.

## Replicating the results

### Quick start (using cached results)

The repository includes pre-computed results. The results presented in the paper can be found at:

- Table 1: The `expr/*.ipynb` files for each experiments respectively
- Table 2: `expr/tables/smece/merge.ipynb`
- Figure 2: `expr/plot.ipynb`
  The notebooks will read the results from `expr/results` and display the results in the same format as in the paper.

### Full replication

The code will skip the rerun when the results are present. So to replicate, you need to remove the results and rerun the code.

The whole pipeline looks like the following:

```
Original Data (Not provided here)
Need to be downloaded separately to `dataset/`, see below
      |
      |     Train baseline models and inference
      |     Code in `src/data/*.py`
      ↓
Baseline Predictions
Stored to the csv files in `dataset/`
      |
      |     Hyperparameter search for multicalibration models
      |     Code in `src/alg/*.py`, with the help of `src/utils/expr/hyperparam_sweep.py`
      ↓
Results for different hyperparameters
Stored to `expr/hyperparam_results/`
      |
      |     Run multicalibration algorithms on best hyperparams
      |     Code in `expr/*.ipynb`, with the help of `src/utils/expr/`
      ↓
Multicalibration results
Stored to `expr/results/`
```

- To rerun the uncalibrated baselines, you need to remove the csv files in the `dataset` folder. Then follow the instructions in the first line of `src/data/*.py` to prepare the files. Run `src/data/*.py` to retrain the baseline models.
- To search for the best hyperparameters and rerun the algorithms, remove both `expr/hyperparam_results` and `expr/results` before rerunning the corresponding notebook.
- To only rerun the algorithms given the best hyperparameters in `expr/hyperparam_results`, remove `expr/results` and run the notebook again.

## Contact

jinhy21@mails.tsinghua.edu.cn (Hongyi Henry Jin)

## Cite our paper

```
@misc{jin2025discretizationfreemulticalibrationlossminimization,
      title={Discretization-free Multicalibration through Loss Minimization over Tree Ensembles}, 
      author={Hongyi Henry Jin and Zijun Ding and Dung Daniel Ngo and Zhiwei Steven Wu},
      year={2025},
      eprint={2505.17435},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17435}, 
}
```
