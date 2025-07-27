import os
import pandas as pd
import itertools
from joblib import Parallel, delayed
from functools import partial
from .expr_runner import LSBRunner, OursRunner, MARunner, BaselineRunner, MVPRunner, ExprRunner
from IPython.display import display
from typing import Dict
import warnings


def eval(ds, runners: Dict[str, ExprRunner], alg,  force_rerun=False, single_thread=False):
    alg_without_brackets=alg.split("(")[0].strip()
    runner=runners[alg_without_brackets]
    is_rounded_algorithm=runner.bin_sets_cal is not None
    print(alg)
    run = partial(runner.eval, test_ds=ds, )
    alg_path=alg

    if alg!=alg_without_brackets:
        alg_orig=alg
        alg_path=alg_without_brackets
        warnings.warn(f"Unknown description, {alg_orig} is treated as {alg_path}")
    path=f"result/{ds.__name__()}/{alg_path}.csv"
    try:
        if force_rerun:
            raise FileNotFoundError
        results_df=pd.read_csv(path,
                               index_col=[0,1] if not is_rounded_algorithm else [0,1,2]
                               )
    except FileNotFoundError:
        dir=os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        jobs=[]
            
        for i in range(10):
            if is_rounded_algorithm:
                for n_grid_train in runner.bin_sets_cal:
                    jobs.append(delayed(run)(fold=i,n_grid_train=n_grid_train))
            else:
                jobs.append(delayed(run)(fold=i,n_grid_train=None))
        results = Parallel(n_jobs=30 if is_rounded_algorithm and (not single_thread) else 1,
                           verbose=50)(jobs)

        results_df=pd.concat(results,axis=0)
        results_df.to_csv(path)
    if is_rounded_algorithm:
        results_df=results_df.unstack(1)
        new_col=pd.MultiIndex.from_tuples([(metric,alg,n_grid) for metric,n_grid in results_df.columns])
        results_df.columns=new_col
    else:
        new_col=pd.MultiIndex.from_tuples([(metric,alg,"/") for metric in results_df.columns])
        results_df.columns=new_col
    ece, loss, n_level, acc_err, group_smECE = results_df["MC Error"], results_df[
        "Loss"], results_df["#LS"], results_df["MA Error"], results_df["Group smECE"]
    return ece, acc_err, n_level, loss, group_smECE


def eval_wo_round(ds, runners: Dict[str, ExprRunner], alg,  force_rerun=False):
    alg_without_brackets = alg.split("(")[0].strip()
    runner = runners[alg_without_brackets]
    is_rounded_algorithm = runner.bin_sets_cal is not None
    print(alg)
    run = partial(runner.eval_wo_round, test_ds=ds)
    alg_path = alg

    if alg != alg_without_brackets:
        alg_orig = alg
        alg_path = alg_without_brackets
        warnings.warn(
            f"Unknown description, {alg_orig} is treated as {alg_path}")
    path = f"result/{ds.__name__()}/{alg_path}_wo_round.csv"
    try:
        if force_rerun:
            raise FileNotFoundError
        results_df = pd.read_csv(path,
                                 index_col=[0, 1] if not is_rounded_algorithm else [
                                     0, 1, 2]
                                 )
    except FileNotFoundError:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        jobs = []

        for i in range(10):
            if is_rounded_algorithm:
                for n_grid_train in runner.bin_sets_cal:
                    jobs.append(delayed(run)(
                        fold=i, n_grid_train=n_grid_train))
            else:
                jobs.append(delayed(run)(fold=i, n_grid_train=None))
        results = Parallel(n_jobs=30 if is_rounded_algorithm else 1,
                           verbose=50)(jobs)

        results_df = pd.concat(results, axis=0)
        results_df.to_csv(path)
    if is_rounded_algorithm:
        results_df = results_df.unstack(1)
        new_col = pd.MultiIndex.from_tuples(
            [(metric, alg, n_grid) for metric, n_grid in results_df.columns])
        results_df.columns = new_col
    else:
        new_col = pd.MultiIndex.from_tuples(
            [(metric, alg, "/") for metric in results_df.columns])
        results_df.columns = new_col

    return results_df["MA Error"], results_df[
        "Loss"], results_df["Group smECE"], results_df["Group smECE-no-groupsize"]


def collect_results(ds, runners, alg_list=None,  force_rerun=False, raw=False, single_thread=False):
    if alg_list is None:
        alg_list=runners.keys()
    if raw:
        ret = [eval_wo_round(ds=ds, runners=runners, alg=alg,
                             force_rerun=force_rerun) for alg in alg_list]
        acc, loss, smECE, smECE_no_groupsize = zip(*ret)
        mae_df = pd.concat(acc, axis=1).astype(float)
        loss_df = pd.concat(loss, axis=1).astype(float)
        smECE_df = pd.concat(smECE, axis=1).astype(float)
        smECE_no_groupsize_df = pd.concat(
            smECE_no_groupsize, axis=1).astype(float)
        return mae_df, loss_df, smECE_df, smECE_no_groupsize_df
    else:
        ret = [eval(ds=ds, runners=runners, alg=alg,
                    force_rerun=force_rerun, single_thread=single_thread) for alg in alg_list]
        ece, acc, n_level, loss, smECE = zip(*ret)
        ece_df = pd.concat(ece, axis=1).astype(float)
        mae_df = pd.concat(acc, axis=1).astype(float)
        n_level_df = pd.concat(n_level, axis=1).astype(float)
        loss_df = pd.concat(loss, axis=1).astype(float)
        smECE_df = pd.concat(smECE, axis=1).astype(float)
        return ece_df, mae_df, n_level_df, loss_df, smECE_df

def prepare_expr_runners(ds,display_hyperparameters=False, fixed_grid=True,custom_bin=None,minimize_loss=None,to_run=["Uncalibrated Baseline",
        "Multiaccurate Baseline",
        "Ours",
        "MCBoost",
        "LSBoost"]):
    
    runners = {
        "Uncalibrated Baseline": BaselineRunner(ds, fixed_grid, custom_bin_sets=custom_bin, minimize_loss=minimize_loss) if "Uncalibrated Baseline" in to_run else None,
        "Multiaccurate Baseline": MARunner(ds, fixed_grid, custom_bin_sets=custom_bin, minimize_loss=minimize_loss) if "Multiaccurate Baseline" in to_run else None,
        "Ours": OursRunner(ds, fixed_grid, custom_bin_sets=custom_bin, minimize_loss=minimize_loss) if "Ours" in to_run else None,
        "MCBoost": MVPRunner(ds, fixed_grid, custom_bin_sets=custom_bin, minimize_loss=minimize_loss) if "MCBoost" in to_run else None,
        "LSBoost": LSBRunner(ds, fixed_grid, custom_bin_sets=custom_bin, minimize_loss=minimize_loss) if "LSBoost" in to_run else None,
    }

    if display_hyperparameters:
        for name,runner in runners.items():
            if name in to_run:
                display(name)
                display(runner.best_param)
    return runners
