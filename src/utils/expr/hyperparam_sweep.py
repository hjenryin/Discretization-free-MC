import pandas as pd
import numpy as np
from itertools import product
import os
from joblib import Parallel, delayed
from numba import njit
from copy import deepcopy
@njit(nogil=True)
def param_ran(target,all_ran):
    return np.any(np.array([np.allclose(target,a) for a in all_ran]))

def hyperparam_sweep(param_dict, train_fn, save_path, bin_sets=None,folds=10,target="MC ERR",num_workers=16):
    param_dict=deepcopy(param_dict)
    # Fold always goes first
    if len(param_dict)==0:
        return pd.DataFrame({}),None
    assert target in ["Loss","MC ERR"]
    hyper_param_names=list(param_dict.keys())
    if bin_sets is not None:
        param_dict["n_grid"]=bin_sets
    total_params_wo_folds=list(product(*param_dict.values())) 
    types=[type(x) for x in total_params_wo_folds[0]]
    types.insert(0,int) # Adding fold as the first type
    total_params_wo_folds=np.array(total_params_wo_folds).astype(float)
    n_configs, n_params=total_params_wo_folds.shape
    if os.path.exists(save_path):
        metrics_df=pd.read_csv(save_path,index_col=list(range(n_params)))
        already_tested=np.stack(metrics_df.index)
        ran_params_mask=Parallel(n_jobs=30,backend="threading")(delayed(param_ran)(p,already_tested) for p in total_params_wo_folds)
        # Test if the parameters have been tested before
        ran_params_mask=np.array(ran_params_mask)
        total_params_wo_folds=total_params_wo_folds[~ran_params_mask]
        
    total_params=[(fold,*x) for x in total_params_wo_folds for fold in range(folds)]
    
    if len(total_params)>0:
        
        jobs=[]
        params_names=["fold"]+list(param_dict.keys())
        for params in total_params:
            train_params={}
            for t,n,p in zip(types,params_names,params):
                train_params[n]=t(p)
            jobs.append(delayed(train_fn)(**train_params))
        print("In total {} jobs to run".format(len(jobs)))
            
        results=Parallel(n_jobs=num_workers,verbose=50)(jobs)
        indices=pd.MultiIndex.from_tuples(total_params,names=['fold']+list(param_dict.keys()))
        metric=pd.DataFrame(results,index=indices,columns=["Loss","MC ERR","#Output"])
        metrics_df_new=metric.groupby(list(param_dict.keys())).mean()
        if os.path.exists(save_path):
            metrics_df=pd.concat([metrics_df,metrics_df_new])
        else:
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            metrics_df=metrics_df_new
        metrics_df.to_csv(save_path)
    
    best_params={m:[] for m in param_dict.keys()}
    best_params["Loss"]=[]
    best_params["MC ERR"]=[]
    
    if bin_sets is not None:
        for n_grid in bin_sets:
            subset=metrics_df.xs(n_grid,level="n_grid")
            min_index=subset[target].idxmin()
            min_loss=subset.loc[min_index,"Loss"]
            if not isinstance(min_loss,float):
                min_loss=min_loss.item()
            min_mc_error=subset.loc[min_index,"MC ERR"]
            if not isinstance(min_mc_error,float):
                min_mc_error=min_mc_error.item()
            if type(min_index) is not tuple:
                min_index=(min_index,)
            for m_id,metric in enumerate(hyper_param_names):
                best_params[metric].append(min_index[m_id])   
                # n_grid is not in min_index
            best_params["n_grid"].append(n_grid)
            best_params["Loss"].append(min_loss)
            best_params["MC ERR"].append(min_mc_error)
        best_param_df=pd.DataFrame(best_params).set_index("n_grid")
    else:
        min_index=metrics_df[target].idxmin()
        min_loss=metrics_df.loc[min_index,"Loss"]
        min_mc_error=metrics_df.loc[min_index,"MC ERR"]
        best_params["Loss"].append(min_loss)
        best_params["MC ERR"].append(min_mc_error)
        if type(min_index) is not tuple:
            min_index=(min_index,)
        for m_id,metric in enumerate(param_dict.keys()):
            best_params[metric].append(min_index[m_id])
            
        best_param_df=pd.DataFrame(best_params)
    best_param=best_param_df.drop(columns=["Loss","MC ERR"])
    return best_param,best_param_df
            