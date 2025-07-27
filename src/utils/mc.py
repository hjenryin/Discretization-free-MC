import numpy as np
import torch
from .dict_utils import xyFromDict
from numba import njit,prange
from relplot import smECE


def _test_binary(array):
    array=array.astype(float)
    return np.all((array==0) | (array==1))

def multiaccuracy_from_single_model(model,dataset,loss_type):
    error=[]
    xTest, yTest = xyFromDict(dataset,label="y")
    pred = model.predict(xTest)
    if isinstance(pred, torch.Tensor):
        pred = np.array(pred)
    loss_fn=np.square if loss_type=="l2" else np.abs
    error.append(loss_fn((yTest-pred).mean()))
    for group,group_indicator in xTest["group"].items():
        if not _test_binary(group_indicator.to_numpy()):
            continue
            # For some debug purposes, groups may not be binary. Skip them.
        inGroup = group_indicator.to_numpy().astype(bool)
        yGroup = yTest[inGroup]
        predGroup = pred[inGroup]
        if len(yGroup)==0:
            error.append(0)
        else:
            error.append(loss_fn((yGroup-predGroup).mean()))
    group_size=xTest["group"].to_numpy().mean(0)
    return error, (error[1:]*group_size).max()
    

def multiaccuracy_error(dataset,models_dict,loss_type='l2',):
    if not isinstance(models_dict,dict):
        models_dict={"unnamed":models_dict}
    result={name:[] for name in models_dict}
    err={name:0 for name in models_dict}

    for name, model in models_dict.items():
        result[name],err[name]=multiaccuracy_from_single_model(model,dataset,loss_type)
    if len(result)==1:
        return err[list(result.keys())[0]]
    else:
        return result


def multicalibration_error(dataset, models_dict,loss_type='l2',bin_results_n=None,print_results=False,return_groupwise=True):
    # If only a single model is used, the MC error is returned directly, and not the groupwise errors even if return_groupwise is True
    if not isinstance(models_dict,dict):
        models_dict={"unnamed":models_dict}
    xTest, sTest = xyFromDict(dataset)
    results={name:[] for name in models_dict}
    err={name:{} for name in models_dict}
    range_sizes={}
    for name, model in models_dict.items():
        pred=model.predict(xTest)
        range_size=len(np.unique(pred))
        range_sizes[name]=range_size
        if range_size>1000 and bin_results_n is None:
            results[name]=[np.nan]*len(xTest["group"].columns)
            continue
        group=xTest["group"].to_numpy()
        ret=multicalibrate_error_groupwise(pred,group,sTest,loss_type=loss_type,bin_results_n=bin_results_n,print_details=print_results)
        if return_groupwise:
            results[name]=ret[0].tolist()
        else:
            results[name]=ret[2]
        err[name]=ret[-1]
    # print(range_sizes)
    if len(results)==1:
        return err[list(results.keys())[0]]
    else:
        return results

@njit(parallel=True,nogil=True)
def multicalibrate_error_groupwise(yPred,group_indicators:np.ndarray,scores,loss_type='l2',bin_results_n=None,print_details=False,):

    assert loss_type=='l1' or loss_type=='l2' or loss_type=='linf'

    
    error_on_groups=np.zeros(group_indicators.shape[1])
    max_patch_violation=np.zeros(group_indicators.shape[1])
    if bin_results_n is None:
        model_range=np.unique(yPred)
        yp,mr=np.broadcast_arrays(yPred[None,:],model_range[:,None]) # For numba compatibility
        value_mask=(yp==mr)
    else:
        bin_border=np.linspace(0,1,bin_results_n+1)
        bin_id=np.digitize(yPred,bins=bin_border[1:-1])
        bi,brn=np.broadcast_arrays(bin_id[None,:],np.arange(bin_results_n)[:,None])
        value_mask=(bi==brn)
            
    for enum in prange(group_indicators.shape[1]):
        group_indicator=group_indicators[:,enum]
        if group_indicator.sum()==0:
            continue
        inGroup = group_indicator>0
        group_size=np.sum(inGroup)
        ig=np.broadcast_to(inGroup[None,:],value_mask.shape)
        value_group_mask=value_mask*ig
        cond_prob=value_group_mask.sum(axis=1)/group_size
        nonzero_masks=cond_prob>0
        nonzero_probs=cond_prob[nonzero_masks]
        losses=np.zeros_like(nonzero_probs)

        vgmnm=value_group_mask[nonzero_masks]
        for i in prange(len(vgmnm)):
            mask=vgmnm[i]
            err=(scores[mask]-yPred[mask]).mean()
            if loss_type=='l2':
                losses[i]=np.square(err)
            else:
                losses[i]=np.abs(err)

        errors=nonzero_probs*losses
        max_error=np.max(errors)
        max_patch_violation[enum]=max_error 
        if loss_type=='linf':
            error=max_error
        else:           
            error=np.sum(errors)
        error_on_groups[enum]=error
    group_size=np.sum(group_indicators,axis=0) / group_indicators.shape[0]
    cal_errors=error_on_groups*group_size
    max_error_index=np.argmax(group_size*error_on_groups)
    cal_error=np.max(cal_errors)
    if print_details:
        print("Max patch Violation: ",max_patch_violation, "at group",np.argmax(max_patch_violation))
        print("Max error: ",cal_error, "at group",max_error_index)
    return error_on_groups,max_patch_violation,cal_error
    
def smECEGroup(dataset, models_dict, times_group_size=True):
    if not isinstance(models_dict, dict):
        models_dict = {"unnamed": models_dict}
    xTest, yTest = xyFromDict(dataset, label="y")
    results = {name: [] for name in models_dict}
    err = {name: {} for name in models_dict}
    range_sizes = {}
    for name, model in models_dict.items():
        pred = model.predict(xTest)
        range_size = len(np.unique(pred))
        range_sizes[name] = range_size
        group = xTest["group"].to_numpy()
        all_errors = np.zeros(group.shape[1])
        for g_index in range(group.shape[1]):
            g = group[:, g_index]
            pred_g = pred[g]
            sTest_g = yTest[g]
            # print(g_index)
            error = smECE(pred_g, sTest_g)
            all_errors[g_index] = error
        if times_group_size:
            group_mean = group.mean(0)
            err[name] = (all_errors*group_mean).max()
        else:
            err[name] = all_errors.max()

    if len(results) == 1:
        return err[list(results.keys())[0]]
    else:
        return results
