import torch
import pandas as pd
from copy import copy
import numpy as np   

def xyFromDict(datasetDict,label="y"):
    ds=copy(datasetDict)
    y=ds[label]
    return ds,y 

def subindexDict(d,idx):
    new_dict={}
    for k,v in d.items():
        if isinstance(v,pd.DataFrame):
            new_dict[k]=v.iloc[idx]
        else:
            new_dict[k]=v[idx]
    return new_dict

torch_generator = torch.Generator()
torch_generator.manual_seed(0)

def splitDsDict(dataset,size):
    train,valid={},{}
    
    n=len(dataset[list(dataset.keys())[0]])
    idx=torch.randperm(n,generator=torch_generator)
    if size[0]<1:
        size=(int(size[0]*n),int(size[1]*n))
    tidx=idx[:size[0]]
    vidx=idx[size[0]:]
    
    for k,v in dataset.items():
        if isinstance(v,torch.Tensor) or isinstance(v,np.ndarray):
            train[k]=v[tidx]
            valid[k]=v[vidx]
        elif isinstance(v,pd.DataFrame):
            train[k]=v.iloc[tidx]
            valid[k]=v.iloc[vidx]
        elif isinstance(v,pd.Series):
            raise ValueError("Series not supported, should be converted to array or tensor")

    return train,valid 
