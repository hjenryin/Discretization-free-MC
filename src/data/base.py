import torch
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn.model_selection import train_test_split
import torch, torchvision, os, shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed

np.random.seed(0)
torch_generator = torch.Generator().manual_seed(42)

class BaseDataset:
    def to_dict(self,i,get_test=False):
        total_test=len(self.test_fold["group"])
        if get_test:
            return *self.folds[i],self.test_fold
        return self.folds[i]

class dictKFoldSplit():
    def __init__(self,dsDict,n_folds,split_size,test_ratio:float,cut_max=np.inf):
        self.n_folds=n_folds
        self.split_size=split_size
        assert sum(split_size)==self.n_folds
        length=None
        for k,v in dsDict.items():
            if length is None:
                length=len(v)
            else:
                assert length==len(v)
        if test_ratio==0:
            calval_index=np.random.permutation(length)
        else:
            calval_index,test_index=train_test_split(np.arange(length),test_size=test_ratio,shuffle=True,random_state=42)
            # Test set is always the same
        if len(calval_index)>cut_max:
            calval_index=calval_index[:cut_max]
        def get_part_from_index(index):
            part={}
            for k,v in dsDict.items():
                if isinstance(v, list):
                    v = np.array(v)
                if isinstance(v,np.ndarray):
                    part[k]=v[index]
                elif isinstance(v,pd.DataFrame):
                    part[k]=v.iloc[index]
                else:
                    raise ValueError("Unknown type"+str(type(v)))
            return part
        
        self.rotating_parts=[]
        for i in range(n_folds):
            fold_index=calval_index[i::n_folds]
            self.rotating_parts.append(get_part_from_index(fold_index))

        self.rotating_parts=cycle(self.rotating_parts)
        if test_ratio>0:
            self.test_fold=get_part_from_index(test_index)
        
    def __iter__(self):
        for i in range(self.n_folds):
            ret=[]
            # Get rotating_parts partitioned into len(self.split_size) parts
            for i in self.split_size:
                this_split={}
                # For each part, take the corresponding portion
                for fold in range(i):
                    new_fold=next(self.rotating_parts)
                    for k,v in new_fold.items():
                        if k not in this_split:
                            this_split[k]=v
                        else:
                            if isinstance(v, list):
                                v = np.array(v)
                            if isinstance(v,np.ndarray):
                                this_split[k]=np.concatenate([this_split[k],v])
                            elif isinstance(v,pd.DataFrame):
                                this_split[k]=pd.concat([this_split[k],v])
                            else:
                                raise ValueError("Unknown type"+str(type(v)))
                ret.append(this_split)
            new_fold=next(self.rotating_parts) # make each fold unique
            yield ret
        
     
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



class _resnetPredictor(torch.nn.Module):
    def __init__(self,n_head=1,project_01=False,backbone="resnet18"):
        super(_resnetPredictor,self).__init__()
        if backbone=="resnet18":
            self.resnet=torchvision.models.resnet18(pretrained=True).cuda()
        elif backbone=="resnet34":
            self.resnet=torchvision.models.resnet34(pretrained=True).cuda()
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        self.resnet.fc=torch.nn.Linear(512,n_head).cuda()
        if project_01:
            assert n_head==1
            self.resnet.fc=torch.nn.Sequential(self.resnet.fc,torch.nn.Sigmoid())
    
    def forward(self,x):
        return self.resnet(x)


def copy_files(files,src_dir,dst_dir,desc="Copying files"):
    def copy_file(f):
        shutil.copy(f"{src_dir}/{f}",f"{dst_dir}/{f}")
    Parallel(n_jobs=8,backend="threading")(delayed(copy_file)(f) for f in tqdm(files,desc=desc))


def partition_dataset(root_dir, data_dir="data", split={"train": 0.4, "test": 0.2, "cal_val": 0.4}, fileNotFoundErrorStr="", other_filter_fn=lambda x: True):
    
    files_dir=f"{root_dir}/{data_dir}"
    files=os.listdir(files_dir)
    files = [f for f in files if f.endswith(
        ".jpg") and other_filter_fn(f.split("/")[-1])]
    total_files=len(files)
    if total_files==0:
        raise FileNotFoundError("No images found in {files_dir}. "+fileNotFoundErrorStr)
    for a,l in split.items():
        aside_list, files = train_test_split(
            files, train_size=l, random_state=42, shuffle=True)
        shutil.rmtree(f"{root_dir}/{a}", ignore_errors=True)
        os.makedirs(f"{root_dir}/{a}",exist_ok=True)
        copy_files(aside_list,files_dir,f"{root_dir}/{a}",desc=f"Copying {a} files")
                    