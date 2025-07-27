# To retrain the base predictor, prepare the images 
# (extracted from https://archive.org/details/UTKFace) under dataset/UTKFace/data.
# Run this file directly. 

import os
cur_file_dir=os.path.dirname(os.path.abspath(__file__))
root=cur_file_dir+"/../../dataset/UTKFace"


import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np
from data.base import BaseDataset,_resnetPredictor,partition_dataset

    
class _resnet_ds(Dataset):
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.files=os.listdir(root_dir)
        self.transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        img=Image.open(f"{self.root_dir}/{self.files[idx]}")
        meta_data=self.files[idx].split("_")
        return self.transform(img),int(meta_data[0])
        

    
class faceData(BaseDataset):
    def __name__(self):
        name = type(self).__name__
        if len(self.name_note) > 0:
            return name + "_" + self.name_note
        else:
            return name

    def __init__(self, name_note=""):
        self.name_note = name_note
        cal_text = "cal"
        val_text = "val"
        test_text = "test"
        loss_text = "loss"

        folds = []
        training_folds = os.listdir(f"{root}")
        training_folds = [f for f in training_folds if f.startswith("fold")]
        if len(training_folds) == 0:
            partition_dataset(root_dir=root, fileNotFoundErrorStr="Please download and extract UTKFace.tar.gz from https://archive.org/download/UTKFace/UTKFace.tar.gz.")
            training_folds = os.listdir(f"{root}")
            training_folds = [f for f in training_folds if f.startswith("fold")]
        os.makedirs(f"{root}/model", exist_ok=True)

        for i in tqdm(training_folds, desc="Folds", position=0):
            fold_dir = f"{root}/{i}"
            try:
                dfCal = pd.read_csv(f"{fold_dir}/{cal_text}.csv")
                dfVal = pd.read_csv(f"{fold_dir}/{val_text}.csv")
            except FileNotFoundError:
                model = self.get_base_predictor(f"{root}/train", f"{root}/model")
                cal_set = _resnet_ds(f"{fold_dir}/cal")
                val_set = _resnet_ds(f"{fold_dir}/val")
                dfCal = self._predict(cal_set, model)
                dfCal.to_csv(f"{fold_dir}/{cal_text}.csv", index=False)
                dfVal = self._predict(val_set, model)
                dfVal.to_csv(f"{fold_dir}/{val_text}.csv", index=False)
            folds.append((dfCal, dfVal))
        self.folds = folds

        try:
            dfTest = pd.read_csv(f"{root}/{test_text}.csv")
        except FileNotFoundError:
            model = self.get_base_predictor(f"{root}/train", f"{root}/model")
            test_set = _resnet_ds(f"{root}/test")
            dfTest = self._predict(test_set, model)
            dfTest.to_csv(f"{root}/{test_text}.csv", index=False)
        self.test = dfTest

    def get_base_predictor(self, image_dir, model_dir):
        if hasattr(self, "model"):
            return self.model
        model_path = f"{model_dir}/trained_model.pth"
        criterion = torch.nn.MSELoss()
        model = _resnetPredictor().cuda()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            loss_text = "loss"
            train_set = _resnet_ds(image_dir)
            train_set, val_set = random_split(train_set, [0.8, 0.2])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=16)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False, num_workers=16)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            train_loss = []
            val_loss = []

            for epoch in tqdm(range(30), desc="Epoch", position=1):
                model.train()
                epoch_loss = 0
                for x, y in train_loader:
                    x = x.cuda()
                    y = y.cuda().float()
                    optimizer.zero_grad()
                    y_pred = model(x).squeeze()
                    loss = criterion(y_pred, y)
                    loss.backward()
                    epoch_loss += loss.item() * len(y)
                    optimizer.step()
                epoch_loss = epoch_loss / len(train_set)
                train_loss.append(epoch_loss)

                model.eval()
                with torch.no_grad():
                    iter_val_loss = 0
                    for x, y in val_loader:
                        x = x.cuda()
                        y = y.cuda().float()
                        y_pred = model(x).squeeze()
                        iter_val_loss += criterion(y_pred, y).item() * len(y)
                    val_loss.append(iter_val_loss / len(val_set))
                    if val_loss[-1] == min(val_loss):
                        torch.save(model.state_dict(), model_path)

            metric = {"train_loss": train_loss, "val_loss": val_loss}
            px.line(metric, y=["train_loss", "val_loss"], title="Loss").write_image(f"{model_dir}/{loss_text}.png")
            print(f"{model_dir}/{loss_text}.png")
            model = _resnetPredictor().cuda()
            model.load_state_dict(torch.load(model_path))
        self.model = model
        return model
        
    def define_groups(self,include_age):
        self.group_names=[c for c in self.folds[0][0].columns if "_" in c and not c=="agegroup_100-119" and not ".jpg" in c]
        if not include_age:
            self.group_names=[c for c in self.group_names if not c.startswith("age")]
        _to_dict=lambda x: {"group":x[self.group_names],"pred":x["pred"].to_numpy()/120,"y":x["age"].to_numpy()/120}
        self.test_fold=_to_dict(self.test)
        self.folds=[(_to_dict(c),_to_dict(v)) for c,v in self.folds]
        return self
    
    
    def _predict(self,dataset:_resnet_ds,model:_resnetPredictor):
        model.eval()
        model=model.cuda()
        loader=torch.utils.data.DataLoader(dataset,batch_size=512,shuffle=False,num_workers=16)
        preds=[]
        with torch.no_grad():
            for x,_ in loader:
                x=x.cuda()
                y_pred=model(x).squeeze()
                preds.extend(y_pred.cpu().tolist())
        group=[f.split("_")[:3] for f in dataset.files]
        group=list(zip(*group))
        ds={"age":group[0],"gender":group[1],"race":group[2],}
        ds["pred"]=preds
        ds=pd.DataFrame(ds).astype(int)
        ds.replace({
            "gender":{0:"male",1:"female"},
            "race":{0:"White",1:"Black",2:"Asian",3:"Indian",4:"Others"}
        },inplace=True)
        ds["age"]=ds["age"].astype(int)
        ds["s"]=np.abs(ds["age"]-ds["pred"])
        ds["gender"]=ds.gender.astype("category")
        ds["race"]=ds.race.astype("category")
        age_group=pd.cut(ds["age"],bins=[0,5,18,25,35,45,65,85,100,120],right=False,labels=["0-4","5-17","18-24","25-34","35-44","45-64","65-84","85-99","100-119"])
        # Bins are from https://www.census.gov/library/visualizations/interactive/exploring-age-groups-in-the-2020-census.html
        ds["agegroup"]=age_group
        ds_dummies=pd.get_dummies(ds[["gender","race","agegroup"]])
        # 100-119 is too rare and affects stability
        ds=pd.concat([ds,ds_dummies],axis=1)
        return ds
    

      
      
if __name__=="__main__":
    faceData()
