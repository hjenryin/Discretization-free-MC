# To retrain the base predictor, first remove the csv files in the dataset/ISIC directory.
# Then
#   prepare the images (extracted from https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip) under dataset/ISIC/ISIC_2019_Training_Input,
#   prepare the metadata (https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv) at dataset/ISIC/ISIC_2019_Training_Metadata.csv
#   prepare the ground-truth labels (https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv) at dataset/ISIC/ISIC_2019_Training_GroundTruth.csv
# And run this script.


import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
cur_file_dir=os.path.dirname(os.path.abspath(__file__))
root=cur_file_dir+"/../../dataset/ISIC"
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np
from data.base import BaseDataset, dictKFoldSplit,_resnetPredictor, copy_files

    
class _resnet_ds(Dataset):
    def __init__(self,root_dir,labels,train):
        self.root_dir=root_dir
        self.files=os.listdir(root_dir)
        self.labels=labels
        if train:
            self.transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Augmentation
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=180,translate=(0.1,0.1),scale=(0.9,1.1)),
            ])
        else:
            self.transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        image_name=self.files[idx]
        img=Image.open(f"{self.root_dir}/{image_name}")
        return self.transform(img),self.labels[image_name[:-4]]

    
class ISICData(BaseDataset):
    def __name__(self):
        return "ISICData-binary"
    def __init__(self):
        try:
            cal_val=pd.read_csv(f"{root}/cal_val.csv",index_col=0)
            test=pd.read_csv(f"{root}/test.csv",index_col=0)
        except FileNotFoundError:
            os.makedirs(f"{root}/trained_model",exist_ok=True)
            labels=pd.read_csv(f"{root}/ISIC_2019_Training_GroundTruth.csv").set_index("image").drop("UNK",axis=1).sort_index(axis=1)
            self.labels=labels
            self.label_counts=labels.sum().to_numpy()
            labels=pd.from_dummies(labels).astype("category")
            labels.columns=["target"]
            self.categories=labels.target.cat.categories
            labels=labels.target.cat.codes.astype(int)
            self.meta_data = pd.read_csv(
                f"{root}/ISIC_2019_Training_Metadata.csv").set_index("image")
            try:
                train_ds=_resnet_ds(f"{root}/train",labels=labels,train=True)
                cal_val_ds=_resnet_ds(f"{root}/cal_val",labels=labels,train=False)
                test_ds=_resnet_ds(f"{root}/test",labels=labels,train=False)
            except FileNotFoundError:
                self.partition_dataset(
                    root_dir=root,
                    data_dir="ISIC_2019_Training_Input",
                    fileNotFoundErrorStr=f"Please download and extract ISIC_2019_Training_Input.zip from https://challenge.isic-archive.com/dataset/#2019."
                )
                train_ds=_resnet_ds(f"{root}/train",labels=labels,train=True)
                cal_val_ds=_resnet_ds(f"{root}/cal_val",labels=labels,train=False)
                test_ds=_resnet_ds(f"{root}/test",labels=labels,train=False)
            model=self.train(train_ds)

            cal_val=self.eval(model,cal_val_ds,f"{root}/cal_val.csv")
            test=self.eval(model,test_ds,f"{root}/test.csv")
        groups_cal_val=self._get_binary_group(cal_val)
        groups_test=self._get_binary_group(test)
        valid_groups_cal_val=groups_cal_val.columns[groups_cal_val.mean()>0.01]
        valid_groups_test=groups_test.columns[groups_test.mean()>0.01]
        self.group_names=valid_groups_cal_val.intersection(valid_groups_test)
        cal_val,self.test_fold=self._to_dict(cal_val),self._to_dict(test)
        self.folds=list(dictKFoldSplit(cal_val,n_folds=10,split_size=[5,5],test_ratio=0))
        
    def _to_dict(self,df):
        y=df["NV"].to_numpy()
        pred=df["prob_NV"].to_numpy()
        group=self._get_binary_group(df)[self.group_names]
        return {"group":group,"y":y,"pred":pred}
        
        
            
    def partition_dataset(self,root_dir,data_dir="data",fileNotFoundErrorStr=""):
    
        files_dir=f"{root_dir}/{data_dir}"
        files=os.listdir(files_dir)
        files=[f for f in files if f.endswith(".jpg")]
        total_files=len(files)
        if total_files==0:
            raise FileNotFoundError("No images found in {files_dir}. "+fileNotFoundErrorStr)
        train=self.meta_data[self.meta_data.lesion_id.str[:3]=="BCN"].index # ~50%
        cal_val=self.meta_data[self.meta_data.lesion_id.str[:3]=="HAM"].sample(frac=0.75).index # ~30%
        test=self.meta_data.index.difference(train.union(cal_val))
        for name,files in zip(["train","cal_val","test"],[train,cal_val,test]):
            files=files+".jpg"
            os.makedirs(f"{root_dir}/{name}",exist_ok=True)
            copy_files(files,files_dir,f"{root_dir}/{name}",desc=f"Copying {name} files")        
            
    def eval(self,model,eval_set,path):
        try:
            df=pd.read_csv(path,index_col="image")
        except FileNotFoundError:
            loader=torch.utils.data.DataLoader(eval_set,batch_size=512,shuffle=False,num_workers=32)
            model.eval()
            pred=[]
            with torch.no_grad():
                for x,y in loader:
                    x=x.cuda()
                    y=y.cuda()
                    y_pred=model(x)
                    y_pred=torch.softmax(y_pred,dim=1)
                    pred.extend(y_pred.cpu().tolist())
            pred=np.array(pred)
            index=[n[:-4] for n in eval_set.files]
            pred=pd.DataFrame(pred,columns=[f"prob_{c}" for c in self.categories],index=index)
            meta=self.meta_data.loc[index].copy()
            truth=self.labels.loc[index].copy()
            df=pd.concat([meta,truth,pred],axis=1)
            df.to_csv(path)
        return df
        
            
    def train(self,train_ds,):
        
        try:
            model=_resnetPredictor(len(self.label_counts)).cuda()
            model.load_state_dict(torch.load(f"{root}/trained_model.pth"))
        except Exception as e:
            train_set,val_set=torch.utils.data.random_split(train_ds,[0.8,0.2])
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=512, shuffle=False, num_workers=32, pin_memory=True)
            model=_resnetPredictor(len(self.label_counts)).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-3)
            class_imbalance=torch.tensor(1/self.label_counts).float()
            ce=torch.nn.CrossEntropyLoss(weight=class_imbalance)
            criterion=ce.cuda()
            train_loss=[]
            val_loss=[]
            train_top1_acc=[]
            val_top1_acc=[]
            best_acc=0
            for epoch in tqdm(range(30), desc="Epoch", position=1):
                model.train()
                epoch_loss=0
                epoch_correct=0
                epoch_total=0
                
                for x,y in train_loader:
                    x=x.cuda()
                    y=y.cuda()
                    optimizer.zero_grad()
                    y_pred=model(x)
                    loss=criterion(y_pred,y)
                    loss.backward()
                    epoch_loss+=loss.item()*len(y)
                    optimizer.step()
                    epoch_correct+=(y_pred.argmax(-1)==y).sum().item()
                    epoch_total+=len(y)
                epoch_loss=epoch_loss/len(train_set)
                train_loss.append(epoch_loss)
                train_top1_acc.append(epoch_correct/epoch_total)
                model.eval()
                with torch.no_grad():
                    epoch_val_loss=0
                    epoch_val_correct=0
                    epoch_val_total=0
                    for x,y in val_loader:
                        x=x.cuda()
                        y=y.cuda()
                        y_pred=model(x)
                        epoch_val_loss+=criterion(y_pred,y).item()*len(y)
                        epoch_val_correct+=(y_pred.argmax(-1)==y).sum().item()
                        epoch_val_total+=len(y)
                    val_loss.append(epoch_val_loss/len(val_set))
                    val_top1_acc.append(epoch_val_correct/epoch_val_total)
                if val_top1_acc[-1]>best_acc:
                    best_acc=val_top1_acc[-1]
                    print(f"Saving model with accuracy {best_acc:.4f} at epoch {epoch}")
                    torch.save(model.state_dict(), f"{root}/trained_model.pth")
            # print(len(train_loss),len(val_loss))
            metric={"train_loss":train_loss,"val_loss":val_loss}
            # print(metric)
            px.line(metric,y=["train_loss","val_loss"],title="Loss").write_image(f"{root}/loss.png")
            px.line({"train_top1_acc":train_top1_acc,"val_top1_acc":val_top1_acc},y=["train_top1_acc","val_top1_acc"],title="Accuracy").write_image(f"{root}/acc.png")
            print(f"{root}/loss.png")
        return model
    
    def _get_binary_group(self,ds):
        ds=ds.copy()
        ds["age"]=ds["age_approx"].astype(float) # contains nan
        ds["sex"]=ds.sex.astype("category")
        ds["anatomsite"]=ds["anatom_site_general"].replace(" ","").replace("/","").astype("category")
        ds_dummies=pd.get_dummies(ds[["sex","anatomsite"]])
        age_cut=pd.cut(ds["age"],bins=[0,5,18,25,35,45,65,85,100,120],right=False,labels=["0-4","5-17","18-24","25-34","35-44","45-64","65-84","85-99","100-119"])
        age_cut=pd.get_dummies(age_cut)
        return pd.concat([ds_dummies,age_cut],axis=1)

      
if __name__=="__main__":
    ISICData()
