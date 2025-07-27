# To rerun the base model, just directly remove the csv files in dataset/folktables/cache. 
# This code will handle the download of the data for you.


import folktables
import warnings

from data.base import dictKFoldSplit, BaseDataset
import numpy as np
import os
import pandas as pd
from folktables import ACSIncome,ACSTravelTime
from folktables import adult_filter
from typing import Union,Literal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
data_cache_path = os.path.join(os.path.dirname(
    __file__), "../../dataset/folktables/cache")
os.makedirs(data_cache_path, exist_ok=True)
cur_dir=os.path.dirname(os.path.abspath(__file__))

def preprocess_folktables(data,features:list,target,preprocess_fn,fill_na=[],remove_na=[],target_postprocess=None,regression_shrink_ratio=1.0):
    
    df = preprocess_fn(data)
    

    df= df[features+[target]]
    
    fill_na=[f for f in fill_na if f in features]
    remove_na=[f for f in remove_na if f in features]+[target]
    
    df.fillna({k:0 for k in fill_na},inplace=True)
    # Remove the rows if na
    df.dropna(axis="index",subset=remove_na,inplace=True)
    if target_postprocess:
        df[target]=target_postprocess(df[target])
    if target_postprocess and (regression_shrink_ratio!=1.0):
        warnings.warn("Regression shrink ratio is not 1.0 but regression is False. Ignoring the shrink ratio")
    else:
        df[target]=df[target]/regression_shrink_ratio
    df=df.reset_index(drop=True)
    return df



class FolkDataset(BaseDataset):
    def __name__(self):
        return "FolkDataset-"+self.task_name+"-"+self.states_name
    
    # https://github.com/ProgBelarus/BatchMultivalidConformal/blob/da08edd1f0a91bd40c94688d0d780e778a00f7d6/experiments/FolktablesExperiment.ipynb
    def __init__(self, task="income", feature_pruning_threshold=0.001, year="2018", states=["CA"], quiet=False, download=False):

        self.task_name=task
        self.quiet=quiet
        data_source = folktables.ACSDataSource(
            survey_year=year, horizon='1-Year', survey='person',root_dir=f"{cur_dir}/../../dataset/folktables")
        acs_data = data_source.get_data(states=states, download=download) #
        self.states_name=",".join(states)

        # regression
        regression_shrink_ratio=1.0
        if task=="income":
            problem=ACSIncome
            regression_shrink_ratio=100000
            score=lambda x,y: np.abs(x-y)
        elif task=="travel_time":
            problem=ACSTravelTime
            regression_shrink_ratio=141
            score=lambda x,y: np.abs(x-y)
        # classification
        elif task=="income-classify":
            problem=ACSIncome
        else:
            raise ValueError(f"Unknown task {task}")
        self.regression_task=task in ["income","travel_time"]
        self.features=problem.features
        self.target=problem.target
        if task in ["travel_time"]:
            data_filter=problem._preprocess
        else:
            data_filter=adult_filter     
        if task=="income":
            data_filter=lambda x: adult_filter(x)[x["PINCP"]<100000]     
            
        # columns with lots of nans or numeric data like postal code that's actually categorical
        feature_removal = ["PUMA", "POWPUMA", "ESP", "GCL"]
        for fr in feature_removal:
            if fr in self.features:
                self.features.remove(fr)
        fill_na=["JWMNP","WKHP"] # These should be numerical data that can be filled with 0
        remove_na=["POVPIP"] # numerical data, can't be categorized and can't be filled with 0.

        definitions = data_source.get_definitions(download=True)
        self.categories = folktables.generate_categories(features=self.features, definition_df=definitions)
        for k,v in self.categories.items():
            self.categories[k]={kk:vv.replace(" ","_").replace(",","_").replace(":","") for kk,vv in v.items()}
        # This is not the same as features. It only records categorical data.
        self.full_data=preprocess_folktables(
            acs_data,
            features=self.features,
            target=self.target,
            preprocess_fn=data_filter,
            fill_na=fill_na,
            remove_na=remove_na,
            target_postprocess=problem.target_transform if not self.regression_task else None,
            regression_shrink_ratio=regression_shrink_ratio
            )
        self.feature_pruning_threshold=feature_pruning_threshold

        
        self.X=self.full_data.copy()
        self.y=self.X.pop(self.target).to_numpy()
        self.X_categorized=self.X.replace(self.categories) # Categorical Data
        for col in self.categories.keys():
            self.X_categorized[col]=self.X_categorized[col].astype("category")
        self.X=pd.get_dummies(self.X_categorized) # One-hot data
        # For stability issue, remove columns that barely exists
        row_avg=self.X.mean(axis=0) 

        col_mask=(row_avg>self.feature_pruning_threshold) & (row_avg<1-self.feature_pruning_threshold)
        numerical_cols=[c for c in self.X.columns if not any([c.startswith(cat) for cat in self.categories.keys()])]
        col_mask[col_mask.index.isin(numerical_cols)]=True
        self.X=self.X.loc[:,col_mask]
        self.X = self.X.astype(np.float64) 

        (self.full_data_train_base,self.full_data,
         self.X_train_base,self.X,
         self.X_categorized_train_base,self.X_categorized,
         self.y_train_base,self.y)=train_test_split(
             self.full_data,
             self.X,
             self.X_categorized,
             self.y,
             train_size=50000,random_state=42
             )
        #  Train is for base predictor, test is for cal+val+test
        self.X=self.X.reset_index(drop=True)
        self.X_categorized=self.X_categorized.reset_index(drop=True)
        self.full_data=self.full_data.reset_index(drop=True)
        try:
            df = pd.read_csv(os.path.join(
                data_cache_path, self.__name__()+".csv"))
            self.pred = df["pred"].to_numpy()
            self.y = df["y"].to_numpy()
        except FileNotFoundError:
            if self.regression_task:
                self.basePredictor = Ridge(alpha=1e-5)
                train_X = self.full_data_train_base.drop(
                    columns=[self.target])
                other_X = self.full_data.drop(columns=[self.target])
            else:
                self.basePredictor = LinearSVC()
                train_X = self.full_data_train_base.drop(
                    columns=[self.target])
                other_X = self.full_data.drop(columns=[self.target])

            print(train_X.shape, other_X.shape)

            self.basePredictor.fit(train_X, self.y_train_base)

            self.y = self.y.astype(np.float64)
            if self.regression_task:
                self.pred = self.basePredictor.predict(other_X)
            else:
                self.pred = self.basePredictor.decision_function(other_X)
                self.pred = 1/(1+np.exp(-self.pred))
            with open(os.path.join(data_cache_path, self.__name__()+".csv"), "w") as f:
                df = pd.DataFrame({"y": self.y, "pred": self.pred})
                df.to_csv(f)
        self.group_product_cache={}

       
  
    def define_groups(self,group_names:Union[list[str],Literal["all"]]=["SEX", "RAC1P"], group_pruning_threshold=0.01):

        group_names=list(self.categories.keys())+["AGEP"] if group_names=="all" else group_names    
        grouping_df=self.X
        if "AGEP" in group_names:
            age_group=pd.cut(self.X["AGEP"],bins=[0,5,18,25,35,45,65,85,100,120],right=False,labels=["0-4","5-17","18-24","25-34","35-44","45-64","65-84","85-99","100-119"])
            # Bins are from https://www.census.gov/library/visualizations/interactive/exploring-age-groups-in-the-2020-census.html
            age_group=pd.get_dummies(age_group,prefix="AGEP")
            grouping_df=pd.concat([grouping_df,age_group],axis=1)
            grouping_df.drop("AGEP",axis=1,inplace=True)
        if "SCHL" in group_names:
            edu_int=self.full_data.SCHL
            edu_group=pd.cut(edu_int,bins=[0,16,18,20,21,22,25],right=False,labels=["NoHighSchool","HighSchool","SomeCollege","Associate","Bachelor","Advanced"])
            # Bins are from https://www.census.gov/newsroom/press-releases/2022/educational-attainment.html
            edu_group=pd.get_dummies(edu_group,prefix="SCHL")
            grouping_df=pd.concat([grouping_df,edu_group],axis=1)
            grouping_df.drop([c for c in self.X.columns if c.startswith("SCHL")],axis=1,inplace=True)
        if "OCCP" in group_names:
            # a fine categorization of OCCP
            occp_nan_changed=self.full_data["OCCP"].fillna(10000)
            occp_group=pd.cut(occp_nan_changed,bins=list(np.arange(0,10200,100)),right=False,labels=[str(i*100) for i in range(101)])
            
            grouping_df=pd.concat([grouping_df,pd.get_dummies(occp_group,prefix="OCCP")],axis=1)
            grouping_df.drop([c for c in self.X.columns if c.startswith("OCCP")],axis=1,inplace=True)
            
        grouping_df=grouping_df[[c for c in grouping_df.columns if any([c.startswith(g) for g in group_names])]]
        relevant_features=[c for c in self.X.columns if any([c.startswith(g) for g in group_names])]
        group_mean=grouping_df.mean(axis=0)
        self.group = grouping_df.copy().loc[:,group_mean>group_pruning_threshold]
        all_features=list(self.X.columns)
        
        feature_category=[f.split("_")[0] for f in all_features]
        unique_features=set(feature_category)
        feature_count={f:feature_category.count(f) for f in unique_features}
        if not self.quiet:
            print("Number of features:",len(self.X.columns),feature_count,"; Number of groups:",len(self.group.columns))
        
        
        self.group = self.group.astype(np.float64)
        self.group_mean=self.group.mean(axis=0)
        # sort the columns according to group_mean
        self.group=self.group.loc[:,self.group_mean.sort_values(ascending=False).index]
        self.group_mean=self.group_mean.sort_values(ascending=False)
        # print(self.group_mean)

        irrelevant_features=set(self.X.columns)-set(relevant_features)
        irrelevant_features=list(irrelevant_features)
        self.irrelevantX=self.X[irrelevant_features]
        self.relevantX=self.X[relevant_features]
        
        self.X_categorized_relevant=self.X_categorized[group_names]
        self.X_categorized_irrelevant=self.X_categorized.drop(columns=group_names)
        
        self.y_val=self.y
        self.group_vec=self.group.to_numpy()
        if not self.quiet:
            print(self.irrelevantX.shape,self.relevantX.shape,self.y.shape)
        self.g=self.group_vec.shape[1]
        
        whole = {"group": self.group.astype(
            bool), "pred": self.pred, "y": self.y_val}
        folds=dictKFoldSplit(whole,n_folds=10,split_size=[5,5],test_ratio=0.4,cut_max=30000)
        self.folds=list(folds)
        self.ds_size=len(self.folds[0][0]["y"])  
        
        self.test_fold=folds.test_fold
        self.group_names=self.group.columns.to_list()
        return self

    def __len__(self):
        return len(self.y_val)

        

if __name__=="__main__":
    ds = FolkDataset(task="income",  states=["CA"], download=False)
    ds.define_groups(group_names=["RAC1P","SEX","AGEP","SCHL","OCCP"])
    print(ds.to_dict(0)[0]["pred"])

# %%
