from .base import Base
from sklearn.model_selection import train_test_split
import lightgbm as lgb
num_threads=8
import os, contextlib

    
class LGB(Base):
    def __init__(self,**gb_param):
        super().__init__(alpha=None)

        self.params={"objective": "regression"}
        
        default_params={
            "boost":"gbdt", 
            "eta": 0.01,
            "subsample": 0.5,
            "num_boost_round": 2000,
            "early_stopping_rounds": 50,
            "bagging_freq": 1,
            "first_metric_only": True,
            "verbose": -1,
            "num_threads":num_threads,
            "max_bin":8192,
            "min_data_in_bin":1,
            "max_depth": 2,
        }
        default_params.update(gb_param)
        self.params.update(default_params)
        self.loss_curve={}
        self.num_boost_round=self.params.pop("num_boost_round")

        self.early_stopping_met=False
        self._is_booster_fitted=False

    def fit(self, X_dict, y, verbose=False, keep_params=False,valset_tuple=None,init_score=None,force_quiet=True):
        if self._is_booster_fitted:
            raise ValueError("Booster is already fitted. Please create a new instance.")
        else:
            self._is_booster_fitted=True
        # self.params.update(min_data_in_leaf=len(y)//1000)
            

        X=self.cat(X_dict)
        if self.params.get("skip_validation",False):
            self.valid_sets=[lgb.Dataset(X, label=y,free_raw_data=False,init_score=init_score)]
            self.valid_names=["train"]
        else:
            if valset_tuple is not None:
                X_test_dict,y_test=valset_tuple
                X_test=self.cat(X_test_dict)
                X_train, y_train=X, y
            else:
                if init_score is not None:
                    X_train, X_test, y_train, y_test, init_train, init_test = train_test_split(X, y, init_score, test_size=0.3,shuffle=True,random_state=42)
                    self.valid_sets=[lgb.Dataset(X_train, label=y_train,free_raw_data=False,init_score=init_train,params={'verbose': -1}),lgb.Dataset(X_test, label=y_test,free_raw_data=False,init_score=init_test,params={'verbose': -1})]
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,shuffle=True)
                    self.valid_sets=[lgb.Dataset(X_train, label=y_train,free_raw_data=False,params={'verbose': -1}),lgb.Dataset(X_test, label=y_test,free_raw_data=False,params={'verbose': -1})]
            self.valid_names=["train","validation"]
        self.loss_curve={}

        with open(os.devnull, 'w') as devnull:
            if force_quiet:
                redirect=contextlib.redirect_stdout(devnull)
            else:
                redirect=contextlib.nullcontext()
            with redirect:
                self.model = lgb.train(
                    self.params,
                    self.valid_sets[0],
                    num_boost_round=self.num_boost_round,
                    valid_sets=self.valid_sets, # will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds
                    valid_names=self.valid_names, 
                    init_model=self.model if keep_params else None,
                )
    
    def predict(self, X_dict,init_score=0):
        assert self._is_booster_fitted, "Booster is not fitted yet."
        X=self.cat(X_dict)
        return self.model.predict(X)+init_score
    

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np  
           
import functools    
from scipy.interpolate import interp1d
class LGB_Grid_LGB_Grid(LGB):
    def __init__(self,previous_model, n_grid=None,**gb_param):
        super().__init__(**gb_param)
        self.prev_model_as_feature=previous_model
        self.previous_model=previous_model
        if n_grid is None:
            self.f_grid=functools.partial(np.clip,a_min=0,a_max=1)
        else:
            grid=np.linspace(0,1,n_grid,endpoint=False)+1/n_grid/2
            self.f_grid=interp1d(grid, grid, kind='nearest',fill_value=(grid[0],grid[-1]), bounds_error=False)
    
    def cat(self, X_dict):
        prev_predict=self.prev_model_as_feature.predict(X_dict)
        prev_predict=self.f_grid(prev_predict)
        X=super().cat(X_dict)
        X["prev_output"]=prev_predict

            
        return X
        
    
    def fit(self, X_dict, y, verbose=False, keep_params=False,valset_tuple=None,**kwargs):
        prev_pred=self.previous_model.predict(X_dict)

        super().fit(X_dict,y,verbose,keep_params,valset_tuple,init_score=prev_pred,**kwargs)
            
    def predict(self, X_dict):
        prev_pred=self.previous_model.predict(X_dict)

        return super().predict(X_dict,init_score=prev_pred).clip(0,1)
