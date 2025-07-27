from .LSBoost import LSBoostingRegressor as _LSBoost_shm
from .LSBoost_no_shared_memory import LSBoostingRegressor as _LSBoost_nshm
from ..base import Base
import src.utils as utils
from sklearn.tree import DecisionTreeRegressor
import contextlib,os
from sklearn.model_selection import train_test_split

class RegressionReductionWrapper(Base):
    def __init__(self, init_model=None,n_grid=100,verbose=True,original_implementation=False,max_features=None,tree_depth=1,**kwargs):
        super().__init__()
        self.verbose=verbose
        self.name="RegressionReduction"
        assert init_model is None or hasattr(init_model,"predict"), "Initial model must have predict method"
        self.init_model=_InitModelWrapper(init_model)
        default_param={
            'T': 100,
            'num_bins': n_grid,
            'min_group_size': 5,
            'global_gamma': 0, # For overfitting control, but since we have early stopping, it's not necessary.
            'weak_learner': DecisionTreeRegressor(max_depth=tree_depth,max_features=max_features),
            'learning_rate': 1,
            'initial_model': self.init_model,
            "n_jobs":4
        }
        default_param.update(kwargs)
        if original_implementation:
            self.model=_LSBoost_shm(**default_param)
        else:
            default_param["quiet"]=not verbose
            self.model=_LSBoost_nshm(**default_param)
    
    def predict(self, X_dict):
        X=self.cat(X_dict)
        if self.init_model is not None:
            self.init_model.load_data_dict(X_dict)
        return self.model.predict(X.to_numpy())

    def fit(self, X_dict, y):
        index=np.arange(len(y))
        train_index,val_index=train_test_split(index, train_size=0.7,random_state=42) 
        trainDict=utils.subindexDict(X_dict,train_index)
        valDict=utils.subindexDict(X_dict,val_index)
        if self.init_model is not None:
            self.init_model.load_data_dict((trainDict,valDict))
        X=self.cat(X_dict).to_numpy()
        if self.verbose:
            self.model.fit_validation(
                X[train_index],y[train_index],X[val_index],y[val_index]
            )
        else:
            # Redirect output to devnull
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                self.model.fit_validation(
                X[train_index],y[train_index],X[val_index],y[val_index]
                )
        return self
    
import numpy as np
class _InitModelWrapper:
    def __init__(self, model):
        self.model=model
    def predict(self, X):
        if isinstance(self.data_dict,tuple):
            for dd in self.data_dict:
                ddg=dd["group"]
                if X.shape==ddg.shape and np.all(X==ddg):
                    return self.model.predict(dd)
            raise ValueError("Base Predictor not found.")
        else:
            assert np.all(X==self.data_dict["group"])
            return self.model.predict(self.data_dict)

    def load_data_dict(self,X_dict):
        self.data_dict=X_dict
    
        
        
        
        
        
        