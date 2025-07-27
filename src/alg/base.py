import torch
import numpy as np
import pandas as pd
class Base():
    def __init__(self,alpha=None):

        if alpha is None:
            self.loss_fn=lambda yPred, yTrue: np.mean((yPred-yTrue)**2)
        else:
            self.alpha=alpha
            self.loss_fn = self.MSE_with_penalty       
        
    def MSE_with_penalty(self, yPred, yTrue):
        return torch.mean((yPred-yTrue)**2)+self.alpha*torch.mean(torch.abs(self.model.weight))
    
    
    def fit(self, X_dict, y, verbose=False, keep_params=False):
        self.is_fitted_=True
        raise NotImplementedError()
    
    def predict(self, X_dict):
        raise NotImplementedError()
    
    def eval_loss(self, X_dict, y):
        if self.loss_fn:
            return self.loss_fn(self.predict(X_dict), y)
        else:
            return np.nan
    
    def cat(self, X_dict):
        if not isinstance (X_dict, dict):
            assert isinstance(X_dict, pd.DataFrame) or isinstance(X_dict, np.ndarray), "X_dict must be a dictionary (to be concatenated) or data to be used as is. "
            return X_dict 
        X_df=X_dict["group"].copy()
        return X_df

class BaselineRetrieval(Base):
    # Retrieve the prediction from the dataset
    def __init__(self,alpha=None,clip01=True):
        super().__init__(alpha=alpha)
        self.clip01=clip01
    
    def fit(self,*args,**kwargs):
        pass
    
    def predict(self, X_dict):
        assert "pred" in X_dict, "Prediction not found"
        if self.clip01:
            return np.clip(X_dict["pred"],0,1)
        else:
            return X_dict["pred"]
