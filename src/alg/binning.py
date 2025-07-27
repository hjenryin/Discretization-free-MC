from .base import Base
import numpy as np
  
from scipy import interpolate
class Grid(Base):
    def __init__(self,booster,n,train_previous=False,
                 **kwargs):
        super().__init__()
        self.booster=booster
        self.n=n

        self.grid = np.linspace(0, 1, n,endpoint=False)+1/(2*n)
        self.round=interpolate.interp1d(self.grid,self.grid,kind="nearest",fill_value=(self.grid[0],self.grid[-1]),bounds_error=False)
        
        self.train_previous=train_previous
        self.loss_fn=self.booster.loss_fn
        
    def fit(self, X_dict, y, **kwargs):
        if self.train_previous:
            self.booster.fit(X_dict,y,**kwargs)

        return
    
    def predict(self, X_dict):
        y_pred=self.booster.predict(X_dict).clip(0,1)
        return self.round(y_pred)
