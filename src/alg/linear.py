from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from .base import Base
import numpy as np

class Linear(Base):
    def __init__(self,use_previous_output_as_input_feature,include_groups,previous_model=None, l1=None,l2=None,discretize=None,random_coefficient=False):
        super().__init__()
        l1=None if l1==0 else l1
        l2=None if l2==0 else l2
        if l1 is None and l2 is None:
            self.model = LinearRegression()
        elif l2 is None:
            self.model = Lasso(alpha=l1,max_iter=10000)
        elif l1 is None:
            self.model = Ridge(alpha=l2,max_iter=10000)
        else:
            self.model = ElasticNet(alpha=l1+l2,l1_ratio=l1/(l1+l2))
        self.previous_model = previous_model
        self.discretize=discretize
        self.use_previous_output_as_input_feature=use_previous_output_as_input_feature
        self.include_groups=include_groups
        assert include_groups or use_previous_output_as_input_feature
        self.random_coefficient=random_coefficient
        assert (not random_coefficient) or include_groups
        
    
    def fit(self, X_dict, y):
        if not self.use_previous_output_as_input_feature:
            assert self.include_groups
            X=self.cat(X_dict)
            if self.previous_model is None:
                self.model.fit(X,y)
            else:
                prev_output=self.previous_model.predict(X_dict)
                prev_output=np.clip(prev_output,0,1)
                self.model.fit(X,y-prev_output)
            if self.random_coefficient:
                self.model.coef_=np.random.randn(*self.model.coef_.shape)/10
        else:
            prev_output=self.previous_model.predict(X_dict)
            prev_output=np.clip(prev_output,0,1)
            if self.discretize:
                self.unique_output=np.linspace(0,1,self.discretize)+1/(2*self.discretize)
            else:
                self.unique_output=np.unique(prev_output)

            prev_out_expansion=prev_output[:,None]==self.unique_output[None,:]
            if self.include_groups:
                groups=self.cat(X_dict)
                all_features=np.concatenate([groups,prev_out_expansion],axis=1)
            else:
                all_features=prev_out_expansion
            self.model.fit(all_features,y-prev_output)
            if self.random_coefficient:
                self.model.coef_=np.random.randn(*self.model.coef_.shape)/10
    
    def predict(self, X_dict):
        if not self.use_previous_output_as_input_feature:
            X=self.cat(X_dict)
            if self.previous_model is None:
                return self.model.predict(X)
            else:
                prev_output=self.previous_model.predict(X_dict)
                prev_output=np.clip(prev_output,0,1)
                return self.model.predict(X)+prev_output
        else:
            prev_output=self.previous_model.predict(X_dict)
            prev_output=np.clip(prev_output,0,1)
            # prev_output_round=self.unique_rounding(prev_output)
            prev_out_expansion=prev_output[:,None]==self.unique_output[None,:]
            if self.include_groups:
                groups=self.cat(X_dict)
                all_features=np.concatenate([groups,prev_out_expansion],axis=1)
            else:
                all_features=prev_out_expansion
            return self.model.predict(all_features)+prev_output
