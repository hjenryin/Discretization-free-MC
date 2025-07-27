import numpy as np
from .base import Base
from functools import partial
import plotly.express as px
from plotly.subplots import make_subplots
from numba import njit
from src.utils import splitDsDict

class MCBoost(Base):
    def __init__(self,init_model=None, num_grid=100,opt_rounds=1000,loss_type="l2",es_ratio=0,max_patch_length=10000,min_data_to_change=0,topk=1,):
        super().__init__(alpha=None)
        self.num_grid=num_grid
        self.opt_rounds=opt_rounds
        self.loss_type=loss_type
        self.max_violation=[]
        self.init_model=init_model
        self.max_patch_length=max_patch_length
        self.out_of_bound_warning=0
        self.min_data_to_change=min_data_to_change
        self.topk=topk
        self.es_ratio=es_ratio
        
    def fit(self, X_dict, y,verbose=False,plot=False,val_X_dict=None,val_y=None,es_on_loss=True,**kwargs):
        self.is_fitted_=True
        
        hooks=[]
        if plot:
            hooks.append(partial(self._loss_hook,X=X_dict,y=y))               
        if val_X_dict is not None and val_y is not None:
            hooks.append(partial(self._early_stopping_hook,X=val_X_dict,y=val_y,es_on_loss=es_on_loss))
        elif self.es_ratio>0:
            ds={**X_dict,"y":y}
            train_ds,val_ds=splitDsDict(ds,(1-self.es_ratio,self.es_ratio))
            train_y,val_y=train_ds.pop("y"),val_ds.pop("y")
            hooks.append(partial(self._early_stopping_hook,X=val_ds,y=val_y,es_on_loss=es_on_loss))
            X_dict,y=train_ds,train_y
            
            
        group=X_dict["group"].to_numpy().transpose().astype(bool)
        if self.init_model is not None:
            pred_init=self.init_model.predict(X_dict)
        else:
            pred_init=None
        self.multivalid_coverage(y,group,pred_init, verbose=verbose,eval_hook=hooks,min_data_to_change=self.min_data_to_change,topk=self.topk)
        
        if plot:
            fig=make_subplots(rows=1, cols=2,subplot_titles=("Loss","Max Violation"))
            fig.add_trace(px.line(y=self.loss_curve).data[0],row=1,col=1)
            fig.add_trace(px.line(y=self.max_violation).data[0],row=1,col=2)
            fig.show()

        if hasattr(self,"early_stopping_history"):
            if plot:
                px.line(y=self.early_stopping_history).show()
            es=np.argmin(self.early_stopping_history)
            self.patches=self.patches[:es+1]
                
        return np.nan
        
    def predict(self,X_dict):
        group=X_dict["group"].to_numpy().astype(bool)
        pred_init=self.init_model.predict(X_dict) if self.init_model is not None else np.ones(len(group))*self.init_constant
        pred_init=pred_init.clip(0,1)
        pred_init_index=self.find_closest_grid_point_index(pred_init,self.num_grid)
        patches=self.patches if isinstance(self.patches,np.ndarray) else np.array(self.patches)
        if patches.size==0:
            ret=self.grid[pred_init_index]
        else:
            pred_ind,_=self.group_patch_traceback(group,pred_init_index,patches,max_trace_length=self.max_patch_length)
            ret=self.grid[pred_ind]
        self.n_bins=len(np.unique(ret))
        return ret

    def eval_loss(self, X_dict, y):
        return self.loss_fn(self.predict(X_dict), y)
    
    def multicalibration_error(self, X_dict, y):
        from ..utils import multicalibration_error
        packed_ds={**X_dict,"y":y}
        return multicalibration_error(packed_ds,self,loss_type=self.loss_type)
    
    def _loss_hook(self,X,y):
        if not hasattr(self,"loss_curve"):
            self.loss_curve=[]
        self.loss_curve.append(self.eval_loss(X,y))       
    
    def _early_stopping_hook(self,X,y,es_on_loss=False):
        if not hasattr(self,"early_stopping_history"):
            self.early_stopping_history=[]
        if es_on_loss:
            self.early_stopping_history.append(self.eval_loss(X,y))
        else:
            self.early_stopping_history.append(self.multicalibration_error(X,y))
        
    
    def find_closest_grid_point_index(self,pred,num_grid):
        pred_grid = (np.floor(pred*2*num_grid) + (np.floor(pred*2*num_grid)+1) % 2)/(2*num_grid)
        b_num = ((pred_grid * (2*num_grid) - 1)/2).astype(int)
        if np.max(b_num>num_grid):
            print("Warning: Some values are out of grid")
            print(">=num_grid",np.sum(b_num>=num_grid),"max",np.max(b_num[b_num>=num_grid]))
        if np.min(b_num<0):
            print("Warning: Some values are out of grid")
            print("<=0",np.sum(b_num<=0),"min",np.min(b_num[b_num<=0]))
            
        return b_num.clip(0,num_grid-1)

    # https://github.com/ProgBelarus/BatchMultivalidConformal/blob/da08edd1f0a91bd40c94688d0d780e778a00f7d6/src/MultivalidAlgorithms/MultivalidCoverage.py

    def multivalid_coverage(self,  y_train,groups, pred_init=None,verbose=False,eval_hook=None,min_data_to_change=0,topk=1):
        num_groups = len(groups)
        loss=self.loss_type
        assert loss in ["l2","l1"], "loss must be 'l2' or 'l1'"
        if loss=="l2":
            loss=lambda x: x**2
        else:
            loss=np.abs
        # initialize function-value distribution on training dataset
        num_grid=self.num_grid
        self.grid = np.linspace(0, 1, num_grid,endpoint=False)+1/(2*num_grid)
        
        if pred_init is None:
            pred_init = np.mean(y_train)
            self.init_constant=pred_init
            pred_init=pred_init*np.ones(len(y_train))
        else:
            pred_init=pred_init.clip(0,1)

        f_val = self.grid[self.find_closest_grid_point_index(pred_init,num_grid)] 
        vals_mask = [f_val == val for val in self.grid]

        # for each intersection of (group, value), 
        # # y_num = number of training points there,
        # y_sum = sum of ground truth values for training points there
        # => y_sum/y_num = empirical mean in this intersection
        y_sum, y_num = np.zeros((num_groups, num_grid)), np.zeros((num_groups, num_grid))
        for i in range(num_groups):
            for j in range(num_grid):
                inds = vals_mask[j] & groups[i]
                y_num[i][j] = np.sum(inds)
                y_sum[i][j] = np.sum(y_train[inds])


        self.patches = []
        cur_topk=0
        for step in range(self.opt_rounds):
            with np.errstate(invalid='ignore', divide='raise'):
                # In case of division by zero, y_sum should be sufficiently small
                assert np.all(y_num!=0) or (np.max(y_sum[y_num==0]) < 1e-8)
                y_sum[y_num==0]=0
                coverage =np.nan_to_num(y_sum/y_num)
            ### find most violated constraint

            grid=self.grid[None,:]
            violations = y_num*loss(grid-coverage)


            sorted_index=np.argsort(-violations,axis=None)
            
            g, val = np.unravel_index(sorted_index[cur_topk], violations.shape)
            if y_num[g, val]<min_data_to_change:
                cur_topk+=1
                if cur_topk>=topk:
                    break
                else:
                    continue
            if verbose:
                print('Max violation in round', step, ' : ', violations[g, val]/len(y_train))
            self.max_violation.append(violations[g, val]/len(y_train))
            # indices of points to be re-valued at this round
            ind_mask = vals_mask[val] & groups[g]
            inds = np.where(ind_mask)[0]

            val_new = self.find_closest_grid_point_index(np.mean(y_train[inds]),num_grid)

            if val == val_new:
                cur_topk+=1
                if cur_topk>=topk:
                    break
                else:
                    continue
            else:
                cur_topk=0

            ### append update to transform[]
            self.patches.append((g, val, val_new))
            if verbose:
                print('Update:', (g, val, val_new))

            # update vals
            vals_mask[val][inds] = False
            vals_mask[val_new][inds] = True

            # update y_sum, y_num
            for i in range(num_groups):
                inds_group_mask = ind_mask & groups[i]

                y_sum[i,val]     -= np.sum(y_train[inds_group_mask])
                y_sum[i,val_new] += np.sum(y_train[inds_group_mask])

                n_group= np.sum(inds_group_mask)
                y_num[i,val]     -= n_group
                y_num[i,val_new] += n_group
            if eval_hook is not None:
                if isinstance(eval_hook,list):
                    for hook in eval_hook:
                        hook()
        self.patches=np.array(self.patches)
        
    @staticmethod
    @njit(nogil=True)
    def group_patch_traceback(group:np.ndarray,pred_ind:np.ndarray,patches:np.ndarray, max_trace_length=10000):
        
        trace_length=np.zeros(len(group))
        for (group_ind, level, level_new) in patches:
            selected=group[:,group_ind] & (pred_ind==level) & (trace_length<max_trace_length)
            trace_length[selected]+=1
            pred_ind[selected]=level_new    
        return pred_ind,trace_length
    
