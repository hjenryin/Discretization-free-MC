from copy import copy
from utils.dict_utils import xyFromDict, subindexDict

from ..mc import multiaccuracy_error, multicalibration_error, smECEGroup

from alg import MCBoost, LGLG, Grid, BaselineRetrieval, RegressionBooster, Linear
from .hyperparam_sweep import hyperparam_sweep

import functools
import numpy as np
import pandas as pd
import copy

class ExprRunner:
    bin_sets_cal=[10,20,30,50, 75, 100]
    bin_sets_test=[10,20,30,50, 75, 100]

    def __init__(self, ds, fixed_grid=True, loss_type="l1", custom_bin_sets=None, minimize_loss=None, ):
        if custom_bin_sets is not None:
            self.bin_sets_test=custom_bin_sets
            if self.bin_sets_cal is not None:
                self.bin_sets_cal=custom_bin_sets
                print("Setting custom bin sets for both calibration and test")
            else:
                print("Setting custom bin sets for calibration")
                
        assert (minimize_loss is None) or (minimize_loss==True)
        if minimize_loss:
            self.target="Loss"
        self.fixed_grid=fixed_grid
        self.ds=ds
        dataset_name=self.ds.__name__()
        if "-Shift-" in dataset_name:
            dataset_name=dataset_name.split("-Shift-")[0]
        self.loss_type=loss_type
        print("Hyperparameter sweep for",self.__class__.__name__)
        if "Ours" in self.__class__.__name__:
            num_workers = 1
        else:
            num_workers = 16
        self.best_param=hyperparam_sweep(
            param_dict=self.param_dict,
            train_fn=self.run_hyperparam,
            save_path=f"hyperparam_results/{dataset_name}/{self.alg_name}.csv",
            bin_sets=self.bin_sets_cal,
            target=self.target,
            num_workers=num_workers
        )[0]
        # self.eval=self._eval_single_for_fixed_grid if fixed_grid else self._eval_single_for_mean_grid

        
    
    def run_hyperparam(self, **params):
        raise NotImplementedError
    
    def calibrate_with_best_param(self,fold,n_grid,return_model=False):
        best_param=self.best_param
        if len(best_param)==0:
            param_dict={}
        elif len(best_param)==1:
            param_dict=best_param.iloc[0].to_dict()
        else:
            param_dict=best_param.loc[n_grid].to_dict()
        if n_grid is None:
            return self.run_hyperparam(fold,return_model=return_model,**param_dict)
        # Only one row is present
        else:
            return self.run_hyperparam(fold,n_grid,return_model=return_model,**param_dict)
    
    def eval(self, test_ds, fold, n_grid_train):
        dsCal, _, dsTest = test_ds.to_dict(fold, get_test=True)
        # dsTest,_,_=self.ds.to_dict(fold,get_test=True)

        xTest,yTest=xyFromDict(dsTest,label="y")
        _,_,_,model=self.calibrate_with_best_param(fold,n_grid_train, return_model=True)
        if n_grid_train is None:
            # Ours Algorithm

            index=pd.MultiIndex.from_tuples([(n_grid_test,fold) for n_grid_test in self.bin_sets_test],names=['#Test Grids','Fold'])
            columns = pd.Index(
                ["MC Error", "MA Error", "#LS", "Loss", "Group smECE"])
            df = pd.DataFrame(index=index, columns=columns)
            for n_grid_test in self.bin_sets_test:
                x, y = xyFromDict(dsCal, label="y")

                bwp_grid = Grid(
                    model, n_grid_test)
    
                bwp_grid.fit(x, y)
                rg_ece = multicalibration_error( dsTest, bwp_grid, self.loss_type)
                groupwise_smECE = smECEGroup(dsTest, bwp_grid)
                n_LS=len(np.unique(bwp_grid.predict(dsTest)))
                acc_err = multiaccuracy_error(dsTest,bwp_grid,self.loss_type)
                loss=bwp_grid.eval_loss(xTest, yTest)
                df.loc[(n_grid_test, fold), ["MC Error", "MA Error", "#LS", "Loss",
                                             "Group smECE"]] = rg_ece, acc_err, n_LS, loss, groupwise_smECE
            if not self.bin_sets_test:
                loss = model.eval_loss(xTest, yTest)
                groupwise_smECE = smECEGroup(dsTest, model)
                acc_err = multiaccuracy_error(dsTest, model, self.loss_type)
                df.loc[("/", fold), ["MC Error", "MA Error", "#LS",
                                     "Loss", "Group smECE"]] = np.nan, acc_err, np.nan, loss, groupwise_smECE
            return df
            
        else:
            # Other algorithms
            index_grid_change=pd.MultiIndex.from_tuples([(n_grid_test,n_grid_train,fold) for n_grid_test in self.bin_sets_test],names=['#Test Grids','#Train Grids','Fold'])
            columns = pd.Index(
                ["MC Error", "Loss", "#LS", "MA Error", "Group smECE"])
            df=pd.DataFrame(index=index_grid_change,columns=columns)
            x, y = xyFromDict(dsCal, label="y")
                
            for n_grid_test in self.bin_sets_test:
                if n_grid_test>n_grid_train:
                    continue

                model_grid = Grid(
                    model, n_grid_test, )
     
                model_grid.fit(x, y)
                rg_ece = multicalibration_error( dsTest, model_grid, self.loss_type)
                rg_acc_err = multiaccuracy_error(dsTest,model_grid,self.loss_type)
                loss=model_grid.eval_loss(xTest, yTest)
                groupwise_smECE = smECEGroup(dsTest, model_grid)
                n_LS=len(np.unique(model_grid.predict(dsTest)))
                df.loc[(n_grid_test, n_grid_train, fold), ["MC Error", "Loss", "#LS",
                                                           "MA Error", "Group smECE"]] = rg_ece, loss, n_LS, rg_acc_err, groupwise_smECE
            return df

    def eval_wo_round(self, test_ds, fold, n_grid_train):
        dsCal, _, dsTest = test_ds.to_dict(fold, get_test=True)
        # dsTest,_,_=self.ds.to_dict(fold,get_test=True)
        xTest, yTest = xyFromDict(dsTest, label="y")
        _, _, _, model = self.calibrate_with_best_param(
            fold, n_grid_train, return_model=True)
        if n_grid_train is None:
            index_key = ("/", fold)
            index_name = ["#Test Grids", "Fold"]
        else:
            index_key = (n_grid_train, n_grid_train, fold)
            index_name = ["#Test Grids", "#Train Grids", "Fold"]
        column_names = ["MA Error", "Loss",
                        "Group smECE", "Group smECE-no-groupsize"]
        index = pd.MultiIndex.from_tuples(
            [index_key], names=index_name)
        columns = pd.Index(column_names
                           )
        df = pd.DataFrame(index=index, columns=columns)

        loss = model.eval_loss(xTest, yTest)
        groupwise_smECE = smECEGroup(
            dsTest, model, times_group_size=True)
        no_groupwise_smECE = smECEGroup(
            dsTest, model, times_group_size=False)
        acc_err = multiaccuracy_error(dsTest, model, self.loss_type)
        df.loc[index_key, column_names] = acc_err,  loss, groupwise_smECE, no_groupwise_smECE
        return df

class LSBRunner(ExprRunner):
    param_dict=dict(
        # global_gamma=list(np.geomspace(1e-6,1e-4,3,endpoint=True))+[0],
        lr=[0.1,0.3,1],
        feature_subsample=np.linspace(0.1,1,10,endpoint=True),
        tree_depth=[1, 2]
    )
    target='MC ERR'
    rg_params=dict(verbose=False,n_jobs=1,final_round=True,center_mean=False)
    # For fair comparison the result will be rounded to nearest grid. Therefore center_mean can be set to False.

    def __init__(self, *args, **kwargs):
        self.alg_name = "lsb"
        super().__init__(*args, **kwargs)

    def run_hyperparam(self,fold,n_grid,lr,feature_subsample,tree_depth,return_model=False):
        dsCal, dsVal = self.ds.to_dict(fold)
        baseline = BaselineRetrieval()
        (xCal, yCal), (xVal, yVal) = map(functools.partial(xyFromDict, label="y"), [dsCal, dsVal])
        if self.fixed_grid:
            rg = RegressionBooster( init_model=baseline, n_grid=n_grid, learning_rate=lr,
                                   max_features=feature_subsample, tree_depth=int(tree_depth), **self.rg_params)
            grid = Grid(rg, n_grid, 
                        train_previous=True)
        else:
            self.rg_params["center_mean"] = True
            grid = RegressionBooster(init_model=baseline, n_grid=n_grid, learning_rate=lr,
                                     max_features=feature_subsample, tree_depth=int(tree_depth), **self.rg_params)
        grid.fit(xCal, yCal)
        loss_single_fold = grid.eval_loss(xVal, yVal)
        mc_err_single_fold = multicalibration_error( dsVal, grid, self.loss_type,n_grid)
        n_out=len(np.unique(grid.predict(dsVal)))
        if not return_model:
            return loss_single_fold, mc_err_single_fold,n_out
        else:
            return loss_single_fold, mc_err_single_fold,n_out,grid
        
class OursRunner(ExprRunner):
    target="Loss"
    param_dict={"eta":np.geomspace(0.01,1,5,endpoint=True),"subsample":np.linspace(0.1,1,10,endpoint=True)}
    bin_sets_cal=None

    def __init__(self, *args, **kwargs):
        self.alg_name = "tree"
        super().__init__(*args, **kwargs)

    def run_hyperparam(self,fold,eta,subsample,return_model=False):
        dsCal, dsVal = self.ds.to_dict(fold)
        baseline = BaselineRetrieval()
        model=LGLG(previous_model=baseline,eta=eta,subsample=subsample,early_stopping_rounds=50,num_boost_round=5000,max_depth=2)
        (xCal, yCal), (xVal, yVal) = map(functools.partial(xyFromDict, label="y"), [dsCal, dsVal])
        model.fit(xCal, yCal)
        loss = model.eval_loss(xVal, yVal)
        if not return_model:
            return loss, np.nan, np.nan
        else:
            return loss, np.nan, np.nan,model
        

class OursCustomRunner(ExprRunner):
    param_dict = {"eta": np.geomspace(
        0.01, 1, 5, endpoint=True), "subsample": np.linspace(0.1, 1, 10, endpoint=True)}
    bin_sets_cal = None

    def __init__(self, *args, custom_predictor, **kwargs):
        self.alg_name = "tree2"
        self.custom_predictor = custom_predictor
        super().__init__(*args, **kwargs)

    def run_hyperparam(self, fold, eta, subsample, return_model=False):
        dsCal, dsVal = self.ds.to_dict(fold)
        baseline = copy.deepcopy(self.custom_predictor)
        model = LGLG(previous_model=baseline, eta=eta, subsample=subsample,
                     early_stopping_rounds=50, num_boost_round=5000, max_depth=2)
        (xCal, yCal), (xVal, yVal) = map(
            functools.partial(xyFromDict, label="y"), [dsCal, dsVal])
        baseline.fit(xCal, yCal)
        model.fit(xCal, yCal)
        loss = model.eval_loss(xVal, yVal)
        if not return_model:
            return loss, np.nan, np.nan
        else:
            return loss, np.nan, np.nan, model

class MVPRunner(ExprRunner):
    param_dict={"es_ratio":np.linspace(0.1,0.5,5,endpoint=True)}
    target='MC ERR'
    
    def __init__(self, *args, **kwargs):
        self.alg_name = "mvp"
        super().__init__(*args, **kwargs)

    def run_hyperparam(self,fold,n_grid,es_ratio,return_model=False):
        dsCal, dsVal = self.ds.to_dict(fold)
        baseline = BaselineRetrieval()
        (xCal, yCal), (xVal, yVal) = map(functools.partial(xyFromDict, label="y"), [dsCal, dsVal])
        rg=MCBoost(baseline,num_grid=n_grid,es_ratio=es_ratio)
        if self.fixed_grid:
            grid = Grid(rg, n_grid,
                        train_previous=True)
        else:
            grid = rg
        grid.fit(xCal, yCal)
        loss_single_fold = grid.eval_loss(xVal, yVal)
        mc_err_single_fold = multicalibration_error(dsVal, grid, self.loss_type,n_grid)
        n_out=len(np.unique(grid.predict(dsVal)))
        if not return_model:
            return loss_single_fold, mc_err_single_fold,n_out
        else:
            return loss_single_fold, mc_err_single_fold,n_out,grid


class MARunner(ExprRunner):
    target="Loss"
    param_dict = {"l1": np.geomspace(
        1e-6, 1e-2, 5, endpoint=True).tolist()+[0]}
    bin_sets_cal=None
    
    def __init__(self, *args, **kwargs):
        self.alg_name = "ma"
        super().__init__(*args, **kwargs)

    def run_hyperparam(self, fold, l1, return_model=False):
        dsCal, dsVal = self.ds.to_dict(fold)
        baseline = BaselineRetrieval()
        model = Linear(use_previous_output_as_input_feature=False,
                       include_groups=True, previous_model=baseline, l1=l1)
        (xCal, yCal), (xVal, yVal) = map(functools.partial(xyFromDict, label="y"), [dsCal, dsVal])
        model.fit(xCal, yCal)
        loss = model.eval_loss(xVal, yVal)
        if not return_model:
            return loss, np.nan, np.nan
        else:
            return loss, np.nan, np.nan, model


class BaselineRunner(ExprRunner):
    target="Loss"
    param_dict = {}
    bin_sets_cal=None

    def __init__(self, *args, **kwargs):
        self.alg_name = "baseline"
        super().__init__(*args, **kwargs)

    def run_hyperparam(self, fold, return_model=False):
        dsCal, dsVal = self.ds.to_dict(fold)
        baseline = BaselineRetrieval()
        (xCal, yCal), (xVal, yVal) = map(functools.partial(xyFromDict, label="y"), [dsCal, dsVal])
        loss = baseline.eval_loss(xVal, yVal)
        if not return_model:
            return loss, np.nan, np.nan
        else:
            return loss, np.nan, np.nan, baseline
