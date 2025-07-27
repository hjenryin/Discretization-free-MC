The files here are produced during execution of 
```python
ece,acc_err,n_level,loss,smECE=run(ds,expr_runners)
acc_err_raw, loss_raw, smECE_raw, smECE_raw_no_groupsize = run(ds, expr_runners, raw=True)
```
in the jupyter notebooks. (`from src.utils.expr.run_pipeline import collect_results as run`) Internally, it reads the best hyperparameters from `expr/hyperparam_result`, calibrates the baseline predictors and output the results for different hyperparameters here.