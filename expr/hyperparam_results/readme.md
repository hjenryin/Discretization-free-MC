The files here are produced during execution of 
```python
expr_runners = prepare_expr_runners(ds,display_hyperparameters=True)
```
in the jupyter notebooks. Internally, it initializes the class `ExprRunner` for all algorithms. Within `ExprRunner.__init__`, `hyperparam_sweep` is called, and the results are saved here.