import numpy as np
def average_folds(df):
    mean=df.groupby(level=0,axis=0).mean()
    std=df.groupby(level=0,axis=0).std()
    return mean,std

def mean_std_1darray(df, alg):
    y_mean,y_std=average_folds(df[alg])
    y_mean,y_std=y_mean.values.squeeze(),y_std.values.squeeze()
    if "Boost" in alg:
        y_mean,y_std=np.diagonal(y_mean),np.diagonal(y_std)
    return alg,y_mean,y_std