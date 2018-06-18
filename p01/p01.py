import numpy as np
#Exercise 1
def single_value_matrix(n_rows, n_cols, value):
    return np.zeros((n_rows, n_cols))+value
#Exercise 2
def sum_squared_difference(a, b):
    return np.sum((a-b)**2)
#Exercise 3
def higher_selection(v, e):
    return v[v>e]
#Exercise 4
def means_stds(v, eje="columnas"):
    n = np.array([None,None]);
    if(v.ndim == 2):
        if str(eje) == "filas":
            p = np.mean(v, axis=1)
            d = np.std(v, axis=1)
        elif str(eje) == "columnas":
            p = np.mean(v, axis=0)
            d = np.std(v, axis=0)
        return p,d
    return n;
