import os
import pickle
from functools import wraps

def pickle_hook(func):
    """decorator for dataloader, load pkl if once loaded.
    USAGE:
        @pickle_hook
        loadfunc(path_pkl):
            <procedure>
            return data
        data = loadfunc(path_pkl="path to .pkl")
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        path_pkl = kwargs["path_pkl"]

        if(os.path.exists(path_pkl) == True):
            with open(path_pkl, mode='rb') as f:
                X = pickle.load(f)
            return X
        else:
            X = func(*args, **kwargs)
            with open(path_pkl, mode='wb') as f:
                pickle.dump(X, f, protocol=4)
            return X

    return wrapper