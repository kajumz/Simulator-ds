
from time import sleep
from typing import List
from joblib import Parallel, delayed


def parallel(n_jobs=-1):
    """Parallel computing"""
    result = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=5 * n_jobs)
    (delayed(sleep)(0.2) for _ in range(50))
    return result


print(parallel(n_jobs=1))
print(parallel(n_jobs=2))
