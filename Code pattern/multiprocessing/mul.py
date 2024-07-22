import time
import multiprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#Generate a matrix 50000x1000
size = 1000
num_cores = multiprocessing.cpu_count()
#print(num_cores)
X = np.random.random((size, 100))
y = np.random.randint(2, size=size)

for n_jobs in range(1, num_cores+1):
    start_time = time.time()
    rfc = RandomForestClassifier(n_jobs=n_jobs)
    rfc.fit(X, y)
    end_time = time.time()

    print(f'n_jobs: {n_jobs} | time fitting: {end_time - start_time:.4f}')