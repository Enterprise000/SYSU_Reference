import time
import numpy as np

m = 100
n = 150
k = 200
A = np.random.randint(512, 2048, size=(m, n))
B = np.random.randint(512, 2048, size=(n, k))
C = np.zeros((m,k), dtype=int)

start_time = time.time()
for i in range(m):
    for j in range(k):
        for k in range(n):
            C[i][j] = C[i][j] + A[i][k] * B[k][j]
end_time = time.time()
time = end_time - start_time
print(f"-f runtime: {time:.2f} seconds")