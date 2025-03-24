import numpy as np

test = np.array([
    [1, 2, 3, 11],
    [4, 5, 6, 12],
    [7, 8, 9, 13],
    [0, 0, 0, 1]
])

print(test)

z = test @ np.vstack([0, 0, 1, 0])

print(z)