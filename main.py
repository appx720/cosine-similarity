import numpy as np


def cos_similar(a: np.array, b: np.array):
    return round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 3)


a = np.array([0, 1, 1, 1])
b = np.array([1, 0, 1, 1])
c = np.array([2, 0, 2, 2])

print(cos_similar(a, b))
print(cos_similar(b, c))
print(cos_similar(a, c))