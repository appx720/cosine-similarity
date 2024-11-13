import numpy as np


data = np.array([[2, 3], [4, 6], [1, 1]])


def euclidean_distance(x, y):
    return np.sqrt(np.sum(x - y) ** 2)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


results = []

for i in range(len(data)):
    for j in range(i + 1, len(data)):
        dist = euclidean_distance(data[i], data[j])
        sim = cosine_similarity(data[i], data[j])
        results.append({
            'vector_pair': (data[i], data[j]),
            'euclidean_distance': dist,
            'cosine_similarity': sim
        })


for result in results:
    print(f"Vectors: {result['vector_pair']}")
    print(f"  Euclidean Distance: {result['euclidean_distance']:.2f}")
    print(f"  Cosine Similarity: {result['cosine_similarity']:.2f}")