import heapq
import numpy as np


class IndexScore:
    def __init__(self, similarity, index):
        self.similarity = similarity
        self.index = index

    # For max-heap using heapq (which is a min-heap), negate the comparison
    def __lt__(self, other):
        return self.similarity > other.similarity


class DenseRows:
    def __init__(self, n, d, data):
        self.n = n
        self.d = d
        self.data = data  # flat list or numpy array, row-major


class CSR:
    def __init__(self, n, indptr, indices, data, degree, two_m):
        self.n = n
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.degree = degree
        self.two_m = two_m


def cosine(row_i, row_j):
    dot = np.dot(row_i, row_j)
    norm_i = np.linalg.norm(row_i)
    norm_j = np.linalg.norm(row_j)
    if norm_i == 0 or norm_j == 0:
        return 0.0
    return float(dot / (norm_i * norm_j))


def weighted_knn(dr, k):
    adjacency_matrix = similarities(dr, k)
    return graph_creation(adjacency_matrix, dr)


def similarities(dr, k):
    n = dr.n
    data = np.array(dr.data, dtype=np.float32).reshape(n, dr.d)

    adjacency_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        row_i = data[i]
        scores = []

        for j in range(n):
            if i == j:
                continue
            row_j = data[j]
            sim = cosine(row_i, row_j)
            scores.append(IndexScore(sim, j))

        heapq.heapify(scores)

        for _ in range(min(k, len(scores))):
            item = heapq.heappop(scores)
            j = item.index
            adjacency_matrix[i][j] = item.similarity
            adjacency_matrix[j][i] = item.similarity  # symmetric

    return adjacency_matrix


def graph_creation(ad, dr):
    n = dr.n

    indptr = [0] * (n + 1)
    indices = []
    data = []
    degree = [0.0] * n
    two_m = 0.0
    total_edges = 0

    for cell in range(n):
        row = ad[cell]
        deg = 0.0
        for j in range(len(row)):
            w = row[j]
            if w != 0:
                indices.append(j)
                data.append(float(w))
                deg += w
                total_edges += 1

        indptr[cell + 1] = total_edges
        degree[cell] = deg
        two_m += deg

    return CSR(
        n=n,
        indptr=indptr,
        indices=indices,
        data=data,
        degree=degree,
        two_m=two_m,
    )