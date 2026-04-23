import heapq
import numpy as np


class IndexScore:
    def __init__(self, score, index):
        self.score = score
        self.index = index

    # heapq is a min-heap.
    # We want "best" items first, where larger score is better.
    def __lt__(self, other):
        return self.score > other.score


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


def euclidean_distance(row_i, row_j):
    return float(np.linalg.norm(row_i - row_j))


def weighted_knn(
    dr,
    k,
    metric="euclidean",
    euclidean_as_similarity=False,
    epsilon=1e-12,
):
    """
    Build a weighted kNN graph.

    Parameters
    ----------
    dr : DenseRows
    k : int
        Number of neighbors.
    metric : str
        "cosine" or "euclidean"
    euclidean_as_similarity : bool
        Only used when metric="euclidean".
        If False, store raw Euclidean distance as the edge weight.
        If True, convert distance to similarity via 1 / (1 + distance + epsilon).
    epsilon : float
        Small constant for numerical stability.

    Notes
    -----
    - Neighbor selection:
        * cosine: larger is better
        * euclidean: smaller distance is better
    - Stored edge weights:
        * cosine: cosine similarity
        * euclidean:
            - raw distance if euclidean_as_similarity=False
            - inverse-distance similarity if euclidean_as_similarity=True
    """
    adjacency_matrix = similarities(
        dr,
        k,
        metric=metric,
        euclidean_as_similarity=euclidean_as_similarity,
        epsilon=epsilon,
    )
    return graph_creation(adjacency_matrix, dr)


def similarities(
    dr,
    k,
    metric="cosine",
    euclidean_as_similarity=False,
    epsilon=1e-12,
):
    n = dr.n
    data = np.array(dr.data, dtype=np.float32).reshape(n, dr.d)

    adjacency_matrix = np.zeros((n, n), dtype=np.float32)

    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    for i in range(n):
        row_i = data[i]
        scores = []

        for j in range(n):
            if i == j:
                continue

            row_j = data[j]

            if metric == "cosine":
                sim = cosine(row_i, row_j)
                rank_score = sim          # larger is better
                edge_weight = sim         # store cosine similarity

            else:  # metric == "euclidean"
                dist = euclidean_distance(row_i, row_j)
                rank_score = -dist        # smaller distance is better

                if euclidean_as_similarity:
                    edge_weight = 1.0 / (1.0 + dist + epsilon)
                else:
                    edge_weight = dist

            scores.append(IndexScore(rank_score, j))

        heapq.heapify(scores)

        for _ in range(min(k, len(scores))):
            item = heapq.heappop(scores)
            j = item.index

            if metric == "cosine":
                w = item.score
            else:
                # recompute edge weight from original distance
                dist = euclidean_distance(row_i, data[j])
                if euclidean_as_similarity:
                    w = 1.0 / (1.0 + dist + epsilon)
                else:
                    w = dist

            adjacency_matrix[i][j] = w
            adjacency_matrix[j][i] = w  # symmetric

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