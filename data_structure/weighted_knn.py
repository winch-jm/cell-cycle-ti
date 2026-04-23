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
    weighting="adaptive_gaussian",
    euclidean_as_similarity=False,
    epsilon=1e-12,
    symmetrization="union",
    sigma=None,
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
    weighting : str or None
        Edge weighting scheme.
        Supported:
            - None: default behavior
                * cosine -> cosine similarity
                * euclidean -> raw distance or inverse-distance depending on
                  euclidean_as_similarity
            - "raw"
            - "inverse_distance"
            - "gaussian"
            - "adaptive_gaussian"
    euclidean_as_similarity : bool
        Backward-compatible option for euclidean metric when weighting=None.
    epsilon : float
        Small constant for numerical stability.
    symmetrization : str
        "union" or "mutual"
    sigma : float or None
        Global bandwidth for Gaussian kernel when weighting="gaussian".
        If None, defaults to median directed kNN distance.

    Notes
    -----
    Neighbor selection:
        * cosine: larger is better
        * euclidean: smaller distance is better

    Symmetrization:
        * union: keep edge if i->j or j->i exists
        * mutual: keep edge only if both i->j and j->i exist
    """
    adjacency_matrix = similarities(
        dr,
        k,
        metric=metric,
        weighting=weighting,
        euclidean_as_similarity=euclidean_as_similarity,
        epsilon=epsilon,
        symmetrization=symmetrization,
        sigma=sigma,
    )
    return graph_creation(adjacency_matrix, dr)


def similarities(
    dr,
    k,
    metric="cosine",
    weighting=None,
    euclidean_as_similarity=False,
    epsilon=1e-12,
    symmetrization="union",
    sigma=None,
):
    n = dr.n
    data = np.array(dr.data, dtype=np.float32).reshape(n, dr.d)

    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    if symmetrization not in {"union", "mutual"}:
        raise ValueError("symmetrization must be 'union' or 'mutual'")

    allowed_weighting = {None, "raw", "inverse_distance", "gaussian", "adaptive_gaussian"}
    if weighting not in allowed_weighting:
        raise ValueError(
            "weighting must be one of None, 'raw', 'inverse_distance', "
            "'gaussian', 'adaptive_gaussian'"
        )

    # -----------------------------
    # 1) Compute full pairwise distances / similarities
    # -----------------------------
    dist_mat = np.zeros((n, n), dtype=np.float32)
    sim_mat = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        row_i = data[i]
        for j in range(i + 1, n):
            row_j = data[j]

            dist = euclidean_distance(row_i, row_j)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

            if metric == "cosine":
                sim = cosine(row_i, row_j)
                sim_mat[i, j] = sim
                sim_mat[j, i] = sim

    # -----------------------------
    # 2) Build directed kNN graph
    # -----------------------------
    directed = np.zeros((n, n), dtype=np.float32)
    kth_dist = np.zeros(n, dtype=np.float32)

    for i in range(n):
        scores = []

        for j in range(n):
            if i == j:
                continue

            if metric == "cosine":
                rank_score = sim_mat[i, j]   # larger is better
            else:
                rank_score = -dist_mat[i, j] # smaller is better

            scores.append(IndexScore(rank_score, j))

        heapq.heapify(scores)

        nbrs = []
        for _ in range(min(k, len(scores))):
            item = heapq.heappop(scores)
            nbrs.append(item.index)

        if metric == "euclidean":
            if len(nbrs) > 0:
                kth_dist[i] = dist_mat[i, nbrs[-1]]
            else:
                kth_dist[i] = 0.0
        elif metric == "cosine":
            # needed only if adaptive_gaussian is requested with cosine, which we disallow below
            kth_dist[i] = 0.0

        for j in nbrs:
            directed[i, j] = 1.0

    # -----------------------------
    # 3) Set default weighting behavior
    # -----------------------------
    if weighting is None:
        if metric == "cosine":
            weighting = "raw"
        else:
            weighting = "inverse_distance" if euclidean_as_similarity else "raw"

    if weighting in {"gaussian", "adaptive_gaussian"} and metric != "euclidean":
        raise ValueError("Gaussian weighting requires metric='euclidean'")

    # Default global sigma if needed
    if weighting == "gaussian" and sigma is None:
        nonzero_kth = kth_dist[kth_dist > 0]
        sigma = float(np.median(nonzero_kth)) if len(nonzero_kth) > 0 else 1.0

    # -----------------------------
    # 4) Assign directed edge weights
    # -----------------------------
    weighted_directed = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j or directed[i, j] == 0:
                continue

            if metric == "cosine":
                # cosine-based weights
                if weighting == "raw":
                    w = sim_mat[i, j]
                else:
                    raise ValueError(
                        f"weighting='{weighting}' not supported with metric='cosine'"
                    )

            else:
                # euclidean-based weights
                dist = dist_mat[i, j]

                if weighting == "raw":
                    w = dist

                elif weighting == "inverse_distance":
                    w = 1.0 / (1.0 + dist + epsilon)

                elif weighting == "gaussian":
                    w = np.exp(-(dist ** 2) / (2.0 * (sigma ** 2) + epsilon))

                elif weighting == "adaptive_gaussian":
                    sig_i = kth_dist[i]
                    sig_j = kth_dist[j]

                    if sig_i <= 0 or sig_j <= 0:
                        w = 0.0
                    else:
                        w = np.exp(-(dist ** 2) / (sig_i * sig_j + epsilon))

                else:
                    raise ValueError(f"Unknown weighting: {weighting}")

            weighted_directed[i, j] = float(w)

    # -----------------------------
    # 5) Symmetrize
    # -----------------------------
    adjacency_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            wij = weighted_directed[i, j]
            wji = weighted_directed[j, i]

            if symmetrization == "union":
                if wij != 0 or wji != 0:
                    # average if both exist, otherwise keep the one present
                    if wij != 0 and wji != 0:
                        w = 0.5 * (wij + wji)
                    else:
                        w = wij if wij != 0 else wji
                    adjacency_matrix[i, j] = w
                    adjacency_matrix[j, i] = w

            elif symmetrization == "mutual":
                if wij != 0 and wji != 0:
                    w = 0.5 * (wij + wji)
                    adjacency_matrix[i, j] = w
                    adjacency_matrix[j, i] = w

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