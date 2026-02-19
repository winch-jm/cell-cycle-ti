# Cell Cycle Trajectory Inference
Fundamentals of Bioinformatics (02-604) Project

Group Members:
- [Anoushka Samuel](https://github.com/anoushkasamuel)
- [Sophia Turecki](https://github.com/sophiat1101)
- [Ajay Prabhakar](https://github.com/ajayprab20)
- [Jeff Winchell](https://github.com/winch-jm)

#### Problem / Goal:
Single-cell RNA sequencing (scRNA-seq) captures snapshots of cells undergoing continuous biological processes such as differentiation and cell cycle progression. The goal of our project is to utilize cross-sectional scRNA-seq data to reconstruct cell cycle trajectories from gene expression patterns. Because gene expression data is high-dimensional and noisy, recovering the underlying cyclic structure requires methods that preserve cell-cell relationships while still capturing the global manifold geometry. In this project, we will apply Laplacian Eigenmaps to build graph-based embedding of cells and recover the cyclic manifold corresponding to cell cycle progression. From the embedding, we will infer a circular, continuous pseudotemporal ordering and evaluate its biological validity and robustness. If time allows, we would like to explore trajectory inference for cell differentiation and/or compare Laplacian Eigenmaps to Diffusion Maps or RNA velocity-based methods.

#### Relevance:  
The cell cycle is a fundamental biological process that governs cell proliferation, DNA replication and cell division. In single cell transcriptomic data, cell state can create variation among samples and influence downstream analyses, such as clustering and differentiation inferences. Reconstructing cell cycle trajectories can help identify proliferative subpopulations, which may provide insight into abnormal proliferation of cancer and stem cells, and separate cell cycle effects from other insightful biological signals.

#### Approach:
To address the goal at hand, we propose the following workflow:
- Build cell-cell similarity graph
  - Compute pairwise distance between cells (Euclidean distance) → Construct a k-nearest neighbor graph → Define edge weights
- Construct the graph Laplacian, $L$
- Compute Laplacian Eigenmaps
  - Solve an eigenvalue problem, $Lv = \lambda v$ → Use eigenvectors corresponding to smallest nonzero eigenvalues as manifold embeddings
- Define cyclic pseudotime
  - Identify circular manifold in embedding → Convert 2D coordinates of embedding into angular coordinates →Utilize angular coordinates as continuous cell cycle pseudotime
- Validate trajectory
  - Calculate correlation with between inferred pseudotime and known cell cycle phase scores → Verify phase transition occurs in expected biological order → Test robustness to subsampling and stability across k values in kNN graph
- Testing with other methods/datasets (exploratory)
  - Method: diffusion maps → Data: neural cell differentiation

[Dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142277)  (supports RNA Velocity if needed)

Paper resources:
- [Laplacian Eigenmaps](https://papers.nips.cc/paper_files/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html)
- [Diffusion Maps](https://www.sciencedirect.com/science/article/pii/S1063520306000546)
- [Lior Pachter's criticism of trajectory inference and RNA Velocity](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010492)
