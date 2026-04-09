import os
import pandas as pd
import anndata as ad
import scanpy as sc

# Load raw data
_data_path = os.path.join(os.path.dirname(__file__), '..', 'GSE142277',
                          'GSM4224315_out_gene_exon_tagged.dge_exonssf002_WT.txt')
wt_data = pd.read_csv(_data_path,
    sep='\t',
    index_col=0
)

# genes × cells → transpose to cells × genes for AnnData
adata = ad.AnnData(X=wt_data.T.astype(float))

# log1p
sc.pp.log1p(adata)

# Highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()

# PCA 
sc.tl.pca(adata, n_comps=50)

print(adata)
print("PCA shape:", adata.obsm['X_pca'].shape)
