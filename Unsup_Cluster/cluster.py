
import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Paths
FUSED_PATH = "../Data/common/fused.pt"
# Cole's expected output path - we check for this first
LATENT_PATH = "../Latence/latent.pt" 
OUTPUT_CSV = "cluster_results.csv"
OUTPUT_PLOT = "cluster_vis.png"

def load_embeddings():
    """
    Loads embeddings, preferring the 'latent' encodings from Cole if available.
    Otherwise falls back to 'fused' embeddings.
    """
    if os.path.exists(LATENT_PATH):
        print(f"Loading latent embeddings from {LATENT_PATH}...")
        data = torch.load(LATENT_PATH)
        # Assuming similar structure: dict with 'ids' and 'data' or similar
        # If structure is unknown, we might need adjustments. 
        # For now, let's assume it follows the fused pattern or is a direct tensor.
        if isinstance(data, dict):
            ids = data.get("ids")
            embeddings = data.get("encoded") or data.get("fused") or data.get("data")
        else:
            ids = None
            embeddings = data
    elif os.path.exists(FUSED_PATH):
        print(f"Latent output not found. Falling back to fused embeddings from {FUSED_PATH}...")
        data = torch.load(FUSED_PATH)
        ids = data["ids"]
        embeddings = data["fused"]
    else:
        raise FileNotFoundError(f"Neither {LATENT_PATH} nor {FUSED_PATH} found.")
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
        
    return ids, embeddings

def main():
    # 1. Load Data
    ids, X = load_embeddings()
    print(f"Data shape: {X.shape}")
    
    # 2. Preprocessing / Dimensionality Reduction (Constraint Check)
    # If we are using fused (512-dim) and not latent (low-dim), we act as our own "latent" step 
    # just for the sake of getting good clusters, but we acknowledge this overlaps with Cole.
    if X.shape[1] > 50:
        print("High dimensionality detected. Reducing with PCA for better clustering stability...")
        pca = PCA(n_components=50) # Reduce to 50 for clustering input
        X_pca = pca.fit_transform(X)
        print(f"PCA reduced shape: {X_pca.shape}")
    else:
        X_pca = X
        
    # 3. Clustering
    # Using K-Means. 
    # K=10 is a heuristic starting point for "style groups" in viral videos 
    # (e.g. gaming, vlog, finance, skit, etc.)
    k = 10
    print(f"Clustering with K-Means (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    # 4. Save Results
    if ids is not None:
        df = pd.DataFrame({"video_id": ids, "cluster": labels})
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved cluster assignments to {OUTPUT_CSV}")
    else:
        print("Warning: No IDs found, saving only cluster labels.")
        np.savetxt("cluster_labels.txt", labels, fmt="%d")

    # 5. Visualization (2D PCA)
    print("Generating visualization...")
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title("Viral Video Style Clusters (2D PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved visualization to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
