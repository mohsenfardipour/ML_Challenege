import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset (first three columns only)
data = np.array([line.split()[:3] for line in open("D:\\RML\\challenge.txt").readlines()], dtype=float)

# Check for missing values
missing_values = np.isnan(data).sum()
print(f"Missing values: {missing_values}")  # Should be 0

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(data)

# Visualization
plt.figure(figsize=(12, 5))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title("PCA: 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")

# t-SNE Plot
plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.title("t-SNE: 2D Projection")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.tight_layout()
plt.show()
