from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(pd.DataFrame(X_scaled, columns=X.columns).head())

# 维度
n_components = 3
# PCA降维
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
if n_components == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.values.flatten(), cmap='viridis', edgecolor='k')
    plt.title(f"PCA-2")
    plt.xlabel("Main-feature 1")
    plt.ylabel("Main-feature 2")
    plt.colorbar(label='Target')
    plt.show()

elif n_components == 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y.values.flatten(), cmap='viridis', edgecolor='k')
    ax.set_title("PCA-3")
    ax.set_xlabel("Main-feature 1")
    ax.set_ylabel("Main-feature 2")
    ax.set_zlabel("Main-feature 3")
    plt.show()
else:
    print("维度过高，不能可视化")
