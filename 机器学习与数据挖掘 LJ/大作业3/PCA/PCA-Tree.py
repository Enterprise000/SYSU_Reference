from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# 维度
n_components = 2
# PCA降维
pca = PCA(n_components=n_components)
X_pca_train = pca.fit_transform(X_train_scaled)
X_pca_test = pca.fit_transform(X_test_scaled)

start_time = time.time()
cart_model = DecisionTreeClassifier(criterion='gini')
param_grid = {
    'max_depth': [20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
grid_search = GridSearchCV(cart_model, param_grid, cv=4, scoring='accuracy')
grid_search.fit(X_pca_train, y_train)
print("Best Parameters for CART:", grid_search.best_params_)
cart_model = grid_search.best_estimator_
y_pred_cart = cart_model.predict(X_pca_test)
end_time = time.time()
print("CART Accuracy:", accuracy_score(y_test, y_pred_cart))
print("CART Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cart))
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")
