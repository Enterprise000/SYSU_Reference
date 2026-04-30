
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from id3 import Id3Estimator
from C45 import C45Classifier
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time

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
tree = 1
if tree == 0:
    start_time = time.time()
    cart_model = DecisionTreeClassifier(criterion='gini')
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    grid_search = GridSearchCV(cart_model, param_grid, cv=4, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best Parameters for CART:", grid_search.best_params_)
    cart_model = grid_search.best_estimator_
    y_pred_cart = cart_model.predict(X_test)
    end_time = time.time()
    print("CART Accuracy:", accuracy_score(y_test, y_pred_cart))
    print("CART Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cart))
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.4f} seconds")

elif tree == 1:
    start_time = time.time()
    id3_model = Id3Estimator()
    y_train = y_train.values.flatten()
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(id3_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best Parameters for ID3:", grid_search.best_params_)
    id3_model = grid_search.best_estimator_
    y_pred_id3 = id3_model.predict(X_test)
    end_time = time.time()
    print("ID3 Accuracy:", accuracy_score(y_test, y_pred_id3))
    print("ID3 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_id3))
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.4f} seconds")

elif tree == 2:
    c45_model = C45Classifier()
    c45_model.fit(X_train, y_train)
    c45_model.summary()
    y_pred_c45 = c45_model.predict(X_test)
    print("C4.5 Accuracy:", accuracy_score(y_test, y_pred_c45))
    print("C4.5 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_c45))
