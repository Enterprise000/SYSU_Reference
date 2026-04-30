from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()

word2id = tf.keras.datasets.imdb.get_word_index()
id2word = dict([(value, key) for (key, value) in word2id.items()])
comments_word = lambda word: ' '.join([id2word.get(i - 3, '?') for i in word])
x_train_word = [comments_word(comment) for comment in x_train]
x_test_word = [comments_word(comment) for comment in x_test]

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train = vectorizer.fit_transform(x_train_word)
X_test = vectorizer.fit_transform(x_test_word)

logic_start = time.time()
logic_model = LogisticRegression()
logic_model.fit(X_train, y_train)
logic_predict = logic_model.predict(X_test)
logic_acc = accuracy_score(y_test, logic_predict)
logic_end = time.time()
print(f"Logistic Regression Accuracy: {logic_acc:.2f}, runtime:{logic_end - logic_start}")

svm_start = time.time()
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_acc = svm_model.score(X_test, y_test)
svm_end = time.time()
print(f"SVM Accuracy: {svm_acc}, runtime: {svm_end - svm_start}")
