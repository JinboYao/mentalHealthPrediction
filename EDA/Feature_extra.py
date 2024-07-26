import pandas as pd
import numpy as np
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# 加载数据
file_path = '../dataset/Processed Data.csv'
data = pd.read_csv(file_path)
data['statement'] = data['statement'].astype(str)

# 使用TF-IDF提取特征
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(data['statement'])

# 加载预训练的GloVe模型
word_vectors = api.load("glove-twitter-25")  # 使用25维的Twitter GloVe模型

# 定义函数转换文本为词向量的平均值
def document_vector(doc):
    words = [word for word in word_tokenize(doc.lower()) if word in word_vectors.key_to_index]
    return np.mean(word_vectors[words], axis=0) if words else np.zeros(25)

# 应用函数获取词向量
X_glove = np.array([document_vector(text) for text in data['statement']])

# 合并TF-IDF特征和GloVe特征
X_combined = np.hstack((X_tfidf.toarray(), X_glove))

# 划分数据集
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 训练模型，这里使用逻辑回归和随机森林作为示例
lr_classifier = LogisticRegression(max_iter=1000)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练逻辑回归
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# 训练随机森林
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 使用交叉验证评估模型的泛化能力
lr_scores = cross_val_score(lr_classifier, X_combined, y, cv=5)
rf_scores = cross_val_score(rf_classifier, X_combined, y, cv=5)
print(f"Average cross-validation score for Logistic Regression: {lr_scores.mean():.2f}")
print(f"Average cross-validation score for Random Forest: {rf_scores.mean():.2f}")
