import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Initialize models with specific settings to handle large datasets or potential convergence issues
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),  # Increased max_iter, default solver is usually fine
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes": MultinomialNB()
}

# Load datasets
train_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/train.csv')
test_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/test.csv')
val_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/val.csv')

# Ensure there are no NaN values in 'statement' column across all datasets
train_data['statement'].fillna('', inplace=True)
test_data['statement'].fillna('', inplace=True)
val_data['statement'].fillna('', inplace=True)

# Combine train and validation sets for full training process
full_train_data = pd.concat([train_data, val_data])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the training data
X_train = tfidf.fit_transform(full_train_data['statement'])
y_train = full_train_data['status']

# Transform the test data
X_test = tfidf.transform(test_data['statement'])
y_test = test_data['status']

results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = (accuracy, report)
    print(f"{name} Accuracy: {accuracy}")
    print(f"Classification Report for {name}:\n{report}\n")

#
# result：
# Logistic Regression Accuracy: 0.7346745112924654
# Classification Report for Logistic Regression:
#                       precision    recall  f1-score   support
#
#              Anxiety       0.75      0.72      0.74       406
#              Bipolar       0.89      0.66      0.76       287
#           Depression       0.67      0.71      0.69      1542
#               Normal       0.83      0.94      0.88      1610
# Personality disorder       0.81      0.45      0.58       119
#               Stress       0.64      0.41      0.50       290
#             Suicidal       0.63      0.60      0.62      1015
#
#             accuracy                           0.73      5269
#            macro avg       0.75      0.64      0.68      5269
#         weighted avg       0.73      0.73      0.73      5269
#
#
# SVM Accuracy: 0.7475801859935471
# Classification Report for SVM:
#                       precision    recall  f1-score   support
#
#              Anxiety       0.78      0.77      0.77       406
#              Bipolar       0.93      0.69      0.79       287
#           Depression       0.67      0.73      0.70      1542
#               Normal       0.85      0.94      0.89      1610
# Personality disorder       0.89      0.45      0.60       119
#               Stress       0.70      0.42      0.52       290
#             Suicidal       0.65      0.60      0.62      1015
#
#             accuracy                           0.75      5269
#            macro avg       0.78      0.66      0.70      5269
#         weighted avg       0.75      0.75      0.74      5269
#
#
# Random Forest Accuracy: 0.7270829379388878
# Classification Report for Random Forest:
#                       precision    recall  f1-score   support
#
#              Anxiety       0.82      0.73      0.77       406
#              Bipolar       0.91      0.66      0.76       287
#           Depression       0.61      0.77      0.68      1542
#               Normal       0.84      0.91      0.88      1610
# Personality disorder       0.98      0.41      0.58       119
#               Stress       0.87      0.36      0.50       290
#             Suicidal       0.64      0.52      0.57      1015
#
#             accuracy                           0.73      5269
#            macro avg       0.81      0.62      0.68      5269
#         weighted avg       0.74      0.73      0.72      5269
#
#
# Naive Bayes Accuracy: 0.6418675270449801
# Classification Report for Naive Bayes:
#                       precision    recall  f1-score   support
#
#              Anxiety       0.79      0.61      0.68       406
#              Bipolar       0.93      0.40      0.56       287
#           Depression       0.51      0.78      0.62      1542
#               Normal       0.77      0.80      0.78      1610
# Personality disorder       1.00      0.20      0.34       119
#               Stress       0.74      0.11      0.19       290
#             Suicidal       0.64      0.46      0.53      1015
#
#             accuracy                           0.64      5269
#            macro avg       0.77      0.48      0.53      5269
#         weighted avg       0.68      0.64      0.63      5269
#
