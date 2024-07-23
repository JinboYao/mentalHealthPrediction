import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes": MultinomialNB()
}

# Load datasets
train_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/train.csv')
test_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/test.csv')
val_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/val.csv')

# Clean NaN values in 'statement' across all datasets
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
