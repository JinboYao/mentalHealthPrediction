import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
file_path = r'C:\Users\admin\OneDrive\桌面\UM\mentalHealthPrediction\dataset\Combined Data.csv'
df = pd.read_csv(file_path)

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=['statement'], inplace=True)  # Drop rows with NaN in 'statement'

# Text Normalization
def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text
df['cleaned_statement'] = df['statement'].apply(preprocess_text)
# Remove stopwords
stop = stopwords.words('english')
df['statement'] = df['statement'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))



# Lemmatization
lemmatizer = WordNetLemmatizer()
df['statement'] = df['statement'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Visualization Post-Cleaning
vectorizer = CountVectorizer(max_features=10)  # Adjust accordingly
data_corpus = vectorizer.fit_transform(df['statement'])
sum_words = data_corpus.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
words_df = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 5))
sns.barplot(x='Word', y='Frequency', data=words_df)
plt.title('Top Words in Dataset After Cleaning')
plt.show()

# Data Augmentation


# Export cleaned and processed data
df.to_csv(r'C:\Users\admin\OneDrive\桌面\UM\mentalHealthPrediction\dataset\Processed Data.csv', index=False)
