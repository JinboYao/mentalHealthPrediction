import pandas as pd
import re
import string
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from googletrans import Translator, LANGUAGES
import ssl

# Setup SSL and download necessary NLTK resources
ssl._create_default_https_context = ssl._create_unverified_context
download('stopwords')
download('punkt')
download('wordnet')

# Load the dataset
file_path = r'/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/train.csv'
df = pd.read_csv(file_path)

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=['statement'], inplace=True)  # Drop rows with NaN in 'statement'

# Text Preprocessing Function
def preprocess_and_lemmatize(text):
    # Text normalization
    text = text.lower()
    text = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\n|\w*\d\w*', ' ', text).strip()
    # Tokenization and Lemmatization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return ' '.join(filtered_tokens)
# Apply text preprocessing and update the original column
df['statement'] = df['statement'].apply(preprocess_and_lemmatize)

# Data Augmentation
# def augment_text(text):
#     try:
#         blob = TextBlob(text)
#         return str(blob.translate(to='fr').translate(to='en'))
#     except Exception as e:
#         print(f"Error during translation: {e}")
#         return text
# # Apply augmentation and create a new column for it
# df['augmented_statement'] = df['statement'].apply(augment_text)

# Export cleaned and processed data with only required columns
df_final = df[['statement', 'status']]
df_final.to_csv(r'/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/train.csv', index=False)