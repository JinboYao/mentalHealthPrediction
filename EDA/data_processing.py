import pandas as pd
import re
import string
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ssl

# Setup SSL and download necessary NLTK resources
ssl._create_default_https_context = ssl._create_unverified_context
download('stopwords')
download('punkt')
download('wordnet')

# Load the initial dataset using relative path
file_path = '../dataset/Combined Data.csv'
df = pd.read_csv(file_path)

# Data Cleaning
df.drop_duplicates(inplace=True)  # 去除重复项
df.dropna(subset=['statement'], inplace=True)  # 删除含有 NaN 的行
df['statement'] = df['statement'].astype(str)  # 转换为字符串类型
df['statement'] = df['statement'].apply(lambda x: re.sub(r'\b\d+\b', '', x))  # 去除数字
df['statement'] = df['statement'].apply(lambda x: ' '.join(x.split()))  # 去除多余空格

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
    stop_words = stopwords.words('english')
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)

# Apply text preprocessing and update the original column
df['statement'] = df['statement'].apply(preprocess_and_lemmatize)

# Export intermediate cleaned and processed data with only required columns using relative path
df['statement'] = df['statement']
df.to_csv('../dataset/Processed Data.csv', index=False)
df = pd.read_csv('../dataset/Processed Data.csv')

# Calculate split points
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size

df = df.sample(frac=1).reset_index(drop=True)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

# Save datasets as CSV files using relative path
train_df.to_csv('../dataset/train.csv', index=False)
val_df.to_csv('../dataset/val.csv', index=False)
test_df.to_csv('../dataset/test.csv', index=False)

print("Data split into train, validation, and test sets and saved as CSV files.")