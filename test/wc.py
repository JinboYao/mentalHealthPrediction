import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

file_path = r'/dataset/Processed Data.csv'
data = pd.read_csv(file_path)

# Generate a word cloud for the entire dataset
# Ensure all entries in 'statement' are strings
data['statement'] = data['statement'].astype(str)
text = ' '.join(data['statement'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Mental Health Statements')
plt.show()

# Display most common words
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['statement'])
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Print top 10 most common words
print("Top 10 most common words:")
for word, freq in sorted_word_freq[:10]:
    print(f"{word}: {freq}")