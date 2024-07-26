import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset using relative path
file_path = '../dataset/Processed Data.csv'
data = pd.read_csv(file_path)
data['statement'] = data['statement'].astype(str)

# Label Distribution Visualization
count = data["status"].value_counts()
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
sns.barplot(x=count.index, y=count.values, palette='tab10', ax=axes[0])
axes[0].set_title('Bar Chart of Mental Health Status Distribution')
sns.despine(left=True, bottom=True)

# Pie Chart for Label Distribution
axes[1].pie(count, labels=count.index, autopct="%0.2f%%", colors=[plt.get_cmap('tab10')(i) for i in range(len(count))])
axes[1].set_title('Pie Chart of Mental Health Status Distribution')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# Text Length Analysis
data['text_length'] = data['statement'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=70)
plt.title('Text Length Distribution')
plt.show()

# Boxplot of Text Length by Status
sns.boxplot(data=data, x='status', y='text_length', palette='colorblind')
plt.title('Boxplot of Text Length by Status')
plt.show()

# Word Cloud for Mental Health Statements
data['statement'] = data['statement'].astype(str)
text = ' '.join(data['statement'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Mental Health Statements')
plt.show()

# Display Most Common Words
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['statement'])
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Print Top 10 Most Common Words
print("Top 10 most common words:")
for word, freq in sorted_word_freq[:10]:
    print(f"{word}: {freq}")
