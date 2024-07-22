import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 数据存储在一个CSV文件中
df = pd.read_csv(r'C:\Users\admin\OneDrive\桌面\UM\mentalHealthPrediction\dataset\Combined Data.csv')
print(df.head())

## 缺省值
print(df.isnull().sum())  # 查看缺失值情况
df.dropna(inplace=True)  # 删除缺失值

## 标签分布
count = df["status"].value_counts()
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
index = 0
count.plot(kind="bar", ax=axes[index], color=[plt.get_cmap('tab10')(i) for i in range(len(count))])
for container in axes[index].containers:
    axes[index].bar_label(container)
axes[index].set_yticklabels(())
axes[index].set_ylabel("")
axes[index].set_xlabel("")
axes[index].set_xticklabels(count.index, rotation=45)

index += 1
count.plot(kind="pie", ax=axes[index], autopct="%0.2f%%")
axes[index].set_ylabel("")
axes[index].set_xlabel("")


plt.suptitle('Mental Health Status Distribution')
# 调整布局
plt.tight_layout()
# 添加这行代码可以调整整体标题与子图的间距
plt.subplots_adjust(top=0.88)
# plt.show()


## 文本长度
df['text_length'] = df['statement'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data=df['text_length'],bins= 70)
plt.title('Text Length Distribution')
# plt.show()


# # 词频分析
def plot_wordcloud(emotion):
    text = ' '.join(df[df['emotion'] == emotion]['text'])
    wordcloud = WordCloud(width=800, height=400, max_font_size=100, max_words=100, background_color='white').generate(
        text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {emotion} Emotion')
    plt.show()

plot_wordcloud('positive')
plot_wordcloud('negative')
plot_wordcloud('neutral')
