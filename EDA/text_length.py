import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/Processed Data.csv'
data = pd.read_csv(file_path)

# Ensure 'statement' column exists and convert any NaN to empty strings
data['statement'] = data['statement'].astype(str)

# Calculate the length of each text entry
data['text_length'] = data['statement'].apply(len)

# Define a color palette
palette = "colorblind"  # You can change this to other options like 'muted' or 'bright'

# Plot the data with the specified color palette
sns.boxplot(data=data, x='status', y='text_length', palette=palette)
plt.title('Boxplot of Text Length by Status')
plt.show()
