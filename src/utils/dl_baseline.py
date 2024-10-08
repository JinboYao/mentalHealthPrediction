import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load data
train_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/train.csv')
test_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/test.csv')
val_data = pd.read_csv('/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/val.csv')
# Ensure all text data are strings and replace NaNs
train_data['statement'] = train_data['statement'].fillna('').astype(str)
test_data['statement'] = test_data['statement'].fillna('').astype(str)
val_data['statement'] = val_data['statement'].fillna('').astype(str)

# Combine train and validation data
full_train_data = pd.concat([train_data, val_data])
full_train_data['statement'].fillna('', inplace=True)  # Replace NaNs

# Tokenization and encoding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(full_train_data['statement'])
X_train = tokenizer.texts_to_sequences(full_train_data['statement'])
X_test = tokenizer.texts_to_sequences(test_data['statement'])

# Pad sequences
max_len = max(max(len(x) for x in X_train), max(len(x) for x in X_test))  # Get max length for padding
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(full_train_data['status'])
y_test = label_encoder.transform(test_data['status'])

# Define models
def create_model(model_type):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=256, input_length=max_len))

    if model_type == 'cnn':
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
    elif model_type == 'lstm':
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
    elif model_type == 'gru':
        model.add(GRU(128))
        model.add(Dropout(0.5))
    elif model_type == 'bilstm':
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.5))
    elif model_type == 'cnn_lstm':
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(BatchNormalization())
        model.add(LSTM(64))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

models = {
    'CNN': create_model('cnn'),
    'LSTM': create_model('lstm'),
    'GRU': create_model('gru'),
    'BiLSTM': create_model('bilstm'),
    'CNN-LSTM': create_model('cnn_lstm')
}

# Dictionary to store training history
history_dict = {}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))
    history_dict[name] = history.history
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {accuracy}\n")

# Plotting training and validation accuracy
plt.figure(figsize=(10, 8))
for name, history in history_dict.items():
    plt.plot(history['accuracy'], label=f'{name} Train')
    plt.plot(history['val_accuracy'], label=f'{name} Val')

plt.title('Training and Validation Accuracy by Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
