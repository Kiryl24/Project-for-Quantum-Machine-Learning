from tensorflow.keras import Sequential
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, InputLayer
import keras
import datetime
from ann_visualizer.visualize import ann_viz;
import tensorflow as tf

# Konfiguracja TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Funkcja do wczytywania i przetwarzania dźwięków z plików WAV z przypisanymi etykietami
def load_data_from_folder(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            y, sr = librosa.load(filepath, sr=None)  # Wczytanie dźwięku

            # Ekstrakcja cech z dźwięku
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.append(np.mean(mfccs, axis=1))  # Uśrednienie cech MFCC
            labels.append(label)
    return np.array(features), np.array(labels)


# Wczytanie danych z folderów kotów i psów z przypisanymi etykietami
cats_folder = 'C:/Users/Jakub/PycharmProjects/quantummachinelearning/Frequencynator/Cats'
dogs_folder = 'C:/Users/Jakub/PycharmProjects/quantummachinelearning/Frequencynator/Dogs'
cats_features, cats_labels = load_data_from_folder(cats_folder, 'cat')
dogs_features, dogs_labels = load_data_from_folder(dogs_folder, 'dog')

# Połączenie danych kotów i psów
X = np.concatenate((cats_features, dogs_features))
y = np.concatenate((cats_labels, dogs_labels))

# Kodowanie kategorii dla etykiet
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tworzenie modelu
model = Sequential([
    InputLayer(shape=(X_train.shape[1],)),
    Dense(13, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
for layer in model.layers:
    if isinstance(layer, Dense):
        layer.input_shape = (None, X_train.shape[1])
        layer.output_shape = (None, layer.units)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
epochs = 50
batch_size = 32
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Mieszanie danych treningowych
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Trenowanie modelu z użyciem danych treningowych w losowej kolejności
    model.fit(X_train_shuffled, y_train_shuffled, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback])

# Wyświetlenie warstw modelu
for layer in model.layers:
    print(layer, layer.trainable)

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
