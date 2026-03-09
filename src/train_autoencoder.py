import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Sequential
from preprocess import load_and_preprocess
from sequence_builder import build_sequences

X, y = load_and_preprocess("data/UNSW_NB15.csv")

# train only on normal traffic
X = X[y == 0]

sequences = build_sequences(X)

model = Sequential()

model.add(LSTM(64, input_shape=(sequences.shape[1], sequences.shape[2]), return_sequences=False))

model.add(RepeatVector(sequences.shape[1]))

model.add(LSTM(64, return_sequences=True))

model.add(TimeDistributed(Dense(sequences.shape[2])))

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

model.fit(
    sequences,
    sequences,
    epochs=20,
    batch_size=128,
    validation_split=0.1
)

model.save("models/lstm_autoencoder.h5")