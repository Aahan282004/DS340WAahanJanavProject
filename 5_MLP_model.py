import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

import compat_warnings  # noqa: F401

# hyperparameters
split = 0.85
sequence_length = 10
epochs = 100
learning_rate = 0.01


def _load_close_prices(path: str = "stock_price.csv") -> np.ndarray:
    data = pd.read_csv(path)
    if "Close" not in data.columns:
        raise ValueError("stock_price.csv must contain a 'Close' column.")
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("Unable to extract numeric close prices from stock_price.csv.")
    return close.values.reshape(-1, 1)


def _build_sequences(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - sequence_length):
        window = series[i : i + sequence_length]
        target = series[i + sequence_length]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)


close_prices = _load_close_prices()
train_examples = int(len(close_prices) * split)
train_raw = close_prices[:train_examples]
test_raw = close_prices[train_examples:]

if len(train_raw) <= sequence_length or len(test_raw) <= sequence_length:
    raise ValueError("Not enough rows to build train/test windows. Collect more data.")

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw).flatten()
test_scaled = scaler.transform(test_raw).flatten()

X_train, y_train_scaled = _build_sequences(train_scaled)
X_test, y_test_scaled = _build_sequences(test_scaled)

y_train = y_train_scaled
y_test = scaler.inverse_transform(y_test_scaled)


# creating MLP model
def model_create():
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(units=50, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=30, activation="relu"),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(units=20, activation="relu"),
            tf.keras.layers.Dropout(0.01),
            tf.keras.layers.Dense(units=1, activation="linear"),
        ]
    )

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=0,
    )
    return model


# prediction on test set
def predict(model):
    predictions = model.predict(X_test, verbose=0)
    return scaler.inverse_transform(predictions)


# evaluation
def evaluate(predictions):
    mae = mean_absolute_error(y_test.ravel(), predictions.ravel())
    mape = mean_absolute_percentage_error(y_test.ravel(), predictions.ravel())
    return mae, mape, (1 - mape)


# trial runs
def run_model(n):
    total_mae = total_mape = total_acc = 0
    last_predictions = None
    for _ in range(n):
        model = model_create()
        predictions = predict(model)
        mae, mape, acc = evaluate(predictions)
        total_mae += mae
        total_mape += mape
        total_acc += acc
        last_predictions = predictions
    return (
        total_mae / n,
        total_mape / n,
        total_acc / n,
        [] if last_predictions is None else last_predictions.tolist(),
    )


mae, mape, acc, preds = run_model(1)

print(f"Mean Absolute Error = {mae}")
print(f"Mean Absolute Percentage Error = {mape}%")
print(f"Accuracy = {acc}")
