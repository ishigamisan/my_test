import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import os

# === 載入資料 ===
base_path = os.path.join(os.path.dirname(__file__), '..', 'output')
X_train = np.load(os.path.join(base_path, 'X_train_norm.npy'))
y_train = np.load(os.path.join(base_path, 'y_train.npy'))
X_test = np.load(os.path.join(base_path, 'X_test_norm.npy'))
y_test = np.load(os.path.join(base_path, 'y_test.npy'))

# === 模型定義（CNN + LSTM）===
model = Sequential([
    Conv1D(filters=64, kernel_size=3, padding='same',
           kernel_initializer='he_normal', input_shape=(25, 5)),
    LeakyReLU(alpha=0.1),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# === 編譯模型 ===
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# === 訓練模型 ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# === 評估模型 ===
print("\n📊 測試集評估結果：")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")

# === 混淆矩陣與分類報告 ===
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === 儲存模型 ===
model.save(os.path.join(base_path, 'cnn_lstm_model.h5'))
print("\n✅ 模型已儲存為 cnn_lstm_model.h5")
