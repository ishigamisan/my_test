import pandas as pd
import numpy as np
import os
from collections import Counter

# === 路徑設定 ===
BASE_DIR = os.path.dirname(__file__)
train_csv = os.path.join(BASE_DIR, '..', 'output', 'train_raw.csv')
test_csv = os.path.join(BASE_DIR, '..', 'output', 'test_raw.csv')
np_output_dir = os.path.join(BASE_DIR, '..', 'output')

# === Sliding Window 參數 ===
WINDOW_SIZE = 25
STRIDE = 1
FEATURE_COLS = ['AccX', 'AccY', 'AccZ', 'SVM', 'tiltAngle']

# === 載入資料 ===
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

def extract_windows(df):
    X = []
    y = []
    
    # 根據 RecordName 切分每個序列
    for record_name, group in df.groupby('RecordName'):
        data = group.reset_index(drop=True)
        
        for i in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
            window = data.iloc[i:i+WINDOW_SIZE]
            features = window[FEATURE_COLS].values.astype(np.float32)
            label = Counter(window['Label']).most_common(1)[0][0]
            
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# === 擷取 train / test window 資料 ===
print("🔁 對訓練資料進行 sliding window...")
X_train, y_train = extract_windows(train_df)

print("🔁 對測試資料進行 sliding window...")
X_test, y_test = extract_windows(test_df)

# === 儲存成 .npy ===
np.save(os.path.join(np_output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(np_output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(np_output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(np_output_dir, 'y_test.npy'), y_test)

print(f"✅ Sliding window 完成！")
print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
print(f"X_test.shape  = {X_test.shape}, y_test.shape  = {y_test.shape}")
