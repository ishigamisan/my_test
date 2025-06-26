import pandas as pd
import numpy as np
import os
from math import acos

# === 路徑設定 ===
BASE_DIR = os.path.dirname(__file__)
input_csv = os.path.join(BASE_DIR, '..', 'output', 'All_Sensor_Data.csv')
train_csv = os.path.join(BASE_DIR, '..', 'output', 'train_raw.csv')
test_csv = os.path.join(BASE_DIR, '..', 'output', 'test_raw.csv')

# === 讀取原始資料 ===
df = pd.read_csv(input_csv)

# === 計算 SVM 與 tiltAngle ===
df['SVM'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)

# 防止除以 0
denominator = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
df['tiltAngle'] = np.arccos(df['AccY'] / denominator.clip(lower=1e-8))

# === 擷取 Subject ID 來分群 ===
df['SubjectID'] = df['RecordName'].str.extract(r'S(\d{2})').astype(int)

# === 選擇需要的欄位 ===
df = df[['AccX', 'AccY', 'AccZ', 'SVM', 'tiltAngle', 'Label', 'RecordName', 'SubjectID']]

# === 分割訓練與測試集 ===
train_df = df[(df['SubjectID'] >= 6) & (df['SubjectID'] <= 31)].copy()
test_df = df[(df['SubjectID'] >= 32) & (df['SubjectID'] <= 38)].copy()

# === 儲存檔案 ===
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f'✅ 特徵計算與切割完成')
print(f'📄 訓練集儲存：{train_csv}（{len(train_df)}筆）')
print(f'📄 測試集儲存：{test_csv}（{len(test_df)}筆）')
