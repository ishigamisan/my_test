# KFall 跌倒偵測專案

## 📘 專案簡介
本專案使用 CNN-LSTM 模型對 KFall 資料集進行跌倒偵測，涵蓋資料合併、標記、特徵提取、滑動視窗、正規化與模型訓練等完整流程。

---

## 🏗️ 專案架構

```
test/
│
├── data/ # 原始資料夾
│ ├── KFall/
│ ├── Label_Data/ # 標記檔（各 Subject 的跌倒區段）
│ └── Sensor_Data/ # 感測資料（CSV 檔案）
│
├── output/ # 輸出結果，如處理後資料、訓練結果等
│
├── src/ # Python 腳本程式碼
│ ├── merge_datav2.py # 合併 Sensor_Data 成單一資料表
│ ├── merge_label.py # 將 Label 標註加到資料中
│ ├── preprocess_step1_featuresplit.py # 萃取三軸特徵、SVM、Tilt angle
│ ├── preprocess_step2_window.py # 切分 sliding window
│ ├── preprocess_step3_normalize.py # RobustScaler 正規化
│ ├── train_cnn_lstm.py # CNN-LSTM 模型訓練腳本
│ └── train_cnn_lstm_bias_init.py # 帶有 bias 初始化的訓練版本
│
├── .gitignore # 忽略中間檔、模型、venv等
└── README.md # 本說明文件

```

## 2️⃣資料預處理

```
python src/preprocess_step1_featuresplit.py
python src/preprocess_step2_window.py
python src/preprocess_step3_normalize.py
```

## 3️⃣模型訓練

```
python src/train_cnn_lstm.py

# 或使用 bias 初始化版本：
python src/train_cnn_lstm_bias_init.py
```