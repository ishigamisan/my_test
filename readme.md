# KFall 跌倒偵測專案
## 📘 專案簡介
本專案使用 CNN-LSTM 模型對 KFall 資料集進行跌倒偵測，涵蓋資料合併、標記、特徵提取、滑動視窗、正規化與模型訓練等完整流程。

## 專案架構

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

## 資料預處理
```
python src/preprocess_step1_featuresplit.py
python src/preprocess_step2_window.py
python src/preprocess_step3_normalize.py
```

## 模型訓練
```
python src/train_cnn_lstm.py

# 或使用 bias 初始化版本：
python src/train_cnn_lstm_bias_init.py
```

## train_cnn_lstm.py
```
📊 測試集評估結果：
Loss = 0.0575, Accuracy = 0.9811
21604/21604 [==============================] - 33s 2ms/step

📋 Classification Report:
              precision    recall  f1-score   support

           0     0.9870    0.9932    0.9901    658956
           1     0.8417    0.7334    0.7838     32344

    accuracy                         0.9811    691300
   macro avg     0.9144    0.8633    0.8870    691300
weighted avg     0.9802    0.9811    0.9805    691300


🧩 Confusion Matrix:
[[654495   4461]
 [  8622  23722]]
```

## train_cnn_lstm_bias_init.py
```
📊 測試集評估結果：
Loss = 0.0584, Accuracy = 0.9808
21604/21604 [==============================] - 28s 1ms/step

🔍 Threshold Sweep:
Threshold=0.1 | Recall=0.8724 | Precision=0.6059 | F1-score=0.7151
Threshold=0.2 | Recall=0.8252 | Precision=0.7130 | F1-score=0.7650
Threshold=0.3 | Recall=0.7859 | Precision=0.7751 | F1-score=0.7805
Threshold=0.4 | Recall=0.7505 | Precision=0.8169 | F1-score=0.7823
Threshold=0.5 | Recall=0.7180 | Precision=0.8491 | F1-score=0.7781
Threshold=0.6 | Recall=0.6861 | Precision=0.8749 | F1-score=0.7691
Threshold=0.7 | Recall=0.6536 | Precision=0.8955 | F1-score=0.7557
Threshold=0.8 | Recall=0.6140 | Precision=0.9166 | F1-score=0.7354

📋 Classification Report (Threshold=0.4):
              precision    recall  f1-score   support

           0     0.9878    0.9917    0.9898    658956
           1     0.8169    0.7505    0.7823     32344

    accuracy                         0.9805    691300
   macro avg     0.9023    0.8711    0.8860    691300
weighted avg     0.9798    0.9805    0.9801    691300


🧩 Confusion Matrix:
[[653515   5441]
 [  8070  24274]]
```