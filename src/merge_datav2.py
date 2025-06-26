import pandas as pd
import os

# 設定資料夾路徑
BASE_DIR = os.path.dirname(__file__)
sensor_data_root = os.path.join(BASE_DIR, '..', 'data', 'KFall', 'Sensor_Data')
label_data_root = os.path.join(BASE_DIR, '..', 'data', 'KFall', 'Label_Data')
output_path = os.path.join(BASE_DIR, '..', 'output', 'All_Sensor_Data.csv')

combined_df = pd.DataFrame()

for sid in range(6, 39):  # SA06 ~ SA38
    subject_folder = f'SA{sid:02d}'
    subject_path = os.path.join(sensor_data_root, subject_folder)
    label_file = os.path.join(label_data_root, f'{subject_folder}_Label.xlsx')

    if not os.path.exists(subject_path):
        print(f'[!] 缺少 Sensor 資料夾：{subject_path}')
        continue
    if not os.path.exists(label_file):
        print(f'[!] 缺少 Label 檔案：{label_file}')
        continue

    # 讀取 Label 資料
    label_df = pd.read_excel(label_file)

    # 補上空白的 Task Code 欄（因為 Excel 合併儲存格）
    label_df['Task Code (Task ID)'] = label_df['Task Code (Task ID)'].fillna(method='ffill')

    # 萃取數字 Task_ID
    label_df['Task_ID'] = label_df['Task Code (Task ID)'].astype(str).str.extract(r'\((\d+)\)').astype(int)

    for file in os.listdir(subject_path):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(subject_path, file)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f'[!] 無法讀取 {file_path} → {e}')
            continue

        if 'FrameCounter' not in df.columns:
            print(f'[!] 檔案 {file} 缺少 FrameCounter 欄位，跳過')
            continue

        df['FrameCounter'] = pd.to_numeric(df['FrameCounter'], errors='coerce')
        df['Label'] = 0  # 預設為非跌倒
        df['RecordName'] = os.path.splitext(file)[0]

        # 解析 Task ID 與 Trial ID
        try:
            base = os.path.splitext(file)[0]  # e.g., S06T20R02
            task_id = int(base[4:6])          # T20 → 20
            trial_id = int(base[7:9])         # R02 → 2
        except:
            print(f'[!] 無法解析檔名格式：{file}')
            continue

        # 精準比對對應的 Label 區間
        matched = label_df[
            (label_df['Task_ID'] == task_id) &
            (label_df['Trial ID'] == trial_id)
        ]

        for _, row in matched.iterrows():
            onset = int(row['Fall_onset_frame'])
            impact = int(row['Fall_impact_frame'])

            df.loc[
                (df['FrameCounter'] >= onset) & (df['FrameCounter'] <= impact),
                'Label'
            ] = 1

        # 合併進總表
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# 輸出結果
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined_df.to_csv(output_path, index=False)
print(f'✅ 合併與跌倒標記完成！結果已儲存：{output_path}')
