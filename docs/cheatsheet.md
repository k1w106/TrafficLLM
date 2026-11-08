# Xem mẫu traffic (dataset) có sẵn tải về: (P1)
```
python3 -c "
import numpy as np
t = np.load('data/processed/ustc_seed42/traces.npy', allow_pickle=True)
for i in range(5):
    print(i, ':', t[i][:80], '...')
"
```
- Nằm trong file binary traces.npy (file binary)
# Xem encoded traffic (P2): file parquet do scripts/build_ustc_encoding.py tạo ra
```
python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/ustc_seed42/encoded_gpt2.parquet')
for i in range(3):
    r = df.iloc[i]
    print('nhãn gốc:', r.original_label, '| độ dài:', r.token_length)
    print(r.text[:60], '...')
    print()
"
```
 # Train model (P3)
python3 -u scripts/train_gpt2.py --config configs/ustc_gpt2_seed42.yaml 

train_loss giảm qua 3 epoch (giảm dần) -> đang học

valid_acc ~ 0,99 (paper báo 0,98–0,996)
train loss: chỉ số sai trung bình trên tất cả sample train (càng thấp thì đoán đúng càng nhiều)
train acc: tỉ lệ đoán đúng
valid loss: chỉ số đoán sai trên mẫu chưa từng nhìn thấy
valid acc: tỉ lệ đoán đúng trên tập chưa nhìn thấy
- Dựa vào kết quả của `\outputs\metrics\ustc_gpt2_seed42.json` thì thấy bài toán closed-set mô hình giải quyết khá tốt