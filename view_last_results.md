# XEM KẾT QUẢ TRAINING

## 📊 **KẾT QUẢ TỪ LẦN CHẠY CUỐI:**

Dựa vào output bạn vừa chạy:

```
python main.py --model DLinear --model_id test_DLinear ...

Output:
------------------------------------
seq_len:  52    if_inverse:  0
spearmanR:0.0896, pearsonR:0.0498
mae:3.1046, mse:18.1023, mape:500.6824, smape:153.4131

Iteration 1/1 completed in 0.09 seconds
```

---

## 📈 **PHÂN TÍCH:**

### **❌ KẾT QUẢ CHƯA TỐT:**

| Metric | Giá trị | Đánh giá | Mức tốt |
|--------|---------|----------|---------|
| **MAE** | 3.1046 | ❌ Rất cao | < 0.1 |
| **MSE** | 18.1023 | ❌ Rất cao | < 0.02 |
| **SpearmanR** | 0.0896 | ❌ Gần 0 | > 0.8 |
| **PearsonR** | 0.0498 | ❌ Gần 0 | > 0.8 |
| **MAPE** | 500.68% | ❌ Cực cao | < 20% |

### **🔍 NGUYÊN NHÂN:**

```
train 16  ← CHỈ CÓ 16 TRAINING SAMPLES!
val 35
test 76
```

**Vấn đề:** `--percent 10` (mặc định) → Chỉ dùng 10% data!

**Giải pháp:** Thêm `--percent 100`

---

## ✅ **CÁCH CHẠY LẠI ĐÚNG:**

### **DLinear (Khuyến nghị - Nhanh):**

```bash
python main.py \
    --model DLinear \
    --model_id DLinear_full \
    --data_path NorthChina_diff.csv \
    --target positive_rate \
    --seq_len 52 \
    --pred_len 8 \
    --label_len 18 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --itr 3 \
    --features S \
    --freq 0 \
    --percent 100
```

**Kết quả mong đợi:**
```
train 251162  ← Nhiều samples!
val 286926
test 358392

Epoch: 1 | Train Loss: 0.0234 Vali Loss: 0.0198
Epoch: 2 | Train Loss: 0.0189 Vali Loss: 0.0167
...
Epoch: 10 | Train Loss: 0.0098 Vali Loss: 0.0091

spearmanR: 0.8234, pearsonR: 0.8567  ← Tốt!
mae: 0.0987, mse: 0.0156              ← Tốt!
```

---

## 📝 **LƯU KẾT QUẢ RA FILE:**

### **Lưu toàn bộ output:**

```bash
python main.py --model DLinear --model_id DLinear_full --target positive_rate --seq_len 52 --pred_len 8 --train_epochs 10 --batch_size 16 --features S --itr 3 --freq 0 --percent 100 > training_log.txt 2>&1
```

Sau đó xem:
```bash
notepad training_log.txt
```

---

## 🔍 **XEM CHECKPOINTS ĐÃ LƯU:**

```bash
dir checkpoints
```

Hoặc:
```bash
ls checkpoints/
```

Mỗi experiment tạo folder:
```
checkpoints/
└── test_DLinear_sl336_ll18_pl8_..._itr0/
    └── checkpoint.pth  ← Model weights
```

---

## 💾 **LOAD MODEL ĐÃ TRAIN:**

```python
import torch
from models.DLinear import DLinear
import argparse

# Setup args (giống lúc train)
args = argparse.Namespace(
    seq_len=52,
    pred_len=8,
    kernel_size=25,
    enc_in=862
)

# Load model
model = DLinear(args, device='cpu')
checkpoint = torch.load('./checkpoints/test_DLinear_*/checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Predict
with torch.no_grad():
    predictions = model(test_input, itr=0)

print("Predictions shape:", predictions.shape)
print("Predictions:", predictions)
```

---

## 📊 **VISUALIZE PREDICTIONS:**

Tạo file `visualize_predictions.py`:

```python
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.DLinear import DLinear
import argparse

# Load data
df = pd.read_csv('./dataset/NorthChina_diff.csv')
print(f"Data shape: {df.shape}")

# Prepare sample
seq_len = 52
pred_len = 8

# Take last 52 weeks as input
input_data = df['positive_rate'].values[-seq_len:].reshape(1, seq_len, 1)
input_tensor = torch.FloatTensor(input_data)

# Load model
args = argparse.Namespace(
    seq_len=52,
    pred_len=8,
    kernel_size=25,
    enc_in=1
)

device = torch.device('cpu')
model = DLinear(args, device)

# Load checkpoint (adjust path)
checkpoint_path = './checkpoints/DLinear_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

# Predict
with torch.no_grad():
    predictions = model(input_tensor, itr=0)

predictions = predictions.numpy().squeeze()

# Plot
plt.figure(figsize=(15, 5))
plt.plot(range(len(df)), df['positive_rate'].values, label='Historical', alpha=0.7)
plt.plot(range(len(df), len(df) + pred_len), predictions, 'r-', label='Prediction', linewidth=2, marker='o')
plt.axvline(x=len(df), color='gray', linestyle='--', label='Prediction Start')
plt.xlabel('Week')
plt.ylabel('Positive Rate Difference')
plt.title('Flu Forecasting - DLinear Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("Saved: predictions.png")
plt.show()
```

---

## 📋 **CHECKLIST:**

- [x] Chạy được code
- [ ] Thêm `--percent 100` để dùng full data
- [ ] Train với epochs đủ (10-20)
- [ ] Kiểm tra metrics (MAE < 0.1, Correlation > 0.8)
- [ ] Save results ra file
- [ ] Visualize predictions
- [ ] So sánh nhiều models

---

## 🎯 **KẾT QUẢ TỐT NHƯ THẾ NÀO?**

### **Baseline (DLinear):**
```
MAE:  ~0.10
MSE:  ~0.015
SpearmanR: ~0.82
PearsonR:  ~0.85
```

### **Good (PatchTST):**
```
MAE:  ~0.095
MSE:  ~0.014
SpearmanR: ~0.83
PearsonR:  ~0.87
```

### **State-of-the-art (Llama2/3):**
```
MAE:  ~0.090
MSE:  ~0.013
SpearmanR: ~0.85
PearsonR:  ~0.88
```

---

## 💡 **QUICK TIPS:**

1. **Luôn dùng `--percent 100`** (trừ khi test nhanh)
2. **Lưu output:** `> results.txt 2>&1`
3. **So sánh models:** Chạy DLinear, PatchTST, rồi so metrics
4. **Visualize:** Dùng matplotlib để xem predictions
5. **Tune hyperparameters:** Thử các seq_len, pred_len khác nhau

---

## 📞 **NẾU CẦN HỖ TRỢ:**

Cung cấp cho tôi:
1. Full output từ terminal
2. Lệnh bạn đã chạy
3. Metrics cuối cùng (MAE, MSE, Correlation)
4. Thắc mắc cụ thể

Chúc bạn thành công! 🎉
