# HƯỚNG DẪN XEM KẾT QUẢ DỰ ĐOÁN

## 1. TỔNG QUAN

Sau khi training xong, bạn có **2 loại kết quả**:

### A. Kết quả trên màn hình (đã có)
```
seq_len:  52    if_inverse:  0
spearmanR:0.0896, pearsonR:0.0498
mae:3.1046, mse:18.1023, mape:500.6824, smape:153.4131
```
→ Chỉ là **metrics tổng hợp**, không lưu predictions chi tiết

### B. Kết quả chi tiết (cần tạo mới)
- File predictions (.npy, .csv)
- Biểu đồ so sánh pred vs true
- Phân tích error

---

## 2. CÁCH LẤY KẾT QUẢ CHI TIẾT

### Bước 1: Kiểm tra checkpoint đã có

```powershell
dir checkpoints
```

Bạn sẽ thấy các thư mục như:
```
test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0
test_Llama2_sl336_ll18_pl13_dm4096_nh4_el3_gl6_df768_ebtimeF_itr0
PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0
...
```

Mỗi thư mục chứa file `checkpoint.pth` (model weights đã train)

---

### Bước 2: Lưu predictions ra file

#### Ví dụ 1: DLinear
```powershell
python save_predictions.py `
  --model DLinear `
  --model_id test_DLinear `
  --checkpoint_path "checkpoints/test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth" `
  --data_path NorthChina_diff.csv `
  --target positive_rate `
  --seq_len 52 `
  --pred_len 8 `
  --features S `
  --freq 0 `
  --percent 100
```

#### Ví dụ 2: Llama2
```powershell
python save_predictions.py `
  --model Llama2 `
  --model_id test_Llama2 `
  --checkpoint_path "checkpoints/test_Llama2_sl336_ll18_pl13_dm4096_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth" `
  --data_path NorthChina_diff.csv `
  --target positive_rate `
  --seq_len 52 `
  --pred_len 13 `
  --d_model 4096 `
  --n_heads 4 `
  --d_ff 768 `
  --llama_layers 8 `
  --features S `
  --freq 0 `
  --percent 100 `
  --pretrain 1 `
  --freeze 1
```

**LƯU Ý**: Các parameters phải **GIỐNG HỆT** khi training!

---

### Bước 3: Xem kết quả

Sau khi chạy xong, bạn sẽ có thư mục `predictions/`:

```
predictions/
├── test_DLinear_pred.npy          # Predictions array
├── test_DLinear_true.npy          # Ground truth array
├── test_DLinear_dates.npy         # Date information
├── test_DLinear_results.csv       # Chi tiết từng prediction
└── test_DLinear_summary.txt       # Tóm tắt metrics
```

#### Xem summary:
```powershell
notepad predictions\test_DLinear_summary.txt
```

Nội dung:
```
================================================================================
KẾT QUẢ DỰ ĐOÁN - test_DLinear
================================================================================

Model: DLinear
Dataset: NorthChina_diff.csv
Sequence length: 52
Prediction length: 8
Target: positive_rate

Test samples: 83
Predictions per sample: 8
Total predictions: 664

METRICS:
--------------------------------------------------------------------------------
MAE:  0.023456
MSE:  0.001234
RMSE: 0.035128

Spearman R:  0.8234 (p=1.23e-45)
Pearson R:   0.8567 (p=2.34e-56)

PREDICTION DISTRIBUTION:
--------------------------------------------------------------------------------
Min:    -0.0892
Max:     0.1234
Mean:    0.0023
Median:  0.0019
Std:     0.0456

GROUND TRUTH DISTRIBUTION:
--------------------------------------------------------------------------------
Min:    -0.0856
Max:     0.1189
Mean:    0.0021
Median:  0.0018
Std:     0.0443
```

#### Xem chi tiết CSV:
```powershell
notepad predictions\test_DLinear_results.csv
```

Hoặc mở bằng Excel để xem dạng bảng:
```csv
sample,week,date,prediction,ground_truth,error,abs_error,squared_error
0,1,2018-01-01,0.0523,0.0489,0.0034,0.0034,0.000012
0,2,2018-01-08,0.0567,0.0512,0.0055,0.0055,0.000030
0,3,2018-01-15,0.0601,0.0578,0.0023,0.0023,0.000005
...
1,1,2018-01-08,0.0489,0.0456,0.0033,0.0033,0.000011
...
```

---

### Bước 4: Tạo biểu đồ

```powershell
python visualize_predictions.py --model_id test_DLinear --num_samples 5
```

Kết quả trong thư mục `figures/`:
```
figures/
├── test_DLinear_comparison.png      # So sánh 5 samples
├── test_DLinear_scatter.png         # Scatter plot pred vs true
└── test_DLinear_error_analysis.png  # Phân tích error
```

Mở các file PNG để xem:
```powershell
start figures\test_DLinear_comparison.png
start figures\test_DLinear_scatter.png
start figures\test_DLinear_error_analysis.png
```

---

## 3. GIẢI THÍCH KẾT QUẢ

### A. Comparison Plot (`_comparison.png`)

Hiển thị **5 samples** (có thể thay đổi với `--num_samples`):
- **Đường đỏ (o)**: Predictions của model
- **Đường xanh (□)**: Ground truth (giá trị thực)
- **Đường nối đứt nét**: Sai số (error)

Mỗi sample có title hiển thị:
```
Sample 23/83 | MAE: 0.0234, MSE: 0.0012, SpearmanR: 0.823, PearsonR: 0.857
```

**Cách đọc**:
- Nếu 2 đường trùng nhau → Model dự đoán chính xác
- Nếu 2 đường cách xa → Model sai nhiều
- SpearmanR/PearsonR gần 1 → Model bắt trend tốt

---

### B. Scatter Plot (`_scatter.png`)

- **Trục X**: Ground truth (giá trị thực)
- **Trục Y**: Predictions (dự đoán của model)
- **Đường đỏ chéo**: Perfect prediction (y = x)

**Cách đọc**:
- Điểm nằm trên đường đỏ → Dự đoán chính xác
- Điểm nằm trên đường đỏ → Over-prediction (dự đoán cao hơn thực tế)
- Điểm nằm dưới đường đỏ → Under-prediction (dự đoán thấp hơn thực tế)
- Điểm càng gần đường đỏ → Model càng tốt

---

### C. Error Analysis (`_error_analysis.png`)

Gồm 4 plots:

#### 1. Error Distribution (trái trên)
- Histogram của errors (prediction - true)
- Nếu tập trung quanh 0 → Model không bias
- Nếu lệch sang trái → Thường under-predict
- Nếu lệch sang phải → Thường over-predict

#### 2. Absolute Error Distribution (phải trên)
- Histogram của |errors|
- Nếu tập trung ở giá trị thấp → Model chính xác
- Mean = MAE (Mean Absolute Error)

#### 3. Error Over Prediction Steps (trái dưới)
- Error trung bình theo từng bước (week 1, 2, 3, ...)
- Nếu error tăng dần → Model kém ở long-term forecast
- Nếu error ổn định → Model tốt cho cả short & long term

#### 4. MAE Over Prediction Steps (phải dưới)
- MAE theo từng bước
- Cho biết độ chính xác giảm như thế nào theo thời gian
- Thường MAE tăng dần (dự đoán xa khó hơn dự đoán gần)

---

## 4. SO SÁNH NHIỀU MODELS

Để so sánh DLinear vs Llama2 vs PatchTST:

### Bước 1: Lưu predictions cho tất cả models

```powershell
# DLinear
python save_predictions.py --model DLinear --model_id test_DLinear --checkpoint_path "checkpoints/test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth" --data_path NorthChina_diff.csv --target positive_rate --seq_len 52 --pred_len 8 --features S --freq 0 --percent 100

# PatchTST
python save_predictions.py --model PatchTST --model_id PatchTST_full --checkpoint_path "checkpoints/PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth" --data_path NorthChina_diff.csv --target positive_rate --seq_len 52 --pred_len 8 --features S --freq 0 --percent 100

# Llama2
python save_predictions.py --model Llama2 --model_id test_Llama2 --checkpoint_path "checkpoints/test_Llama2_sl336_ll18_pl13_dm4096_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth" --data_path NorthChina_diff.csv --target positive_rate --seq_len 52 --pred_len 13 --d_model 4096 --n_heads 4 --d_ff 768 --llama_layers 8 --features S --freq 0 --percent 100 --pretrain 1 --freeze 1
```

### Bước 2: Tạo visualizations

```powershell
python visualize_predictions.py --model_id test_DLinear
python visualize_predictions.py --model_id PatchTST_full
python visualize_predictions.py --model_id test_Llama2
```

### Bước 3: So sánh metrics

Xem summary của từng model:
```powershell
notepad predictions\test_DLinear_summary.txt
notepad predictions\PatchTST_full_summary.txt
notepad predictions\test_Llama2_summary.txt
```

So sánh:
- **MAE**: Càng thấp càng tốt (< 0.1 là tốt)
- **SpearmanR/PearsonR**: Càng gần 1 càng tốt (> 0.8 là tốt)

---

## 5. LOAD PREDICTIONS TRONG PYTHON

Nếu bạn muốn phân tích thêm trong Python:

```python
import numpy as np
import pandas as pd

# Load predictions
preds = np.load('predictions/test_DLinear_pred.npy')
trues = np.load('predictions/test_DLinear_true.npy')
dates = np.load('predictions/test_DLinear_dates.npy')

print(f"Shape: {preds.shape}")  # (83, 8, 1)
# 83 test samples, 8 weeks prediction, 1 feature

# Load CSV
df = pd.read_csv('predictions/test_DLinear_results.csv')
print(df.head())

# Tính metrics tự do
mae = np.mean(np.abs(preds - trues))
mse = np.mean((preds - trues) ** 2)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# Lấy prediction cho sample thứ 10
sample_10_pred = preds[10, :, 0]
sample_10_true = trues[10, :, 0]
print(f"Sample 10 prediction: {sample_10_pred}")
print(f"Sample 10 ground truth: {sample_10_true}")
```

---

## 6. TÓM TẮT WORKFLOW

```
1. Training (đã làm)
   python main.py ... → Tạo checkpoint.pth

2. Lưu predictions
   python save_predictions.py ... → Tạo .npy, .csv, .txt

3. Visualize
   python visualize_predictions.py ... → Tạo .png

4. Phân tích
   - Xem summary.txt: Metrics tổng hợp
   - Xem results.csv: Chi tiết từng prediction
   - Xem comparison.png: So sánh pred vs true
   - Xem scatter.png: Phân bố predictions
   - Xem error_analysis.png: Phân tích error
```

---

## 7. LƯU Ý

### A. Parameters phải match với training
```powershell
# ❌ SAI - parameters không khớp
python save_predictions.py --model DLinear --seq_len 100  # Training dùng 52
python save_predictions.py --model Llama2 --d_model 768   # Training dùng 4096

# ✅ ĐÚNG - parameters khớp với training
python save_predictions.py --model DLinear --seq_len 52
python save_predictions.py --model Llama2 --d_model 4096 --n_heads 4 --d_ff 768
```

### B. Checkpoint path
```powershell
# Kiểm tra checkpoint có tồn tại không
dir "checkpoints\test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0\checkpoint.pth"

# Nếu không có → Train lại hoặc dùng checkpoint khác
```

### C. Memory
- DLinear, PatchTST: Chạy nhanh, ít RAM
- Llama2, Llama3: Chậm, cần nhiều RAM (8GB+)
- Nếu bị out of memory → Giảm batch_size

---

## 8. TROUBLESHOOTING

### Lỗi: "Checkpoint không tồn tại"
```
❌ Checkpoint không tồn tại: checkpoints/test_DLinear_...
```
→ Kiểm tra tên checkpoint trong thư mục `checkpoints/`
→ Copy đúng tên path

### Lỗi: "Parameters không khớp"
```
RuntimeError: Error loading state_dict
```
→ Đảm bảo parameters (seq_len, d_model, n_heads, ...) giống khi training

### Lỗi: "FileNotFoundError" khi visualize
```
FileNotFoundError: Không tìm thấy file: predictions/test_DLinear_pred.npy
```
→ Chạy `save_predictions.py` trước rồi mới chạy `visualize_predictions.py`

---

## 9. VÍ DỤ OUTPUT HOÀN CHỈNH

### Terminal output:
```
================================================================================
LƯU KẾT QUẢ DỰ ĐOÁN
================================================================================

Model: DLinear
Model ID: test_DLinear
Checkpoint: checkpoints/test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth
Dataset: NorthChina_diff.csv
Output: ./predictions/
Device: cpu

Đang load test data...
✅ Test data loaded: 83 samples

Đang load model...
✅ Loaded checkpoint: checkpoints/test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth

Đang chạy predictions...
  Processed 10/83 batches
  Processed 20/83 batches
  ...

✅ Predictions shape: (83, 8, 1)
✅ Ground truth shape: (83, 8, 1)
✅ Dates shape: (83, 60, 3)

💾 Đã lưu predictions:
  - predictions\test_DLinear_pred.npy
  - predictions\test_DLinear_true.npy
  - predictions\test_DLinear_dates.npy
  - predictions\test_DLinear_results.csv
  - predictions\test_DLinear_summary.txt

✅ Hoàn thành! Kiểm tra thư mục: ./predictions/
```

Sau đó chạy visualize:
```
================================================================================
VISUALIZE PREDICTIONS
================================================================================

Model ID: test_DLinear
Predictions directory: ./predictions/
Output directory: ./figures/

Đang load predictions...
✅ Loaded predictions:
   Shape: (83, 8, 1)
   Samples: 83
   Pred length: 8

Đang tạo visualizations...

1. Comparison plots...
✅ Đã lưu: figures\test_DLinear_comparison.png

2. Scatter plot...
✅ Đã lưu: figures\test_DLinear_scatter.png

3. Error analysis...
✅ Đã lưu: figures\test_DLinear_error_analysis.png

================================================================================
✅ HOÀN THÀNH!
================================================================================

Các hình ảnh đã được lưu trong: ./figures/
- test_DLinear_comparison.png    : So sánh pred vs true
- test_DLinear_scatter.png       : Scatter plot
- test_DLinear_error_analysis.png: Phân tích error
```

**Xong! Bây giờ bạn có đầy đủ kết quả dự đoán!** 🎉
