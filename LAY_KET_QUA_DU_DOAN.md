# LẤY KẾT QUẢ DỰ ĐOÁN SAU TRAINING

## 1. KẾT QUẢ HIỆN TẠI

Hiện tại, sau khi training xong, chương trình **KHÔNG TỰ ĐỘNG LƯU** kết quả dự đoán ra file.

Kết quả dự đoán chỉ được:
- ✅ In ra màn hình (metrics: MAE, MSE, Correlation)
- ✅ Dùng để tính toán metrics
- ❌ KHÔNG được lưu ra file .npy hoặc .csv

### Kết quả bạn thấy trên màn hình:

```
seq_len:  52    if_inverse:  0
spearmanR:0.0896, pearsonR:0.0498
mae:3.1046, mse:18.1023, mape:500.6824, smape:153.4131
```

Đây là **kết quả tổng hợp** từ:
- `preds`: Giá trị dự đoán của model
- `trues`: Giá trị thực tế (ground truth)

Nhưng các giá trị `preds` và `trues` **KHÔNG được lưu ra file**.

---

## 2. CÁCH LẤY KẾT QUẢ DỰ ĐOÁN

### A. Sửa file `utils/tools.py` để lưu predictions

Tôi sẽ tạo một phiên bản mới của hàm `test()` để lưu predictions.

### B. Chạy script mới để lưu kết quả

Tôi sẽ tạo script `save_predictions.py` để:
- Load model đã train từ checkpoints
- Chạy prediction trên test set
- Lưu kết quả ra file .npy và .csv

---

## 3. GIẢI THÍCH CẤU TRÚC DỰ ĐOÁN

### Shape của dữ liệu:

```python
# Ví dụ với dataset NorthChina_diff.csv
# - Total samples: 416 weeks
# - Train: 70% = 291 samples
# - Val: 10% = 42 samples  
# - Test: 20% = 83 samples

# Với --seq_len 52 --pred_len 8:
X (input):  (batch_size, 52, 1)  # 52 tuần quá khứ
y (output): (batch_size, 8, 1)   # 8 tuần tương lai

# Kết quả predictions:
preds: (num_test_samples, 8, 1)  # Dự đoán của model
trues: (num_test_samples, 8, 1)  # Giá trị thực tế
```

### Ý nghĩa:

- Model nhìn vào **52 tuần quá khứ** để dự đoán **8 tuần tương lai**
- Mỗi test sample là một sliding window
- Predictions là giá trị `positive_rate` (tỷ lệ test dương tính flu)

---

## 4. SCRIPT ĐỂ LƯU KẾT QUẢ

Tôi sẽ tạo 2 scripts:

### A. `save_predictions.py`
- Load checkpoint đã train
- Chạy inference trên test set
- Lưu predictions ra file

### B. `visualize_predictions.py`
- Đọc predictions đã lưu
- Vẽ biểu đồ so sánh pred vs true
- Lưu hình ảnh

---

## 5. SỬ DỤNG

### Bước 1: Chạy training (đã làm rồi)
```bash
python main.py --model DLinear --model_id test_DLinear --data_path NorthChina_diff.csv --target positive_rate --seq_len 52 --pred_len 8 --train_epochs 10 --batch_size 16 --features S --itr 1 --freq 0 --percent 100
```

### Bước 2: Lưu predictions
```bash
python save_predictions.py --model DLinear --model_id test_DLinear --checkpoint_path checkpoints/test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth
```

### Bước 3: Visualize
```bash
python visualize_predictions.py --pred_file predictions/test_DLinear_pred.npy --true_file predictions/test_DLinear_true.npy
```

---

## 6. OUTPUT MẪU

Sau khi chạy xong, bạn sẽ có:

```
predictions/
├── test_DLinear_pred.npy        # Predictions (83, 8, 1)
├── test_DLinear_true.npy        # Ground truth (83, 8, 1)
├── test_DLinear_results.csv     # Metrics chi tiết
└── test_DLinear_comparison.png  # Biểu đồ so sánh
```

### File CSV sẽ có dạng:

```csv
sample,week,prediction,ground_truth,error
0,1,0.0523,0.0489,0.0034
0,2,0.0567,0.0512,0.0055
0,3,0.0601,0.0578,0.0023
...
```

### Biểu đồ sẽ hiển thị:

- Line plot: Prediction (đỏ) vs Ground Truth (xanh)
- Error bars
- Correlation coefficient
- MAE/MSE values

---

## 7. TÓM TẮT

| Thông tin | Giá trị |
|-----------|---------|
| Input shape | (batch, 52, 1) |
| Output shape | (batch, 8, 1) |
| Test samples | ~83 (20% của 416) |
| Predictions được lưu | ❌ Chưa có (cần thêm code) |
| Metrics được in | ✅ Có (MAE, MSE, Corr) |
| Visualization | ❌ Chưa có (cần thêm code) |

**Tiếp theo**: Tôi sẽ tạo các scripts để lưu và visualize predictions!
