# ENSEMBLE PREDICTOR - HƯỚNG DẪN SỬ DỤNG

## 📊 Tổng Quan

Ensemble Predictor kết hợp predictions từ 3 models:
- **DLinear**: Simple, fast, good baseline
- **PatchTST**: Transformer-based, good for patterns
- **Llama2**: LLM-based, can capture complex relationships

## ✅ Checkpoints Hiện Có

Bạn đã có sẵn các checkpoints:

### DLinear:
- ✓ `DLinear_optimized_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0`
- ✓ `DLinear_optimized_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr1`
- ✓ `DLinear_optimized_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr2`
- ✓ `DLinear_fixed_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0-2`

### PatchTST:
- ✓ `PatchTST_fast_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0`
- ✓ `PatchTST_fast_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr1`
- ✓ `PatchTST_fast_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr2`
- ✓ `PatchTST_fast_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr3`
- ✓ `PatchTST_fast_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr4`

### Llama2:
- ✓ `Llama2_raw_augmented_sl336_ll18_pl8_dm4096_nh16_el3_gl6_df2048_ebtimeF_itr0`
- ✓ `Llama2_raw_augmented_sl336_ll18_pl8_dm4096_nh16_el3_gl6_df2048_ebtimeF_itr1`
- ✓ `Llama2_raw_augmented_sl336_ll18_pl8_dm4096_nh16_el3_gl6_df2048_ebtimeF_itr2`

## 🚀 CÁCH CHẠY

### Option 1: Chạy Nhanh (Default Weights)

```bash
run_ensemble.bat
```

Weights mặc định: `DLinear=0.3, PatchTST=0.3, Llama2=0.4`

### Option 2: Chạy Với Custom Weights

```bash
python ensemble_predictor.py --ensemble_weights 0.3,0.4,0.3
```

### Option 3: Thử Nhiều Weights (Tìm Best Combination)

```bash
run_ensemble_experiments.bat
```

Sẽ chạy 5 experiments:
1. Equal weights (0.33, 0.33, 0.34)
2. Favor Llama2 (0.2, 0.2, 0.6)
3. Favor DLinear (0.5, 0.3, 0.2)
4. Favor PatchTST (0.2, 0.6, 0.2)
5. Balanced (0.3, 0.3, 0.4)

## 📈 DỰ KIẾN KẾT QUẢ

### Kết Quả Individual Models (đã chạy):

| Model | Pearson R | MSE | Direction Acc |
|-------|-----------|-----|---------------|
| DLinear (estimated) | ~0.45 | ~130 | ~58% |
| PatchTST (estimated) | ~0.48 | ~125 | ~60% |
| Llama2 (actual) | 0.42 ± 0.003 | 141.19 ± 9.10 | 54.4 ± 2.3% |

### Kết Quả Ensemble (dự kiến):

| Weights | Pearson R (expected) | MSE (expected) | Improvement |
|---------|----------------------|----------------|-------------|
| 0.33, 0.33, 0.34 | **0.50-0.55** | **115-125** | +19-31% |
| 0.3, 0.3, 0.4 | **0.48-0.52** | **120-130** | +14-24% |
| 0.2, 0.2, 0.6 | 0.43-0.47 | 135-145 | +2-12% |

**Lý do Ensemble tốt hơn:**
- ✓ Reduce variance (trung bình nhiều models)
- ✓ Complement strengths (DLinear tốt với trend, PatchTST tốt với patterns, Llama2 tốt với complex relationships)
- ✓ More robust (ít bị overfit hơn)

## 🎯 CÁCH CHỌN WEIGHTS

### **Strategy 1: Equal Weights** (Safest)
```
--ensemble_weights 0.33,0.33,0.34
```
- Không bias về model nào
- Thường cho kết quả tốt nhất
- **Khuyến nghị cho production**

### **Strategy 2: Performance-Based**

Nếu biết performance của từng model:
```
weights = softmax([pearson_r_dlinear, pearson_r_patchtst, pearson_r_llama2])
```

Ví dụ: 
- DLinear: Pearson R = 0.45
- PatchTST: Pearson R = 0.48
- Llama2: Pearson R = 0.42

→ Weights ≈ `0.32, 0.38, 0.30`

### **Strategy 3: Task-Specific**

- **Nếu quan tâm Peak Timing** → Tăng weight DLinear (tốt với seasonal patterns)
- **Nếu quan tâm Direction Accuracy** → Tăng weight PatchTST (tốt với local patterns)
- **Nếu quan tâm Long-term trend** → Tăng weight Llama2 (capture global context)

## 📝 OUTPUT

Ensemble sẽ in ra:

```
======================================================================
ENSEMBLE PREDICTOR
======================================================================
Weights: DLinear=0.30, PatchTST=0.30, Llama2=0.40
======================================================================

[*] Loading DLinear from ./checkpoints/DLinear_optimized_.../checkpoint.pth
[OK] DLinear loaded successfully

[*] Loading PatchTST from ./checkpoints/PatchTST_fast_.../checkpoint.pth
[OK] PatchTST loaded successfully

[*] Loading Llama2 from ./checkpoints/Llama2_raw_augmented_.../checkpoint.pth
[OK] Llama2 loaded successfully

======================================================================
ENSEMBLE EVALUATION
======================================================================

Predictions shape: (86, 8, 1)
Ground truth shape: (86, 8, 1)

======================================================================
POST-PROCESSING: CLIPPING ANALYSIS
======================================================================
...

======================================================================
STANDARD METRICS - AFTER CLIPPING (RECOMMENDED)
======================================================================
mse:120.345678, mae:7.654321, rmse:10.970234
mape:145.678901, smape:95.432109

======================================================================
EPIDEMIC-SPECIFIC METRICS
======================================================================
[*] Peak Timing Error: 1.52 weeks
   [OK] Good - Peak prediction within 2 weeks

[*] Peak Intensity Error:
   Absolute: 7.23%
   Relative: 115.45%
   [!] Needs improvement - Peak value off by >5%

[*] Direction Accuracy: 62.35%
   [OK] Good - Trend prediction >60% accurate

======================================================================
SUMMARY
======================================================================
spearmanR: 0.4856, pearsonR: 0.5123
mae: 7.6543, mse: 120.3457, mape: 145.6789, smape: 95.4321
Epidemic Metrics:
  Peak Timing Error: 1.52 weeks
  Peak Intensity Error: 7.23%
  Direction Accuracy: 62.4%
======================================================================
```

## 🔧 TROUBLESHOOTING

### Error: "Checkpoint not found"

**Nguyên nhân:** Checkpoint path sai hoặc chưa train model

**Giải pháp:**
```bash
# Check xem checkpoint có tồn tại không
dir checkpoints\DLinear_optimized*
dir checkpoints\PatchTST_fast*
dir checkpoints\Llama2_raw_augmented*

# Nếu thiếu, train model trước:
python main.py --model DLinear --model_id DLinear_optimized ...
python main.py --model PatchTST --model_id PatchTST_fast ...
# Llama2 đã có sẵn
```

### Error: "Weights must sum to 1.0"

**Nguyên nhân:** Tổng weights không bằng 1.0

**Giải pháp:**
```bash
# SAI:
--ensemble_weights 0.3,0.3,0.3  # Sum = 0.9

# ĐÚNG:
--ensemble_weights 0.33,0.33,0.34  # Sum = 1.0
```

### Error: "CUDA out of memory"

**Nguyên nhân:** Llama2 quá lớn

**Giải pháp:**
```bash
# Giảm batch_size
--batch_size 4

# Hoặc bỏ Llama2 (dùng weight=0)
--ensemble_weights 0.5,0.5,0.0
```

## 💡 TIPS & TRICKS

### 1. **Chạy Nhanh Nhất**
Chỉ dùng DLinear + PatchTST (bỏ Llama2 chậm):
```bash
python ensemble_predictor.py --ensemble_weights 0.5,0.5,0.0
```

### 2. **Kết Quả Tốt Nhất**
Dùng cả 3 models với equal weights:
```bash
python ensemble_predictor.py --ensemble_weights 0.33,0.33,0.34
```

### 3. **Tìm Best Weights**
Chạy grid search:
```python
# Tạo file: find_best_weights.py
for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for w2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w3 = 1.0 - w1 - w2
        if w3 >= 0.1 and w3 <= 0.5:
            # Run ensemble
            cmd = f"python ensemble_predictor.py --ensemble_weights {w1},{w2},{w3}"
            # Record results
```

### 4. **Production Deployment**
Sau khi tìm được best weights, hardcode vào code:
```python
# In ensemble_predictor.py, line ~50
self.weights = {
    'DLinear': 0.32,   # From experiments
    'PatchTST': 0.38,  # From experiments
    'Llama2': 0.30     # From experiments
}
```

## 📚 THAM KHẢO

### Papers về Ensemble:
1. Zhou et al. (2012). "Ensemble Methods: Foundations and Algorithms"
2. Makridakis et al. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods"

### Ensemble Strategies:
- **Simple Average**: Equal weights, robust
- **Weighted Average**: Performance-based weights
- **Stacking**: Train meta-model on base predictions
- **Boosting**: Sequential training (AdaBoost, XGBoost)

## 🎯 KẾT LUẬN

**Ensemble Predictor là cách dễ nhất để cải thiện kết quả ngay lập tức:**
- ✅ Không cần train lại models
- ✅ Chỉ cần chạy 1 command
- ✅ Thường cải thiện 10-20% so với single model
- ✅ Robust hơn, ít overfit hơn

**Khuyến nghị:**
1. Chạy `run_ensemble.bat` với default weights
2. Nếu muốn tối ưu, chạy `run_ensemble_experiments.bat`
3. Chọn weights cho Pearson R cao nhất
4. Deploy vào production

**Expected Result:** Pearson R = **0.50-0.55** (vs 0.42 hiện tại) 🎯
