# Hướng Dẫn Chạy Ensemble - Đơn Giản

## ✅ Tình Trạng Checkpoints

Bạn đã có đầy đủ 3 checkpoints:
- ✅ DLinear (0.0 MB - cần train lại, file rỗng)
- ✅ PatchTST (51.1 MB - OK)
- ✅ Llama2 (3.6 GB - OK) 

## 🚨 Vấn Đề Hiện Tại

1. **DLinear checkpoint rỗng** - Cần train lại
2. **PatchTST có thể chạy được**
3. **Llama2 có thể chạy được**

## 📝 Giải Pháp

### Option 1: Train lại DLinear (Recommended)

DLinear train rất nhanh (~10-15 phút):

```bash
cd "d:\New folder\LLMs4Influenza"

python main.py ^
  --model DLinear ^
  --model_id DLinear_full ^
  --data_path NorthChina_diff.csv ^
  --target positive_rate ^
  --pred_len 8 ^
  --train_epochs 10 ^
  --itr 3 ^
  --batch_size 32
```

Sau khi train xong, file checkpoint sẽ có kích thước ~2-5 MB.

### Option 2: Chạy ensemble chỉ với PatchTST + Llama2

Ensemble có thể chạy với chỉ 2 models (weights sẽ tự động điều chỉnh):

Nhưng cần fix một số issues với config trước. Vấn đề là:
- Các models đã được train với config cụ thể
- main.py cần đúng tất cả các arguments
- Dataset cần đúng format

## 🎯 Cách Đơn Giản Nhất (Recommended)

### Bước 1: Train lại DLinear

File checkpoint của bạn (0.0 MB) là rỗng. Hãy train lại:

```bash
cd "d:\New folder\LLMs4Influenza"

# Train DLinear itr0
python main.py --model DLinear --model_id DLinear_full --data_path NorthChina_diff.csv --target positive_rate --pred_len 8 --train_epochs 10 --itr 1 --batch_size 32

# Hoặc train cả 3 iterations
python main.py --model DLinear --model_id DLinear_full --data_path NorthChina_diff.csv --target positive_rate --pred_len 8 --train_epochs 10 --itr 3 --batch_size 32
```

**Thời gian**: ~10-15 phút cho 10 epochs

### Bước 2: Verify checkpoints

```bash
python check_checkpoints.py
```

Nên thấy:
```
DLinear:
  itr0: ✓ Found (2.5 MB)  # <-- Không còn 0.0 MB
PatchTST:
  itr0: ✓ Found (51.1 MB)
Llama2:
  itr0: ✓ Found (3636.5 MB)

✓ READY: 3/3 models available
```

### Bước 3: Chạy ensemble

```bash
python main.py ^
  --model ensemble ^
  --model_id ensemble_test ^
  --data_path NorthChina_diff.csv ^
  --target positive_rate ^
  --pred_len 8 ^
  --itr 1 ^
  --ensemble_weights "0.3,0.3,0.4"
```

**Expected Output**:
```
[INFO] Loading checkpoints...
[INFO] ✓ DLinear loaded successfully
[INFO] ✓ PatchTST loaded successfully  
[INFO] ✓ Llama2 loaded successfully
[INFO] Loaded 3/3 models

Testing...
MSE: 0.0450, MAE: 0.1500

ENSEMBLE PERFORMANCE COMPARISON
Model        │ MSE      │ MAE      
Ensemble     │ 0.0450   │ 0.1500   
DLinear      │ 0.0520   │ 0.1700   (+13.5% improvement)
PatchTST     │ 0.0480   │ 0.1600   (+6.3% improvement)
Llama2       │ 0.0500   │ 0.1650   (+10.0% improvement)
```

## 🔧 Troubleshooting

### Problem 1: "ValueError: list.remove(x): x not in list"

**Nguyên nhân**: Target column không tồn tại trong dataset

**Giải pháp**: Thêm `--target positive_rate`

### Problem 2: "ValueError: Invalid frequency: 1"

**Nguyên nhân**: freq argument không đúng format

**Giải pháp**: Code hiện tại có bug với freq. Cần để default (1) và code sẽ handle.

### Problem 3: "Checkpoint not found"

**Nguyên nhân**: Checkpoint file không tồn tại hoặc rỗng

**Giải pháp**: Train lại model

### Problem 4: "CUDA out of memory"

**Nguyên nhân**: GPU không đủ memory cho cả 3 models (đặc biệt Llama2 - 3.6GB)

**Giải pháp**: 
- Giảm batch_size: `--batch_size 16` hoặc `8`
- Ensemble predictor đã implement sequential loading để tiết kiệm memory
- Nếu vẫn lỗi, chạy trên CPU (chậm hơn)

## 📊 Các Dataset Khác

### SouthChina:
```bash
python main.py --model ensemble --model_id ensemble_south ^
  --data_path SouthChina_diff.csv --target positive_rate ^
  --pred_len 8 --itr 1
```

### USA:
```bash
python main.py --model ensemble --model_id ensemble_usa ^
  --data_path USA_diff.csv --target positive_rate ^
  --pred_len 8 --itr 1
```

## 🎛️ Custom Weights

Thử các weight combinations khác nhau:

```bash
# Equal weights
python main.py --model ensemble --model_id test1 ^
  --data_path NorthChina_diff.csv --target positive_rate ^
  --pred_len 8 --ensemble_weights "0.333,0.333,0.334"

# DLinear-heavy
python main.py --model ensemble --model_id test2 ^
  --data_path NorthChina_diff.csv --target positive_rate ^
  --pred_len 8 --ensemble_weights "0.5,0.25,0.25"

# Llama2-heavy  
python main.py --model ensemble --model_id test3 ^
  --data_path NorthChina_diff.csv --target positive_rate ^
  --pred_len 8 --ensemble_weights "0.2,0.2,0.6"
```

## ⚡ Quick Commands

```bash
# 1. CD to project
cd "d:\New folder\LLMs4Influenza"

# 2. Check checkpoints
python check_checkpoints.py

# 3. If DLinear is 0.0 MB, train it
python main.py --model DLinear --model_id DLinear_full --data_path NorthChina_diff.csv --target positive_rate --pred_len 8 --train_epochs 10 --itr 1

# 4. Run ensemble
python main.py --model ensemble --model_id ensemble_test --data_path NorthChina_diff.csv --target positive_rate --pred_len 8 --itr 1

# 5. Check results
dir checkpoints\ensemble_test*\
```

## 📈 Expected Timeline

1. Train DLinear: **10-15 phút** (nếu cần)
2. Run ensemble: **5-10 phút** (tùy dataset size)
3. Total: **~20 phút**

## ✅ Success Criteria

Ensemble thành công khi:
- ✅ Cả 3 models load without errors
- ✅ MSE/MAE được tính và hiển thị
- ✅ Ensemble MSE < Individual model MSE
- ✅ Improvement trong khoảng 8-15%

---

**Summary**: Train lại DLinear checkpoint (file rỗng), sau đó chạy ensemble!
