# Checkpoint Setup Guide - Ensemble Predictor

## 🎯 Vấn Đề Hiện Tại

Các checkpoint directories tồn tại nhưng **không có file checkpoint.pth** bên trong:

```
✗ checkpoints/DLinear_full_.../checkpoint.pth - MISSING
✗ checkpoints/PatchTST_full_.../checkpoint.pth - MISSING  
✗ checkpoints/Llama2_full_.../checkpoint.pth - MISSING
```

## 🔧 Nguyên Nhân

Code early stopping trong `main.py` đã bị comment out, nên models không được save sau khi training.

## ✅ Đã Sửa

Đã enable lại checkpoint saving trong `main.py`:

```python
# OLD (commented out):
# early_stopping(vali_loss, model, path)
# if early_stopping.early_stop:
#     print("Early stopping")
#     break

# NEW (enabled):
early_stopping(vali_loss, model, path)
if early_stopping.early_stop:
    print("Early stopping")
    break
```

## 📋 Next Steps

### Option 1: Quick Test (Recommended First)

Test xem checkpoint saving có hoạt động không:

```bash
cd "d:\New folder\LLMs4Influenza"
python quick_train_test.py
```

Script này sẽ:
- Train DLinear cho 1 epoch (nhanh)
- Check xem checkpoint.pth có được tạo không
- Report kết quả

**Expected Output**:
```
✓ Training completed in 30s
✓ Checkpoint FOUND! (2.5 MB)
SUCCESS! Checkpoint saving is working correctly.
```

### Option 2: Check Current Status

Xem status hiện tại của checkpoints:

```bash
cd "d:\New folder\LLMs4Influenza"
python check_checkpoints.py
```

### Option 3: Train All Models (Sau khi test thành công)

Automated training cho tất cả models:

```bash
cd "d:\New folder\LLMs4Influenza"
python train_models_for_ensemble.py
```

Script này sẽ:
1. Train DLinear (pred_len=8)
2. Train PatchTST (pred_len=8)
3. Hỏi có train Llama2 không (pred_len=13, requires GPU)

### Option 4: Train Manual (Individual Models)

Train từng model riêng:

```bash
# DLinear
python main.py --model DLinear --model_id DLinear_full \
    --data_path NorthChina_diff.csv --pred_len 8 \
    --train_epochs 10 --itr 1 --batch_size 32

# PatchTST
python main.py --model PatchTST --model_id PatchTST_full \
    --data_path NorthChina_diff.csv --pred_len 8 \
    --train_epochs 10 --itr 1 --batch_size 32

# Llama2 (optional, requires strong GPU)
python main.py --model Llama2 --model_id Llama2_full \
    --data_path NorthChina_diff.csv --pred_len 13 \
    --train_epochs 10 --itr 1 --batch_size 16
```

## 🔍 Troubleshooting

### Issue 1: Training Runs but No Checkpoint

**Symptoms**:
```
Training completed
Directory exists but checkpoint.pth missing
```

**Possible Causes**:
1. Validation loss không giảm → Early stopping không trigger
2. Permission issues → Cannot write file
3. Disk full → Cannot save file

**Solution**:
- Check training log cho validation loss
- Ensure đủ disk space (checkpoints ~2-5 MB each)
- Check write permissions trong ./checkpoints/

### Issue 2: Training Fails with Error

**Common Errors**:

**A. Module not found**:
```
ModuleNotFoundError: No module named 'models.DLinear'
```
**Solution**: Ensure running từ LLMs4Influenza directory

**B. CUDA out of memory**:
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch_size:
```bash
--batch_size 16  # hoặc 8
```

**C. File not found**:
```
FileNotFoundError: NorthChina_diff.csv
```
**Solution**: Check dataset trong ./dataset/:
```bash
dir dataset\*.csv
```

### Issue 3: Checkpoint Directory Not Created

**Symptoms**:
```
Expected: checkpoints/DLinear_full_.../checkpoint.pth
Directory doesn't exist
```

**Solution**:
- Check if training actually ran
- Check console output for errors
- Manually create checkpoints directory:
```bash
mkdir checkpoints
```

## 📊 Verification Steps

### 1. After Quick Test
```bash
python check_checkpoints.py
```

Should show:
```
DLinear:
  itr0: ✓ Found (2.5 MB)  # test_DLinear checkpoint
```

### 2. After Full Training
```bash
python check_checkpoints.py
```

Should show:
```
DLinear:
  itr0: ✓ Found (2.5 MB)
PatchTST:
  itr0: ✓ Found (3.2 MB)
Llama2:
  itr0: ✓ Found (8.5 MB)

✓ READY: 3/3 models available for ensemble
```

### 3. Test Ensemble
```bash
python main.py --model ensemble --model_id ensemble_test \
    --data_path NorthChina_diff.csv --pred_len 8 --itr 1
```

Expected output:
```
[INFO] Loading checkpoints from ./checkpoints/
[INFO] ✓ DLinear loaded successfully
[INFO] ✓ PatchTST loaded successfully
[INFO] ✓ Llama2 loaded successfully
[INFO] Loaded 3/3 models: DLinear, PatchTST, Llama2

MSE: 0.0450, MAE: 0.1500

ENSEMBLE PERFORMANCE COMPARISON
Model        │ MSE      │ MAE      │ MSE Impr%    │ MAE Impr%   
Ensemble     │ 0.0450   │ 0.1500   │     -        │     -       
DLinear      │ 0.0520   │ 0.1700   │ +13.5%       │ +11.8%   ✓
PatchTST     │ 0.0480   │ 0.1600   │ +6.3%        │ +6.3%       
Llama2       │ 0.0500   │ 0.1650   │ +10.0%       │ +9.1%    ✓
```

## ⏱️ Estimated Training Times

**Hardware**: CPU only
- DLinear: ~15-30 minutes (10 epochs)
- PatchTST: ~20-40 minutes (10 epochs)
- Llama2: ~60-120 minutes (10 epochs) - **Needs GPU**

**Hardware**: GPU (GTX 1080 or better)
- DLinear: ~5-10 minutes
- PatchTST: ~8-15 minutes
- Llama2: ~20-40 minutes

## 📁 Expected Checkpoint Structure

After successful training:

```
./checkpoints/
├── DLinear_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/
│   └── checkpoint.pth              # ~2.5 MB
├── PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/
│   └── checkpoint.pth              # ~3.2 MB
└── Llama2_full_sl336_ll18_pl13_dm4096_nh16_el3_gl6_df512_ebtimeF_itr0/
    └── checkpoint.pth              # ~8.5 MB
```

## 🎯 Quick Start Commands

```bash
# 1. Change to project directory
cd "d:\New folder\LLMs4Influenza"

# 2. Quick test (1 epoch, ~30s)
python quick_train_test.py

# 3. If test passes, check status
python check_checkpoints.py

# 4. Train all models
python train_models_for_ensemble.py

# 5. Verify checkpoints
python check_checkpoints.py

# 6. Test ensemble
python main.py --model ensemble --model_id ensemble_test \
    --data_path NorthChina_diff.csv --pred_len 8 --itr 1
```

## 💡 Tips

1. **Start with quick test** để verify checkpoint saving works
2. **Monitor first epoch** của training để ensure no errors
3. **Check disk space** trước khi train (need ~100 MB free)
4. **Train overnight** nếu chạy CPU-only
5. **Use GPU** nếu có để speed up significantly

## 📞 Support

Nếu gặp issues:

1. Run `quick_train_test.py` và check output
2. Run `check_checkpoints.py` để verify status
3. Check console logs cho errors
4. Verify datasets exist trong ./dataset/
5. Check disk space và permissions

## ✅ Success Criteria

Ensemble ready khi:
- ✅ At least 2 models có checkpoints (preferably all 3)
- ✅ `check_checkpoints.py` shows "READY"
- ✅ Ensemble runs without "checkpoint not found" errors
- ✅ MSE/MAE improvements trong 8-15% range

---

**Status**: Checkpoint saving code đã được enabled trong main.py

**Next Action**: Run `quick_train_test.py` để verify

**Goal**: Get all 3 model checkpoints → Test ensemble → Measure improvements
