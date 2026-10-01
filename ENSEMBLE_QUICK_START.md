# Ensemble Predictor - Quick Start Guide

## 🚀 Quick Start (5 phút)

### Bước 1: Kiểm tra Installation
```bash
cd "d:\New folder\LLMs4Influenza"
python -c "import torch; import numpy; import matplotlib; print('✓ All dependencies OK')"
```

### Bước 2: Run Tests
```bash
# Test basic functionality
python test_ensemble_basic.py

# Test metrics & visualization
python test_ensemble_metrics.py
```

### Bước 3: Run Ensemble (sau khi có checkpoints)
```bash
python main.py --model ensemble --model_id ensemble_test --data_path NorthChina_diff.csv --pred_len 8 --itr 1
```

## 📋 Prerequisites

### Checkpoints Required
Cần train các models trước:
```bash
# Train DLinear
python main.py --model DLinear --model_id DLinear_full --data_path NorthChina_diff.csv --pred_len 8 --train_epochs 10

# Train PatchTST
python main.py --model PatchTST --model_id PatchTST_full --data_path NorthChina_diff.csv --pred_len 8 --train_epochs 10

# Train Llama2 (nếu có GPU đủ mạnh)
python main.py --model Llama2 --model_id Llama2_full --data_path NorthChina_diff.csv --pred_len 13 --train_epochs 10
```

### Checkpoint Paths
Checkpoints sẽ được save tại:
```
./checkpoints/
├── DLinear_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/
│   └── checkpoint.pth
├── PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/
│   └── checkpoint.pth
└── Llama2_full_sl336_ll18_pl13_dm4096_nh16_el3_gl6_df512_ebtimeF_itr0/
    └── checkpoint.pth
```

## 💡 Usage Examples

### Example 1: Basic Ensemble (Default Weights)
```bash
python main.py \
    --model ensemble \
    --model_id ensemble_basic \
    --data_path NorthChina_diff.csv \
    --pred_len 8 \
    --itr 1
```

**Weights sử dụng**: DLinear=0.3, PatchTST=0.3, Llama2=0.4

### Example 2: Custom Weights
```bash
python main.py \
    --model ensemble \
    --model_id ensemble_custom \
    --data_path SouthChina_diff.csv \
    --pred_len 13 \
    --ensemble_weights "0.25,0.35,0.40" \
    --itr 1
```

**Weights sử dụng**: DLinear=0.25, PatchTST=0.35, Llama2=0.40

### Example 3: With Visualization
```bash
python main.py \
    --model ensemble \
    --model_id ensemble_visual \
    --data_path USA_diff.csv \
    --pred_len 8 \
    --visualize 1 \
    --itr 1
```

**Output**: Tạo file visualization trong `./checkpoints/ensemble_visual_.../ `

### Example 4: Equal Weights
```bash
python main.py \
    --model ensemble \
    --model_id ensemble_equal \
    --data_path NorthChina_diff.csv \
    --pred_len 8 \
    --ensemble_weights "0.333,0.333,0.334" \
    --itr 1
```

**Weights sử dụng**: All models equal weight

### Example 5: DLinear-Heavy
```bash
python main.py \
    --model ensemble \
    --model_id ensemble_dlinear \
    --data_path NorthChina_diff.csv \
    --pred_len 8 \
    --ensemble_weights "0.5,0.25,0.25" \
    --itr 1
```

**Use case**: Khi DLinear performs best trên validation set

## 🎯 Command-Line Arguments

### Required Arguments
- `--model ensemble` - Specify ensemble model
- `--model_id <name>` - Unique identifier cho run
- `--data_path <file>` - Dataset file name

### Ensemble-Specific Arguments
- `--ensemble_weights <w1,w2,w3>` - Weights cho [DLinear, PatchTST, Llama2]
  - Default: "0.3,0.3,0.4"
  - Must sum to ~1.0 (auto-normalized nếu không)
  - Example: "0.25,0.35,0.40"

- `--visualize <0|1>` - Enable/disable visualization
  - Default: 0 (disabled)
  - 1 = Generate prediction plots

- `--checkpoint_paths <paths>` - Custom checkpoint paths (optional)
  - Default: Auto-discover from ./checkpoints/
  - Format: "path1,path2,path3"

### Standard Arguments (từ existing pipeline)
- `--pred_len <int>` - Prediction length (8 hoặc 13)
- `--seq_len <int>` - Sequence length (default: 336)
- `--label_len <int>` - Label length (default: 18)
- `--batch_size <int>` - Batch size (default: 32)
- `--itr <int>` - Number of iterations (default: 3)

## 📊 Expected Output

### Console Output
```
[2026-10-01 09:15:00] [INFO] Creating ensemble model
[2026-10-01 09:15:00] [INFO] Initialized EnsemblePredictor with weights: {'DLinear': 0.3, 'PatchTST': 0.3, 'Llama2': 0.4}
[2026-10-01 09:15:00] [INFO] Loading checkpoints from ./checkpoints/
[2026-10-01 09:15:01] [INFO] ✓ DLinear loaded successfully
[2026-10-01 09:15:02] [INFO] ✓ PatchTST loaded successfully
[2026-10-01 09:15:03] [INFO] ✓ Llama2 loaded successfully
[2026-10-01 09:15:03] [INFO] Loaded 3/3 models: DLinear, PatchTST, Llama2
------------------------------------
MSE: 0.0450, MAE: 0.1500
Spearman R: 0.85, Pearson R: 0.87

======================================================================
  ENSEMBLE PERFORMANCE COMPARISON
======================================================================
Model        │ MSE      │ MAE      │ MSE Impr%    │ MAE Impr%   
----------------------------------------------------------------------
Ensemble     │ 0.0450   │ 0.1500   │     -        │     -       
DLinear      │ 0.0520   │ 0.1700   │ +13.5%       │ +11.8%       ✓
PatchTST     │ 0.0480   │ 0.1600   │ +6.3%        │ +6.3%       
Llama2       │ 0.0500   │ 0.1650   │ +10.0%       │ +9.1%        ✓
======================================================================

Average MSE improvement: +9.93%
Average MAE improvement: +9.07%

✓ Target improvement range (8-15%) ACHIEVED for 2/3 model(s)!
```

### Files Generated
```
./checkpoints/ensemble_<model_id>_<config>/
├── predictions.npy           # Ensemble predictions
├── visualization.png         # (if --visualize 1)
└── metrics.json              # Performance metrics
```

## 🔧 Troubleshooting

### Problem 1: "Checkpoint not found"
```
[WARNING] Checkpoint not found for DLinear itr0: ./checkpoints/...
```

**Solution**: Train the model first
```bash
python main.py --model DLinear --model_id DLinear_full --train_epochs 10
```

### Problem 2: "No models available"
```
RuntimeError: No models loaded. Call load_checkpoints() first.
```

**Solution**: Ensure at least one checkpoint exists
- Check `./checkpoints/` directory
- Train at least one model

### Problem 3: "Weights don't sum to 1.0"
```
[WARNING] Weights do not sum to 1.0, normalizing: {'DLinear': 0.3, 'PatchTST': 0.3, 'Llama2': 0.6}
```

**Solution**: Này là warning, không phải error
- Weights được auto-normalized
- Nếu muốn tránh warning, provide weights summing to 1.0

### Problem 4: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Sequential loading should help, nhưng nếu vẫn lỗi:
- Reduce `--batch_size`
- Run on CPU (slower): remove CUDA
- Close other GPU processes

### Problem 5: Import Error
```
ModuleNotFoundError: No module named 'ensemble_predictor'
```

**Solution**: Ensure đúng working directory
```bash
cd "d:\New folder\LLMs4Influenza"
python main.py ...
```

## 📈 Performance Tips

### 1. Weight Tuning
Start với default weights, sau đó tune dựa trên validation performance:
```bash
# Test different weight combinations
python main.py --model ensemble --ensemble_weights "0.2,0.3,0.5" ...
python main.py --model ensemble --ensemble_weights "0.4,0.3,0.3" ...
python main.py --model ensemble --ensemble_weights "0.333,0.333,0.334" ...
```

### 2. Dataset-Specific Weights
Different datasets có thể benefit từ different weights:
- **NorthChina**: Default (0.3, 0.3, 0.4) usually works well
- **SouthChina**: Try (0.25, 0.35, 0.40)
- **USA**: Try (0.35, 0.30, 0.35)

### 3. Prediction Length
- **Short-term (pl=8)**: All models contribute equally
- **Long-term (pl=13)**: Llama2 might need higher weight

### 4. Visualization Analysis
Enable visualization để understand model contributions:
```bash
python main.py --model ensemble --visualize 1 ...
```

Look for:
- Which models track ground truth best
- Where ensemble improves over individuals
- Outliers and failure cases

## 🧪 Testing & Validation

### Run Unit Tests
```bash
# Test basic functionality
python test_ensemble_basic.py

# Test metrics & visualization  
python test_ensemble_metrics.py

# Test weight parsing (if available)
python test_ensemble_weights.py
```

### Expected Test Results
```
============================================================
✓ ALL BASIC TESTS PASSED!
============================================================

============================================================
✓ ALL METRICS AND VISUALIZATION TESTS PASSED!
============================================================
```

### Manual Validation
After running ensemble, compare results:
```bash
# Run individual models
python main.py --model DLinear --model_id test_dlinear
python main.py --model PatchTST --model_id test_patchtst
python main.py --model Llama2 --model_id test_llama2

# Run ensemble
python main.py --model ensemble --model_id test_ensemble

# Compare MSE/MAE in console output
```

## 📚 Additional Resources

- **Full Documentation**: `ENSEMBLE_IMPLEMENTATION_SUMMARY.md`
- **Implementation Status**: `IMPLEMENTATION_STATUS.md`
- **Main Code**: `ensemble_predictor.py`
- **Tests**: `test_ensemble_*.py`

## 💬 FAQ

### Q: Tại sao default weights là (0.3, 0.3, 0.4)?
**A**: Based on research literature về model characteristics:
- DLinear: Good at trend decomposition
- PatchTST: Good at local patterns
- Llama2: Good at complex dependencies (slightly higher weight)

### Q: Có thể dùng chỉ 2 models không?
**A**: Có! System tự động adjust weights nếu một model thiếu:
```
Available: DLinear, Llama2 (PatchTST missing)
Adjusted weights: DLinear=0.429, Llama2=0.571
```

### Q: Làm sao để tìm best weights?
**A**: Grid search trên validation set:
```python
for w1 in [0.2, 0.25, 0.3, 0.35, 0.4]:
    for w2 in [0.2, 0.25, 0.3, 0.35, 0.4]:
        w3 = 1.0 - w1 - w2
        if w3 > 0:
            # Test weights (w1, w2, w3)
```

### Q: Ensemble có chậm hơn single model không?
**A**: Có, ensemble chạy 3 models nên mất ~3x thời gian. Nhưng:
- Sequential loading giúp tiết kiệm memory
- Accuracy improvement đáng giá với trade-off

### Q: Có thể train ensemble weights không?
**A**: Hiện tại sử dụng fixed weights. Có thể implement learned weights:
- Meta-learning approach
- Stacking with neural network
- Bayesian optimization

### Q: Khi nào nên dùng ensemble?
**A**: Use ensemble khi:
- ✅ Need highest accuracy possible
- ✅ Have computational resources
- ✅ Individual models already trained
- ❌ Don't use if: Need realtime inference, limited memory

## 🎓 Best Practices

1. **Always train individual models first**
   - Verify each model works independently
   - Note their individual performance

2. **Start with default weights**
   - Test ensemble với default weights
   - Only tune if needed

3. **Enable visualization initially**
   - Understand model contributions
   - Identify potential issues

4. **Monitor GPU memory**
   - Check console logs
   - Watch for OOM errors

5. **Keep checkpoints organized**
   - Follow naming convention
   - Document training configurations

---

**Need Help?** Check the full documentation files:
- `ENSEMBLE_IMPLEMENTATION_SUMMARY.md` - Complete overview
- `IMPLEMENTATION_STATUS.md` - Current status
- `ensemble_predictor.py` - Source code with docstrings

**Found a Bug?** Check existing tests first, then report with:
- Command used
- Console output
- Error message
- System info
