# HƯỚNG DẪN CHẠY DỰ ÁN - CHI TIẾT

## 🚀 CÁCH CHẠY LỆNH: python main.py --model DLinear --train_epochs 10

### ⚠️ **LƯU Ý QUAN TRỌNG:**

Command của bạn **CHƯA ĐỦ tham số**! Cần thêm `--model_id` (bắt buộc).

---

## ✅ **LỆNH ĐÚNG - COMPLETE:**

```bash
python main.py \
    --model DLinear \
    --model_id test_DLinear \
    --train_epochs 10
```

**Hoặc đầy đủ hơn:**

```bash
python main.py \
    --model DLinear \
    --model_id test_DLinear_north \
    --root_path ./dataset/ \
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
    --percent 100
```

---

## 📋 **CÁC THAM SỐ QUAN TRỌNG:**

### **Bắt buộc:**
```python
--model DLinear           # Tên model: DLinear, PatchTST, Llama2, Llama3, Gemma2
--model_id <tên>          # ID cho thí nghiệm (REQUIRED!)
```

### **Dữ liệu:**
```python
--root_path ./dataset/              # Thư mục chứa data
--data_path NorthChina_diff.csv     # File CSV (mặc định)
--target positive_rate              # Cột target trong CSV
--features S                        # S=single variable, M=multivariate
```

### **Time series:**
```python
--seq_len 52            # Độ dài input (52 tuần = 1 năm)
--pred_len 8            # Độ dài predict (8 tuần)
--label_len 18          # Độ dài label (overlap)
```

### **Training:**
```python
--train_epochs 10       # Số epoch
--batch_size 16         # Batch size
--learning_rate 0.0001  # Learning rate
--itr 3                 # Số lần chạy lại để tính mean/std
--percent 100           # % data dùng cho training (100=full)
```

---

## 🖥️ **HƯỚNG DẪN TỪNG BƯỚC:**

### **BƯỚC 1: Mở Terminal/Command Prompt**

**Windows:**
- Nhấn `Win + R`, gõ `cmd`, Enter
- Hoặc: Search "Command Prompt"

**Hoặc dùng PowerShell:**
- Nhấn `Win + X`, chọn "Windows PowerShell"

**Hoặc dùng Git Bash:**
- Chuột phải trong folder → "Git Bash Here"

---

### **BƯỚC 2: Di chuyển vào thư mục dự án**

```bash
cd "d:\New folder\LLMs4Influenza"
```

**Kiểm tra:**
```bash
# Windows CMD
dir

# PowerShell/Git Bash
ls
```

Bạn phải thấy các file: `main.py`, `requirements.txt`, folder `models/`, `dataset/`

---

### **BƯỚC 3: Kích hoạt môi trường ảo (nếu có)**

**Nếu dùng conda:**
```bash
conda activate llm4flu
```

**Nếu dùng venv:**
```bash
# Windows CMD
llm4flu_env\Scripts\activate

# PowerShell
llm4flu_env\Scripts\Activate.ps1

# Git Bash
source llm4flu_env/Scripts/activate
```

**Kiểm tra Python:**
```bash
python --version
# Phải là Python 3.8+
```

---

### **BƯỚC 4: Kiểm tra đã cài đặt thư viện chưa**

```bash
python -c "import torch; import pandas; import numpy; print('OK')"
```

**Nếu lỗi**, cài đặt:
```bash
pip install -r requirements.txt
```

---

### **BƯỚC 5: Chạy lệnh**

#### **Option 1: Lệnh ngắn gọn (Recommended cho test)**

```bash
python main.py --model DLinear --model_id test_DLinear --train_epochs 10
```

#### **Option 2: Lệnh đầy đủ (Recommended cho production)**

```bash
python main.py ^
    --model DLinear ^
    --model_id DLinear_north_52_8 ^
    --root_path ./dataset/ ^
    --data_path NorthChina_diff.csv ^
    --target positive_rate ^
    --seq_len 52 ^
    --pred_len 8 ^
    --label_len 18 ^
    --batch_size 16 ^
    --learning_rate 0.0001 ^
    --train_epochs 10 ^
    --itr 3 ^
    --features S ^
    --percent 100 ^
    --kernel_size 25
```

**Lưu ý Windows CMD**: Dùng `^` để xuống dòng

**PowerShell/Git Bash**: Dùng `\` để xuống dòng

---

## 📊 **KẾT QUẢ KHI CHẠY:**

### **Output mẫu:**

```
self.enc_in = 862
self.data_x = (291, 862)
train 251162
border1:  0 border2:  291
self.enc_in = 862
self.data_x = (333, 862)
val 286926
border1:  239 border2:  333
self.enc_in = 862
self.data_x = (416, 862)
test 358392
border1:  333 border2:  416
freq = 1

Epoch: 1, Steps: 7846 | Train Loss: 0.0234567 Vali Loss: 0.0198765
Epoch: 2, Steps: 7846 | Train Loss: 0.0189234 Vali Loss: 0.0167543
Epoch: 3, Steps: 7846 | Train Loss: 0.0156789 Vali Loss: 0.0145678
...
Epoch: 10, Steps: 7846 | Train Loss: 0.0098765 Vali Loss: 0.0091234

------------------------------------
seq_len:  52         if_inverse:  0
spearmanR: 0.8234, pearsonR: 0.8567
mae: 0.0987, mse: 0.0156, mape: 12.34, smape: 11.56

Iteration 1/3 completed in 123.45 seconds
CPU Memory Usage: Start 500.00 MB, End 1200.00 MB
GPU Memory Usage: Allocated Start 0.00 MB, End 0.00 MB
GPU Memory Usage: Reserved Start 0.00 MB, End 0.00 MB

... (Iteration 2, 3)

mse_mean = 0.0156, mse_std = 0.0023
mae_mean = 0.0987, mae_std = 0.0012

spearmanR_mean = 0.8234, spearmanP_mean = 0.0001
pearsonR_mean = 0.8567, pearsonP_mean = 0.0000
```

---

## 📁 **FILES ĐƯỢC TẠO:**

Sau khi chạy xong, sẽ tạo:

```
./checkpoints/
└── test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/
    └── checkpoint.pth
└── test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr1/
    └── checkpoint.pth
└── test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr2/
    └── checkpoint.pth
```

**Mỗi folder chứa:**
- `checkpoint.pth`: Model weights đã train

---

## 🔧 **CÁC LỆNH KHÁC:**

### **1. Train PatchTST (Transformer baseline):**

```bash
python main.py ^
    --model PatchTST ^
    --model_id PatchTST_north ^
    --seq_len 52 ^
    --pred_len 8 ^
    --d_model 768 ^
    --n_heads 4 ^
    --e_layers 3 ^
    --patch_size 16 ^
    --stride 8 ^
    --train_epochs 20 ^
    --batch_size 16 ^
    --itr 3
```

### **2. Train với dữ liệu khác (South China):**

```bash
python main.py ^
    --model DLinear ^
    --model_id DLinear_south ^
    --data_path SouthChina_diff.csv ^
    --train_epochs 10 ^
    --itr 3
```

### **3. Train với ILI data (số ca bệnh):**

```bash
python main.py ^
    --model DLinear ^
    --model_id DLinear_ILI_north ^
    --data_path ILI_NorthChina_diff.csv ^
    --train_epochs 10 ^
    --itr 3
```

### **4. Train với dữ liệu Mỹ:**

```bash
python main.py ^
    --model DLinear ^
    --model_id DLinear_USA ^
    --data_path USA_diff.csv ^
    --train_epochs 10 ^
    --itr 3
```

### **5. Dự đoán dài hạn (13 tuần):**

```bash
python main.py ^
    --model DLinear ^
    --model_id DLinear_long ^
    --pred_len 13 ^
    --train_epochs 10
```

---

## 🐛 **XỬ LÝ LỖI THƯỜNG GẶP:**

### **Lỗi 1: `ModuleNotFoundError: No module named 'torch'`**

**Nguyên nhân:** Chưa cài PyTorch

**Giải pháp:**
```bash
pip install torch==1.8.1 torchvision==0.9.1
pip install -r requirements.txt
```

---

### **Lỗi 2: `error: the following arguments are required: --model_id`**

**Nguyên nhân:** Thiếu tham số `--model_id`

**Giải pháp:**
```bash
python main.py --model DLinear --model_id test_run --train_epochs 10
```

---

### **Lỗi 3: `FileNotFoundError: [Errno 2] No such file or directory: './dataset/NorthChina_diff.csv'`**

**Nguyên nhân:** Không tìm thấy file data

**Giải pháp:**
```bash
# Kiểm tra file tồn tại
dir dataset\NorthChina_diff.csv

# Hoặc chỉ định đường dẫn đầy đủ
python main.py --model DLinear --model_id test --root_path "d:\New folder\LLMs4Influenza\dataset"
```

---

### **Lỗi 4: `CUDA out of memory`**

**Nguyên nhân:** GPU không đủ memory (chỉ với LLMs)

**Giải pháp:**
```bash
# Giảm batch size
python main.py --model DLinear --model_id test --batch_size 8

# Hoặc dùng CPU (DLinear rất nhanh trên CPU)
python main.py --model DLinear --model_id test --device cpu
```

---

### **Lỗi 5: `RuntimeError: DataLoader worker (pid XXXX) is killed by signal`**

**Nguyên nhân:** `num_workers` quá lớn trên Windows

**Giải pháp:**
```bash
python main.py --model DLinear --model_id test --num_workers 0
```

---

## ⚡ **QUICK START - COPY-PASTE:**

### **Test nhanh với 5 epochs:**

```bash
cd "d:\New folder\LLMs4Influenza"
python main.py --model DLinear --model_id quick_test --train_epochs 5 --itr 1
```

### **Chạy đầy đủ:**

```bash
cd "d:\New folder\LLMs4Influenza"
python main.py --model DLinear --model_id DLinear_full --data_path NorthChina_diff.csv --seq_len 52 --pred_len 8 --train_epochs 10 --batch_size 16 --itr 3 --features S --target positive_rate --percent 100
```

---

## 📊 **THEO DÕI TRAINING:**

### **Trong quá trình chạy:**

```
Epoch: 1, Steps: 7846 | Train Loss: 0.0234 Vali Loss: 0.0198
                       ↑                    ↑
                   Loss giảm dần         Validation loss
```

**Tốt:** Train loss và Vali loss đều giảm

**Overfit:** Train loss giảm nhưng Vali loss tăng

---

## 🎯 **SAU KHI CHẠY XONG:**

### **Xem kết quả:**

Kết quả được in ra terminal:
```
mse_mean = 0.0156, mse_std = 0.0023
mae_mean = 0.0987, mae_std = 0.0012
spearmanR_mean = 0.8234
pearsonR_mean = 0.8567
```

**Metrics:**
- **MSE**: Mean Squared Error (càng thấp càng tốt)
- **MAE**: Mean Absolute Error (càng thấp càng tốt)
- **SpearmanR**: Spearman correlation (càng gần 1 càng tốt)
- **PearsonR**: Pearson correlation (càng gần 1 càng tốt)

### **Load model đã train:**

```python
import torch
from models.DLinear import DLinear

# Load checkpoint
checkpoint = torch.load('./checkpoints/test_DLinear_*/checkpoint.pth')
model.load_state_dict(checkpoint)

# Predict
predictions = model(test_input)
```

---

## 🔄 **SO SÁNH NHIỀU MODELS:**

### **Script để chạy nhiều models:**

Tạo file `run_all_models.bat` (Windows):

```batch
@echo off
echo Running DLinear...
python main.py --model DLinear --model_id compare_DLinear --train_epochs 10 --itr 3

echo Running PatchTST...
python main.py --model PatchTST --model_id compare_PatchTST --train_epochs 20 --itr 3

echo Done!
pause
```

Chạy:
```bash
run_all_models.bat
```

---

## 📝 **TIPS:**

1. **Test nhanh trước:**
   ```bash
   python main.py --model DLinear --model_id test --train_epochs 2 --itr 1
   ```

2. **Tăng dần epochs:**
   - Test: 2-5 epochs
   - Development: 10-20 epochs
   - Final: 30-64 epochs

3. **Monitor loss:**
   - Loss không giảm? → Tăng learning rate
   - Loss giảm quá nhanh? → Giảm learning rate
   - Overfit? → Thêm dropout hoặc giảm model size

4. **Save logs:**
   ```bash
   python main.py --model DLinear --model_id test --train_epochs 10 > training.log 2>&1
   ```

---

## 🎓 **TÓM TẮT:**

**Lệnh tối thiểu:**
```bash
python main.py --model DLinear --model_id test_run --train_epochs 10
```

**Lệnh recommended:**
```bash
python main.py --model DLinear --model_id DLinear_north --data_path NorthChina_diff.csv --seq_len 52 --pred_len 8 --train_epochs 10 --itr 3 --batch_size 16
```

**Thời gian:**
- DLinear: ~2-5 phút/10 epochs (CPU)
- PatchTST: ~5-10 phút/20 epochs (CPU)
- Llama2/3: ~30-60 phút/10 epochs (GPU cần)

**Kết quả:**
- Checkpoints saved trong `./checkpoints/`
- Metrics printed ra terminal
- MSE, MAE, Correlation scores

Chúc bạn thành công! 🎉
