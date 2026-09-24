# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN LLMs4Influenza

## 📋 TỔNG QUAN DỰ ÁN

Dự án này sử dụng các mô hình Large Language Models (Llama2, Llama3, Gemma2) để dự báo cúm (influenza forecasting). Dự án fine-tune các LLMs pre-trained để dự đoán xu hướng dịch cúm dựa trên dữ liệu time series.

### Kiến trúc dự án:
```
LLMs4Influenza/
├── dataset/               # Dữ liệu cúm (Bắc/Nam Trung Quốc, Mỹ)
├── data_provider/         # Xử lý và load dữ liệu
├── models/               # Các model: Llama2, Llama3, Gemma2, PatchTST, DLinear, GPT4TS
├── scripts/              # Scripts chạy thử nghiệm
├── utils/                # Công cụ hỗ trợ (metrics, tools)
├── main.py               # File chính để train/test
└── requirements.txt      # Các thư viện cần thiết
```

---

## 🔧 YÊU CẦU HỆ THỐNG

### Phần cứng:
- **GPU**: Tesla T4 hoặc tương đương (CUDA-enabled)
- **RAM**: Tối thiểu 16GB
- **Disk**: ~10GB cho models và dataset

### Phần mềm:
- **Python**: >= 3.8
- **PyTorch**: 1.8.1
- **CUDA**: Compatible với PyTorch 1.8.1
- **OS**: Linux/Windows với WSL (scripts dùng bash)

---

## 📦 BƯỚC 1: CÀI ĐẶT MÔI TRƯỜNG

### 1.1. Tạo môi trường ảo (khuyến nghị):
```bash
# Dùng conda
conda create -n llm4flu python=3.8
conda activate llm4flu

# Hoặc dùng venv
python -m venv llm4flu_env
source llm4flu_env/bin/activate  # Linux/Mac
# llm4flu_env\Scripts\activate  # Windows
```

### 1.2. Cài đặt PyTorch với CUDA:
```bash
# PyTorch 1.8.1 với CUDA 11.1 (ví dụ)
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Kiểm tra CUDA:
python -c "import torch; print(torch.cuda.is_available())"
```

### 1.3. Cài đặt các thư viện cần thiết:
```bash
cd "d:\New folder\LLMs4Influenza"
pip install -r requirements.txt
```

### 1.4. Cài đặt thêm (nếu cần):
```bash
# Transformers cho LLMs
pip install transformers==4.30.1

# Bitsandbytes cho quantization (8-bit)
pip install bitsandbytes

# Accelerate cho training hiệu quả
pip install accelerate
```

---

## 📊 BƯỚC 2: CHUẨN BỊ DỮ LIỆU

### 2.1. Dữ liệu có sẵn trong thư mục `dataset/`:
- `Flu_in_NorthChina.csv` - Dữ liệu cúm Bắc Trung Quốc
- `Flu_in_SouthChina.csv` - Dữ liệu cúm Nam Trung Quốc
- `Flu_in_USA.csv` - Dữ liệu cúm Mỹ
- `ILI_in_NorthChina.csv` - Dữ liệu ILI (Influenza-Like Illness) Bắc TQ
- `ILI_in_SouthChina.csv` - Dữ liệu ILI Nam TQ
- Các file `*_diff.csv` - Dữ liệu sau khi differencing

### 2.2. Tải thêm dữ liệu (nếu cần):
Theo README, dữ liệu gốc từ: https://www.nature.com/articles/s41467-021-23440-1

### 2.3. Cấu trúc dữ liệu CSV:
```
date,positive_rate (hoặc các features khác)
2020-01-01,0.123
2020-01-08,0.145
...
```

---

## 🤖 BƯỚC 3: CHUẨN BỊ PRETRAINED MODELS

### 3.1. Tải các pre-trained models:

**Llama2:**
```bash
# Tạo thư mục
mkdir -p /data_disk/lichx/Model_from_HF/LLAMA2

# Download từ HuggingFace (cần đăng ký với Meta)
# https://huggingface.co/meta-llama/Llama-2-7b-hf
```

**Llama3:**
```bash
mkdir -p /data_disk/lichx/Model_from_HF/LLAMA3
# https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

**Gemma2:**
```bash
mkdir -p /data_disk/lichx/Kaggle  
# https://huggingface.co/google/gemma-2-9b
```

### 3.2. Cập nhật đường dẫn models:
Sửa đường dẫn trong các file model nếu bạn lưu ở nơi khác:
- `models/Llama2.py` (line 31): `/data_disk/lichx/Model_from_HF/LLAMA2`
- `models/Llama3.py` (line 26): `/data_disk/lichx/Model_from_HF/LLAMA3`
- `models/Gemma2.py` (line 26): `/data_disk/lichx/Kaggle`

---

## 🚀 BƯỚC 4: CHẠY DỰ ÁN

### 4.1. Chạy với PatchTST (baseline, không dùng LLM):
```bash
cd "d:\New folder\LLMs4Influenza"

python main.py \
    --root_path ./dataset/ \
    --data_path NorthChina_diff.csv \
    --model_id test_PatchTST \
    --data custom \
    --seq_len 52 \
    --label_len 18 \
    --pred_len 8 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 2 \
    --percent 100 \
    --gpt_layers 6 \
    --itr 3 \
    --model PatchTST \
    --features S \
    --target positive_rate
```

### 4.2. Chạy với Llama2:
```bash
python main.py \
    --root_path ./dataset/ \
    --data_path NorthChina_diff.csv \
    --model_id test_Llama2 \
    --data custom \
    --seq_len 52 \
    --label_len 18 \
    --pred_len 13 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --d_model 4096 \
    --n_heads 4 \
    --d_ff 768 \
    --llama_layers 8 \
    --itr 3 \
    --model Llama2 \
    --pretrain 1 \
    --freeze 1 \
    --features S \
    --target positive_rate
```

### 4.3. Chạy script có sẵn (Linux/WSL):
```bash
# Chuyển đổi line endings nếu cần (Windows -> Unix)
dos2unix scripts/flucdc_north.sh

# Chạy script
bash scripts/flucdc_north.sh
```

---

## 📝 BƯỚC 5: CÁC THAM SỐ QUAN TRỌNG

### Tham số dữ liệu:
- `--root_path`: Thư mục chứa dataset
- `--data_path`: Tên file CSV
- `--seq_len`: Độ dài chuỗi đầu vào (52 tuần)
- `--pred_len`: Độ dài dự đoán (8-13 tuần)
- `--label_len`: Độ dài label
- `--features`: 'S' (single variable), 'M' (multivariate)
- `--target`: Tên cột target (vd: 'positive_rate')

### Tham số training:
- `--batch_size`: 8-32 (tùy GPU memory)
- `--learning_rate`: 0.0001-0.001
- `--train_epochs`: Số epoch (10-64)
- `--itr`: Số lần lặp lại thử nghiệm (3)
- `--percent`: % dữ liệu train dùng (100)

### Tham số model:
- `--model`: PatchTST, DLinear, Llama2, Llama3, Gemma2, GPT4TS
- `--d_model`: Dimension của model (768, 4096)
- `--llama_layers`: Số layers Llama dùng (8-32)
- `--pretrain`: 1 (dùng pretrained), 0 (train from scratch)
- `--freeze`: 1 (freeze LLM weights), 0 (fine-tune)

---

## 📊 BƯỚC 6: XEM KẾT QUẢ

### 6.1. Kết quả trong quá trình chạy:
```
Epoch: 1, Steps: 100 | Train Loss: 0.0234 Vali Loss: 0.0189
...
mse_mean = 0.0156, mse_std = 0.0023
mae_mean = 0.0987, mae_std = 0.0012
spearmanR_mean = 0.8234, spearmanP_mean = 0.0001
pearsonR_mean = 0.8567, pearsonP_mean = 0.0000
```

### 6.2. Checkpoints được lưu tại:
```
./checkpoints/{model_id}_sl{seq_len}_ll{label_len}_pl{pred_len}_..._{itr}/
```

### 6.3. Metrics đánh giá:
- **MSE** (Mean Squared Error): Sai số bình phương trung bình
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **Spearman/Pearson Correlation**: Tương quan giữa dự đoán và thực tế

---

## 🐛 KHẮC PHỤC SỰ CỐ

### Lỗi 1: CUDA out of memory
```
Solution:
- Giảm batch_size (16 -> 8 -> 4)
- Giảm seq_len
- Giảm llama_layers
- Dùng quantization (8-bit): Đã có trong code
```

### Lỗi 2: Model không load được
```
Solution:
- Kiểm tra đường dẫn model trong models/*.py
- Kiểm tra quyền truy cập HuggingFace token
- Đổi pretrain=0 để không dùng pretrained
```

### Lỗi 3: Thiếu thư viện
```
Solution:
pip install <tên_thư_viện>
```

### Lỗi 4: Scripts không chạy (Windows)
```
Solution:
- Cài WSL (Windows Subsystem for Linux)
- Hoặc chuyển đổi script thành command Python trực tiếp
```

---

## 📈 TÙY CHỈNH CHO DỮ LIỆU RIÊNG

### Để dùng dữ liệu riêng của bạn:

1. **Chuẩn bị CSV với format:**
```csv
date,your_target_column
2020-01-01,100
2020-01-08,105
...
```

2. **Chạy với tham số:**
```bash
python main.py \
    --root_path ./your_data_folder/ \
    --data_path your_data.csv \
    --target your_target_column \
    --model_id your_experiment_name \
    ...
```

3. **Điều chỉnh split data** (trong `data_loader.py` line 360-363):
```python
num_train = int(len(df_raw) * 0.7)  # 70% train
num_test = int(len(df_raw) * 0.2)   # 20% test
num_vali = len(df_raw) - num_train - num_test  # 10% validation
```

---

## 📚 TÀI LIỆU THAM KHẢO

- Paper gốc: https://www.nature.com/articles/s41467-021-23440-1
- Time-Series-Library: https://github.com/thuml/Time-Series-Library
- One-Fits-All: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
- HuggingFace Transformers: https://huggingface.co/docs/transformers

---

## 💡 GHI CHÚ

1. **Đối với người dùng Windows**: Khuyến nghị cài WSL để chạy bash scripts
2. **GPU Memory**: Model Llama2/3 cần ~8-16GB VRAM
3. **Training time**: 1 epoch ~10-30 phút tùy dataset và GPU
4. **Quantization**: Code đã tích hợp 8-bit quantization để tiết kiệm memory

---

## 🎯 QUICKSTART (Chạy nhanh với PatchTST)

```bash
# 1. Cài môi trường
conda create -n llm4flu python=3.8
conda activate llm4flu

# 2. Cài thư viện
cd "d:\New folder\LLMs4Influenza"
pip install torch==1.8.1 torchvision==0.9.1
pip install -r requirements.txt

# 3. Chạy thử với PatchTST (không cần download LLM)
python main.py \
    --root_path ./dataset/ \
    --data_path NorthChina_diff.csv \
    --model_id quickstart_test \
    --data custom \
    --model PatchTST \
    --seq_len 52 \
    --pred_len 8 \
    --batch_size 16 \
    --train_epochs 5 \
    --itr 1 \
    --features S \
    --target positive_rate
```

---

**Chúc bạn thành công! 🎉**
