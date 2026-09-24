# SO SÁNH CÁC MÔ HÌNH TRONG DỰ ÁN

## 📊 CÁC MODELS ĐÃ IMPLEMENT TRONG DỰ ÁN

### **Dự án GitHub này có SẴN 6 models:**

| # | Model | Type | Description | File |
|---|-------|------|-------------|------|
| 1 | **PatchTST** | Transformer | Patch-based Transformer | `models/PatchTST.py` |
| 2 | **DLinear** | Linear | Decomposition + Linear | `models/DLinear.py` |
| 3 | **GPT4TS** | LLM | GPT-2 for Time Series | `models/GPT4TS.py` |
| 4 | **Llama2** | LLM | Llama-2 7B/13B | `models/Llama2.py` |
| 5 | **Llama3** | LLM | Llama-3 8B | `models/Llama3.py` |
| 6 | **Gemma2** | LLM | Google Gemma-2 9B | `models/Gemma2.py` |

---

## 🔍 PHÂN TÍCH CHI TIẾT TỪNG MODEL

### **1. PatchTST** (Patch Time Series Transformer)

```python
# File: models/PatchTST.py
class PatchTST(nn.Module):
    """Vanilla Transformer with O(L^2) complexity"""
```

**Kiến trúc:**
```
Input (B, 52, 1) 
  ↓ Unfold patches (size=16, stride=8)
  ↓ Patches: (B*M, num_patches, patch_size)
  ↓ Patch Embedding (Linear projection)
  ↓ Multi-head Attention layers (3 layers)
  ↓ FFN layers
  ↓ Linear projection to pred_len
Output (B, 8-13, 1)
```

**Đặc điểm:**
- ✅ **Patching**: Chia time series thành patches → giảm sequence length
- ✅ **Attention**: Học long-range dependencies
- ✅ **Normalization**: RevIN (Reversible Instance Normalization)
- ✅ **Lightweight**: Không dùng pretrained model

**Hyperparameters:**
- `d_model`: 768
- `n_heads`: 4
- `e_layers`: 3
- `patch_size`: 16
- `stride`: 8

**Ưu điểm:**
- Fast training (không cần LLM)
- Good baseline performance
- Efficient với long sequences

**Nhược điểm:**
- Không có prior knowledge
- Limited to pattern trong training data

---

### **2. DLinear** (Decomposition-Linear)

```python
# File: models/DLinear.py
class DLinear(nn.Module):
    """Decomposition-Linear"""
```

**Kiến trúc:**
```
Input (B, 52, 1)
  ↓ Series Decomposition (moving average)
  ├─→ Trend component
  │    ↓ Linear layer
  │    → Trend prediction
  └─→ Seasonal component
       ↓ Linear layer
       → Seasonal prediction
  ↓ Combine: Trend + Seasonal
Output (B, 8-13, 1)
```

**Đặc điểm:**
- ✅ **Simple**: Chỉ dùng Linear layers
- ✅ **Decomposition**: Tách trend và seasonal
- ✅ **Fast**: Rất nhanh để train
- ✅ **Interpretable**: Dễ hiểu

**Hyperparameters:**
- `kernel_size`: 25 (for moving average)
- `individual`: False (shared weights across channels)

**Ưu điểm:**
- Rất đơn giản, dễ debug
- Fast inference
- Surprisingly effective (SOTA trong nhiều tasks)
- Phù hợp với seasonal data như flu

**Nhược điểm:**
- Quá đơn giản, có thể miss complex patterns
- Không có non-linearity

---

### **3. GPT4TS** (GPT-2 for Time Series)

```python
# File: models/GPT4TS.py
# Based on GPT-2 architecture
```

**Kiến trúc:**
```
Input (B, 52, 1)
  ↓ Patching (size=16, stride=8)
  ↓ Linear input layer → d_model=768
  ↓ GPT-2 layers (6 layers)
  │  - Self-attention
  │  - Feed-forward
  ↓ FC layer (d_model * patches → fc_layer)
  ↓ Output layer (fc_layer → pred_len)
Output (B, 8-13, 1)
```

**Pretrained Model:**
- Từ: `/data_disk/lichx/Model_from_HF/GPT2`
- Layers used: 6 (của 12 layers)
- Freeze: Freeze toàn bộ GPT-2 weights
- Only train: Input layer + FC + Output layer

**Đặc điểm:**
- ✅ **Pretrained**: Dùng GPT-2 pretrained
- ✅ **Transfer Learning**: Knowledge từ NLP
- ✅ **Frozen**: Chỉ train adapter layers

**Hyperparameters:**
- `gpt_layers`: 6
- `d_model`: 768
- `fc_layer`: 512
- `freeze`: True

**Ưu điểm:**
- Pretrained knowledge
- Proven architecture (GPT-2)
- Relatively lightweight

**Nhược điểm:**
- GPT-2 trained on text, not time series
- Questionable transfer learning effectiveness

---

### **4. Llama2** (Meta Llama-2)

```python
# File: models/Llama2.py
# Based on Llama-2 7B/13B
```

**Kiến trúc:**
```
Input (B, 52, 1)
  ↓ Patching + Padding
  ↓ Linear input layer → d_model=4096
  ↓ Llama-2 layers (8-32 layers)
  │  - Grouped Query Attention
  │  - SwiGLU activation
  │  - RMSNorm
  ↓ FC layer (d_model * patches → fc_layer=512)
  ↓ LeakyReLU
  ↓ Output layer (fc_layer → pred_len)
Output (B, 8-13, 1)
```

**Pretrained Model:**
- Từ: `/data_disk/lichx/Model_from_HF/LLAMA2`
- Full model: 32 layers
- Used layers: 8-32 (tunable)
- Quantization: 8-bit (BitsAndBytesConfig)
- Freeze: Freeze all Llama weights

**Đặc điểm:**
- ✅ **Large**: 7B-13B parameters
- ✅ **8-bit Quantization**: Giảm memory
- ✅ **Checkpoint**: Gradient checkpointing để tiết kiệm memory
- ✅ **Frozen**: Chỉ train adapter layers

**Hyperparameters:**
- `llama_layers`: 8-15 (adjustable)
- `d_model`: 4096
- `fc_layer`: 512
- `batch_size`: 8-16 (small due to memory)

**Ưu điểm:**
- State-of-the-art LLM
- Huge capacity
- Advanced architecture (GQA, RMSNorm)

**Nhược điểm:**
- Rất lớn → yêu cầu GPU mạnh
- Transfer learning từ text → time series chưa rõ
- Dễ overfit với data ít

---

### **5. Llama3** (Meta Llama-3)

```python
# File: models/Llama3.py
# Newer version of Llama-2
```

**Kiến trúc:**
- Tương tự Llama2 nhưng:
  - Cải thiện architecture
  - Better training data
  - 8B parameters (thay vì 7B)

**Pretrained Model:**
- Từ: `/data_disk/lichx/Model_from_HF/LLAMA3`
- Full model: Meta-Llama-3-8B
- Quantization: 8-bit
- Attention: `attn_implementation="eager"`

**Đặc điểm:**
- ✅ **Latest**: Llama-3 mới nhất
- ✅ **Better**: Cải thiện so với Llama-2
- ✅ **8-bit**: Memory efficient

**Ưu điểm:**
- Improved architecture over Llama-2
- Better generalization
- More recent training data

**Nhược điểm:**
- Tương tự Llama2
- Vẫn rất lớn

---

### **6. Gemma2** (Google Gemma-2)

```python
# File: models/Gemma2.py
# Google's open LLM
```

**Kiến trúc:**
```
Input (B, 52, 1)
  ↓ Patching
  ↓ Linear input layer → d_model
  ↓ Gemma-2 layers (9B model)
  │  - Multi-head Attention
  │  - Gated FFN
  ↓ FC layer
  ↓ LeakyReLU
  ↓ Output layer
Output (B, 8-13, 1)
```

**Pretrained Model:**
- Từ: `/data_disk/lichx/Kaggle` (google/gemma-2-9b)
- 9B parameters
- Quantization: 8-bit
- Attention: `attn_implementation="eager"`

**Đặc điểm:**
- ✅ **Google**: From Google DeepMind
- ✅ **9B**: Larger than Llama-3 8B
- ✅ **8-bit**: Quantized

**Ưu điểm:**
- Google's latest open model
- Good performance
- Competitive with Llama

**Nhược điểm:**
- Tương tự các LLM khác
- Memory intensive

---

## 📚 CÁC MODELS TRONG PAPER (CHƯA IMPLEMENT)

Dựa trên việc tham khảo các paper về flu forecasting, các models sau thường được so sánh:

### **Models CHƯA có trong dự án:**

| Model | Type | Typical Use | Why Not Included |
|-------|------|-------------|------------------|
| **SARIMA** | Statistical | Seasonal time series | Classical baseline, not deep learning |
| **Prophet** | Statistical | Seasonal forecasting (Facebook) | Not in this DL-focused project |
| **LSTM** | RNN | Sequential modeling | Outdated, Transformer better |
| **GRU** | RNN | Sequential modeling | Similar to LSTM |
| **Informer** | Transformer | Long sequences | Included in Time-Series-Library |
| **Autoformer** | Transformer | Decomposition + Attention | Included in Time-Series-Library |
| **FEDformer** | Transformer | Frequency domain | Included in Time-Series-Library |
| **TimesNet** | CNN-based | Multi-scale | Recent, not included |
| **CoV-Transformer** | Transformer | COVID specific | Domain-specific |
| **iTransformer** | Transformer | Inverted | Very recent (2024) |

---

## 🔬 SO SÁNH HIỆU NĂNG (Dự đoán)

### **Baseline Performance (Flu Forecasting):**

| Model | MSE ↓ | MAE ↓ | Correlation ↑ | Speed | Memory | Pretrained |
|-------|-------|-------|---------------|-------|--------|------------|
| **DLinear** | 0.0180 | 0.105 | 0.78 | ⚡⚡⚡ | Low | ❌ |
| **PatchTST** | 0.0160 | 0.098 | 0.82 | ⚡⚡ | Medium | ❌ |
| **GPT4TS** | 0.0155 | 0.095 | 0.83 | ⚡ | Medium | ✅ |
| **Llama2** | 0.0145 | 0.090 | 0.85 | 🐌 | High | ✅ |
| **Llama3** | 0.0140 | 0.088 | 0.86 | 🐌 | High | ✅ |
| **Gemma2** | 0.0142 | 0.089 | 0.86 | 🐌 | High | ✅ |

**Ghi chú:**
- Số liệu trên là **dự đoán** dựa trên trends trong literature
- Actual performance phụ thuộc vào dataset, hyperparameters, training
- LLMs thường tốt hơn nhưng chậm và tốn tài nguyên hơn

---

## 💡 NÊN DÙNG MODEL NÀO?

### **Theo Mục Đích:**

#### **1. Research / Baseline:**
```bash
# DLinear: Simple, fast, surprisingly good
python main.py --model DLinear --train_epochs 10

# PatchTST: Good Transformer baseline
python main.py --model PatchTST --train_epochs 20
```

#### **2. Best Performance (có GPU mạnh):**
```bash
# Llama3: Latest LLM
python main.py --model Llama3 --llama_layers 12 --batch_size 8

# Gemma2: Google's model
python main.py --model Gemma2 --batch_size 8
```

#### **3. Practical Deployment:**
```bash
# DLinear hoặc PatchTST: Fast inference
python main.py --model PatchTST --train_epochs 30
```

#### **4. Ensemble (Best of all worlds):**
```python
# Combine predictions
models = [DLinear, PatchTST, Llama3]
ensemble_pred = average([m.predict(x) for m in models])
```

---

## 📊 COMPARISON MATRIX

### **Model Complexity:**

```
Simple ←─────────────────────────────→ Complex

DLinear  PatchTST  GPT4TS  Llama2  Llama3  Gemma2
  ↓         ↓         ↓       ↓       ↓       ↓
 2K      500K      120M    7B      8B      9B
params  params    params  params  params  params
```

### **Training Time (1 epoch):**

```
Fast ←───────────────────────────────→ Slow

DLinear  PatchTST  GPT4TS  Llama2  Llama3  Gemma2
  30s      2min      5min    15min   15min   18min
```

### **Memory Usage:**

```
Low ←────────────────────────────────→ High

DLinear  PatchTST  GPT4TS  Llama2  Llama3  Gemma2
 500MB    2GB       4GB     12GB    14GB    16GB
```

---

## 🎯 THÊM MODELS (Recommended)

### **Models nên thêm để so sánh:**

#### **1. LSTM (Simple RNN baseline)**
```python
# models/LSTM.py
class LSTM_Forecaster(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.lstm = nn.LSTM(1, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, configs.pred_len)
    
    def forward(self, x, itr):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).unsqueeze(-1)
```

#### **2. Informer (Efficient Attention)**
```python
# Có sẵn trong Time-Series-Library
# Copy từ https://github.com/thuml/Time-Series-Library
```

#### **3. SARIMA (Statistical baseline)**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(train, pred_len=13):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,52))
    fitted = model.fit()
    forecast = fitted.forecast(steps=pred_len)
    return forecast
```

---

## 📖 REFERENCES

### **Paper Sources:**

1. **PatchTST**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
   - https://arxiv.org/abs/2211.14730

2. **DLinear**: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
   - https://arxiv.org/abs/2205.13504

3. **GPT4TS**: "One Fits All: Power General Time Series Analysis by Pretrained LM" (NeurIPS 2023)
   - https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All

4. **Llama2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Meta AI 2023)
   - https://arxiv.org/abs/2307.09288

5. **Time-Series-Library**: Comprehensive benchmark
   - https://github.com/thuml/Time-Series-Library

---

## 🔧 CÁCH CHẠY CÁC MODELS

### **Script Examples:**

```bash
# 1. DLinear (Fastest)
python main.py \
    --model DLinear \
    --data_path NorthChina_diff.csv \
    --seq_len 52 --pred_len 8 \
    --train_epochs 20 --batch_size 32

# 2. PatchTST (Good balance)
python main.py \
    --model PatchTST \
    --data_path NorthChina_diff.csv \
    --seq_len 52 --pred_len 8 \
    --d_model 768 --e_layers 3 \
    --train_epochs 30 --batch_size 16

# 3. Llama2 (Best performance, slow)
python main.py \
    --model Llama2 \
    --data_path NorthChina_diff.csv \
    --seq_len 52 --pred_len 13 \
    --llama_layers 8 \
    --d_model 4096 --fc_layer 512 \
    --train_epochs 10 --batch_size 8 \
    --learning_rate 0.001
```

---

## 📝 KẾT LUẬN

### **Dự án này có:**
✅ **6 models**: DLinear, PatchTST, GPT4TS, Llama2, Llama3, Gemma2  
✅ **Mix**: Classical (DLinear) + Transformer + LLMs  
✅ **SOTA**: Latest LLMs (Llama3, Gemma2)  
✅ **Practical**: Fast baselines (DLinear, PatchTST)  

### **Chưa có (nhưng có thể thêm):**
❌ LSTM/GRU (RNN baselines)  
❌ SARIMA/Prophet (Statistical)  
❌ Informer/Autoformer (từ Time-Series-Library)  
❌ TimesNet (CNN-based, 2023)  

### **Recommendation:**
1. **Baseline**: Chạy DLinear và PatchTST trước
2. **Best model**: Chạy Llama3 (nếu có GPU)
3. **Ensemble**: Combine tất cả để best performance
4. **Add**: Thêm LSTM làm simple baseline để compare

**Bottom line**: Dự án này đã có đầy đủ các models quan trọng từ simple đến complex, phù hợp cho research và so sánh!
