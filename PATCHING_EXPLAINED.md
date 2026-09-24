# PATCHING MECHANISM - GIẢI THÍCH CHI TIẾT

## 🎯 PATCHING LÀ GÌ?

**Patching** là kỹ thuật chia time series thành các **đoạn nhỏ (patches)** để:
1. Giảm độ dài sequence → Giảm complexity của Transformer
2. Tạo local features → Capture pattern tốt hơn
3. Giảm memory usage

**Tương tự**: Như Vision Transformer (ViT) chia ảnh thành patches 16x16

---

## 📊 PATCHING PROCESS - BƯỚC TỪNG BƯỚC

### **Input Parameters:**
```python
seq_len = 52        # 52 tuần (1 năm)
patch_size = 16     # Mỗi patch chứa 16 time steps
stride = 8          # Sliding window với bước nhảy 8
```

### **Step-by-Step Visualization:**

```
ORIGINAL INPUT (52 tuần):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│15│16│17│18│19│20│...│52
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
     positive_rate values (flu rate)

Shape: (Batch, 52, 1)
```

---

### **STEP 1: Normalization (RevIN)**

```python
# Tính mean và std
means = x.mean(1, keepdim=True).detach()
stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

# Normalize
x = (x - means) / stdev
```

**Mục đích**: 
- Chuẩn hóa về mean=0, std=1
- Giúp model học stable hơn
- Sau khi predict, sẽ denormalize lại

```
BEFORE:  [4.5, 5.2, 3.8, 6.1, ...]  (flu rates)
AFTER:   [-0.2, 0.1, -0.4, 0.3, ...]  (normalized)
```

---

### **STEP 2: Reshape - Batch × Length × Channel**

```python
x = rearrange(x, 'b l m -> b m l')
# From: (Batch, Length, Channels)
# To:   (Batch, Channels, Length)
```

**Ví dụ:**
```
BEFORE: (32, 52, 1)  → Batch=32, Length=52, Channels=1
AFTER:  (32, 1, 52)  → Batch=32, Channels=1, Length=52
```

**Lý do**: Để áp dụng Conv1D operations (PyTorch convention)

---

### **STEP 3: Padding**

```python
self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
x = self.padding_patch_layer(x)
```

**Padding thêm `stride=8` time steps ở cuối:**

```
BEFORE PADDING (52 time steps):
[w1, w2, w3, ..., w52]

AFTER PADDING (52 + 8 = 60 time steps):
[w1, w2, w3, ..., w52, w52, w52, w52, w52, w52, w52, w52, w52]
                        └────────── 8 copies của w52 ──────────┘
```

**Shape**: (32, 1, 52) → (32, 1, 60)

**Lý do**: Đảm bảo patch cuối không bị thiếu data

---

### **STEP 4: Unfold - Tạo Patches (QUAN TRỌNG NHẤT!)**

```python
x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
```

**PyTorch unfold()** hoạt động như **sliding window**:
- `dimension=-1`: Theo chiều Length
- `size=patch_size=16`: Mỗi patch có 16 time steps
- `step=stride=8`: Mỗi lần trượt 8 bước

#### **Visual Explanation:**

```
INPUT (60 time steps after padding):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐...┬──┬──┬──┐
│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│15│16│   │58│59│60│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   └──┴──┴──┘

SLIDING WINDOW với patch_size=16, stride=8:

Patch 0: [1──→16]        ← 16 time steps
         └──────────────────┘

Patch 1:         [9──→24]     ← Overlapping 8 steps!
                 └──────────────────┘

Patch 2:                 [17──→32]
                         └──────────────────┘

Patch 3:                         [25──→40]
                                 └──────────────────┘

Patch 4:                                 [33──→48]
                                         └──────────────────┘

Patch 5:                                         [41──→56]
                                                 └──────────────────┘

Patch 6:                                                 [45──→60]
                                                         └──────────────────┘
```

**Công thức tính số patches:**
```python
patch_num = (seq_len - patch_size) // stride + 1
          = (52 - 16) // 8 + 1
          = 36 // 8 + 1
          = 4 + 1
          = 5 patches (trước padding)

# Sau padding thêm stride:
seq_len_padded = 52 + 8 = 60
patch_num = (60 - 16) // 8 + 1
          = 44 // 8 + 1
          = 5 + 1
          = 6 patches
```

**Shape sau unfold:**
```python
Before: (32, 1, 60)  → (Batch, Channels, Length_padded)
After:  (32, 1, 6, 16) → (Batch, Channels, num_patches, patch_size)
```

---

### **STEP 5: Rearrange Patches**

```python
x = rearrange(x, 'b m n p -> (b m) n p')
```

**Reshape để xử lý từng channel như batch:**

```python
From: (32, 1, 6, 16)  → (Batch, Channels, num_patches, patch_size)
To:   (32, 6, 16)     → (Batch×Channels, num_patches, patch_size)
```

**Ý nghĩa**:
- Mỗi sample giờ có 6 patches
- Mỗi patch có 16 time steps
- Sẽ xử lý 6 patches như 6 "tokens" trong Transformer

---

## 🧮 CONCRETE EXAMPLE

### **Ví dụ với flu data thực tế:**

```python
# Input: 52 tuần flu data
input_data = [
    4.83, 5.01, 4.45, 4.73, 3.43, 3.79, 2.49, 3.18,  # Tuần 1-8
    2.92, 3.05, 2.56, 3.59, 4.04, 4.20, 4.48, 7.14,  # Tuần 9-16
    3.04, 3.44, 3.39, 2.96, 2.67, 4.88, 6.24, 2.80,  # Tuần 17-24
    ...                                               # ... đến tuần 52
]

# Sau normalization
normalized = [-0.15, -0.09, -0.21, ...]

# Sau patching (6 patches × 16 time steps each)
patches = [
    Patch 0: [-0.15, -0.09, -0.21, -0.18, -0.32, ...],  # 16 values
    Patch 1: [-0.32, -0.28, -0.41, -0.29, -0.31, ...],  # 16 values  
    Patch 2: [-0.31, -0.35, -0.33, -0.39, -0.42, ...],  # 16 values
    Patch 3: [-0.42, -0.36, 0.12, 0.22, 0.28, ...],     # 16 values
    Patch 4: [0.28, 0.35, 0.42, 0.51, 0.65, ...],       # 16 values
    Patch 5: [0.65, 0.78, 0.92, 0.98, 0.98, ...],       # 16 values
]
```

---

## 💡 TẠI SAO LẠI DÙNG OVERLAPPING PATCHES?

### **Overlap = stride < patch_size**

```
NO OVERLAP (stride = patch_size = 16):
[────Patch 0────][────Patch 1────][────Patch 2────]
  No information sharing between patches!

WITH OVERLAP (stride = 8, patch_size = 16):
[────Patch 0────]
        [────Patch 1────]
                [────Patch 2────]
   └─8 overlap─┘  └─8 overlap─┘
   Information flows smoothly!
```

**Lợi ích:**
1. ✅ **Smooth transition**: Không bỏ lỡ thông tin giữa các patches
2. ✅ **More patches**: 6 patches thay vì 3 → More training samples
3. ✅ **Better context**: Mỗi time step xuất hiện trong nhiều patches

---

## 🔢 DETAILED CALCULATION

### **Với các hyperparameters mặc định:**

```python
# Hyperparameters
seq_len = 52
patch_size = 16
stride = 8

# Step 1: Calculate initial patches (before padding)
num_patches_init = (seq_len - patch_size) // stride + 1
                 = (52 - 16) // 8 + 1
                 = 36 // 8 + 1
                 = 4 + 1
                 = 5

# Step 2: Add padding
padding = stride = 8
seq_len_padded = seq_len + padding
               = 52 + 8
               = 60

# Step 3: Calculate final patches (after padding)
num_patches_final = (seq_len_padded - patch_size) // stride + 1
                  = (60 - 16) // 8 + 1
                  = 44 // 8 + 1
                  = 5 + 1
                  = 6 patches

# Step 4: Shape transformations
Input:           (Batch=32, Length=52, Channels=1)
After normalize: (32, 52, 1)
After reshape:   (32, 1, 52)
After padding:   (32, 1, 60)
After unfold:    (32, 1, 6, 16)  ← 6 patches, each 16 time steps
After rearrange: (32, 6, 16)
```

---

## 🎨 VISUALIZATION - COMPLETE FLOW

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Flu Time Series (52 tuần)                          │
│  Shape: (Batch=32, Length=52, Channels=1)                  │
│  Values: [4.83, 5.01, 4.45, ..., 1.87]                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 1: Normalization         │
        │   mean = 10.5, std = 8.2        │
        │   x = (x - mean) / std          │
        └─────────────────────────────────┘
                          ↓
      (32, 52, 1) with normalized values
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 2: Reshape               │
        │   'b l m -> b m l'              │
        └─────────────────────────────────┘
                          ↓
              (32, 1, 52)
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 3: Padding               │
        │   Add 8 time steps at end       │
        └─────────────────────────────────┘
                          ↓
              (32, 1, 60)
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 4: Unfold (PATCHING!)    │
        │   size=16, stride=8             │
        │   Sliding window                │
        └─────────────────────────────────┘
                          ↓
           (32, 1, 6, 16)
           └─┬──┬──┬──┬──┬──┬─┘
             6 patches
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 5: Rearrange             │
        │   'b m n p -> (b m) n p'        │
        └─────────────────────────────────┘
                          ↓
              (32, 6, 16)
           ┌──┴───┴───┴───┴───┴──┐
           Batch × num_patches × patch_size
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 6: Linear Projection     │
        │   16 → d_model (768 or 4096)    │
        └─────────────────────────────────┘
                          ↓
          (32, 6, 768) or (32, 6, 4096)
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 7: Transformer/LLM       │
        │   6 patches as 6 tokens         │
        │   Self-attention between patches│
        └─────────────────────────────────┘
                          ↓
          (32, 6, 768) or (32, 6, 4096)
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 8: Flatten & FC          │
        │   6*768 → fc_layer (512)        │
        └─────────────────────────────────┘
                          ↓
              (32, 512)
                          ↓
        ┌─────────────────────────────────┐
        │   STEP 9: Output Layer          │
        │   512 → pred_len (8 or 13)      │
        └─────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Predicted Flu Rates                                │
│  Shape: (Batch=32, pred_len=13, Channels=1)                │
│  Values: [predicted flu rates for next 13 weeks]            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 CODE DEEP DIVE

### **Unfold Function - How It Works:**

```python
import torch

# Ví dụ đơn giản
x = torch.arange(1, 13).float()  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x = x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 12)

# Unfold với patch_size=4, stride=2
patches = x.unfold(dimension=-1, size=4, step=2)
# Shape: (1, 1, 5, 4)
#             └─┬─┘  └─┘
#           5 patches, each 4 elements

print(patches)
# Patch 0: [1, 2, 3, 4]
# Patch 1: [3, 4, 5, 6]      ← Overlap 2 elements with Patch 0
# Patch 2: [5, 6, 7, 8]      ← Overlap 2 elements with Patch 1
# Patch 3: [7, 8, 9, 10]
# Patch 4: [9, 10, 11, 12]
```

### **Actual Code in Models:**

```python
# From Llama2.py (line 68-71)
x = rearrange(x, 'b l m -> b m l')           # (B, 52, 1) → (B, 1, 52)
x = self.padding_patch_layer(x)              # (B, 1, 52) → (B, 1, 60)
x = x.unfold(dimension=-1, 
             size=self.patch_size,            # 16
             step=self.stride)                # 8
                                              # (B, 1, 60) → (B, 1, 6, 16)
x = rearrange(x, 'b m n p -> (b m) n p')     # (B, 1, 6, 16) → (B, 6, 16)
```

---

## 📊 COMPARISON: With vs Without Patching

### **WITHOUT Patching (Naive Transformer):**
```python
Input: (Batch=32, Length=52, Channels=1)
       ↓ Embed to d_model=768
       ↓ (32, 52, 768)
       ↓ Transformer with 52 tokens
       ↓ Attention: O(52² × d_model) = Very expensive!
```

**Problems:**
- ❌ Long sequence (52 tokens)
- ❌ O(L²) attention complexity
- ❌ Hard to learn local patterns

### **WITH Patching:**
```python
Input: (Batch=32, Length=52, Channels=1)
       ↓ Patching → 6 patches
       ↓ (32, 6, 16)
       ↓ Embed each patch to d_model=768
       ↓ (32, 6, 768)
       ↓ Transformer with 6 tokens
       ↓ Attention: O(6² × d_model) = Much cheaper!
```

**Benefits:**
- ✅ Short sequence (6 tokens instead of 52)
- ✅ O(6²) << O(52²) → 75x faster!
- ✅ Each patch captures local pattern (16 weeks)
- ✅ Attention between patches captures global pattern

---

## 🎯 INTUITION - WHY IT WORKS

### **Analogy với ngôn ngữ tự nhiên:**

```
Sentence:  "The quick brown fox jumps over the lazy dog"
           └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
              Phrase 1      Phrase 2      Phrase 3

Time Series:  [Week 1-16] [Week 9-24] [Week 17-32] ...
               └─Patch 0─┘ └─Patch 1─┘ └─Patch 2─┘
```

**Mỗi patch = một "phrase" trong câu:**
- Mỗi phrase có nghĩa riêng (local pattern)
- Attention giữa phrases tạo nghĩa toàn câu (global pattern)
- Overlapping giúp phrases liên kết mượt mà

### **Cho flu forecasting:**

```
Patch 0 (Week 1-16):   Mùa xuân → Flu thấp, giảm dần
Patch 1 (Week 9-24):   Cuối xuân, đầu hè → Flu rất thấp
Patch 2 (Week 17-32):  Mùa hè → Flu ở mức thấp nhất
Patch 3 (Week 25-40):  Cuối hè, đầu thu → Bắt đầu tăng
Patch 4 (Week 33-48):  Mùa thu → Flu tăng mạnh
Patch 5 (Week 41-56):  Mùa đông → Flu ở đỉnh cao

→ Model học được: 
  - Local: Pattern trong mỗi mùa
  - Global: Chu kỳ mùa vụ qua năm
```

---

## 🧪 EXPERIMENT: Thử nghiệm với các giá trị khác

### **Tăng patch_size:**
```python
patch_size = 26  # Thay vì 16
stride = 13      # Thay vì 8

num_patches = (52 - 26) // 13 + 1 = 3 patches
```
**Effect:**
- ✅ Mỗi patch capture nhiều context hơn (26 tuần = 6 tháng)
- ❌ Ít patches hơn (3 thay vì 6) → Ít training samples
- ❌ Coarser granularity

### **Giảm stride (More overlap):**
```python
patch_size = 16
stride = 4       # Thay vì 8 (overlap nhiều hơn)

num_patches = (52 - 16) // 4 + 1 = 10 patches
```
**Effect:**
- ✅ Nhiều patches hơn (10 thay vì 6)
- ✅ Smoother transitions
- ❌ More computation (10 tokens vs 6)
- ❌ More redundancy

---

## 💡 KEY TAKEAWAYS

1. **Patching = Sliding Window** với overlap
2. **6 patches** từ 52 tuần (với patch_size=16, stride=8)
3. **Each patch = 16 consecutive weeks** của flu data
4. **Overlap = 8 weeks** giữa các patches liên tiếp
5. **Giảm complexity** từ O(52²) xuống O(6²)
6. **Capture both local và global patterns**

---

## 🔧 CODE ĐỂ VISUALIZE PATCHING

```python
import torch
import matplotlib.pyplot as plt

# Simulate flu data (52 weeks)
flu_data = torch.sin(torch.linspace(0, 4*3.14159, 52)) + torch.randn(52) * 0.1
flu_data = flu_data.unsqueeze(0).unsqueeze(-1)  # (1, 52, 1)

# Patching
patch_size = 16
stride = 8

# Add batch and channel dims
x = flu_data.permute(0, 2, 1)  # (1, 1, 52)

# Padding
padding = torch.nn.ReplicationPad1d((0, stride))
x_padded = padding(x)  # (1, 1, 60)

# Unfold
patches = x_padded.unfold(-1, patch_size, stride)  # (1, 1, 6, 16)

# Visualize
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    patch = patches[0, 0, i, :].numpy()
    axes[i].plot(patch, marker='o')
    axes[i].set_title(f'Patch {i} (Overlap with Patch {i-1} if i>0)')
    axes[i].set_xlabel('Time step within patch')
    axes[i].set_ylabel('Flu rate')
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('patching_visualization.png')
print("Saved patching_visualization.png")
```

Chạy code này sẽ tạo hình minh họa 6 patches!
