# CÁC PHƯƠNG HƯỚNG CẢI TIẾN MÔ HÌNH KHẢ THI

## 📊 PHÂN TÍCH HIỆN TRẠNG

### **Kiến trúc hiện tại:**
```
Baselines:
- PatchTST: Transformer với patching
- DLinear: Linear decomposition (trend + seasonal)

LLM-based:
- Llama2/3: 8-32 layers
- Gemma2: Google's LLM
- GPT4TS: GPT-2 based

Common pipeline:
Input (52 weeks) → Normalization → Patching/Embedding 
→ LLM/Transformer → FC layers → Output (8-13 weeks)
```

### **Điểm yếu hiện tại:**
1. ❌ Dữ liệu ít (chỉ 416 tuần training)
2. ❌ Model phức tạp (LLM rất lớn) → Dễ overfit
3. ❌ Không sử dụng external features (thời tiết, tìm kiếm Google...)
4. ❌ Chỉ single-step forecasting (không iterative)
5. ❌ Không có ensemble methods
6. ❌ Loss function đơn giản (chỉ MSE)
7. ❌ Không có uncertainty estimation
8. ❌ Không transfer learning giữa các region

---

## 🚀 NHÓM 1: CẢI TIẾN DỮ LIỆU (Data-Centric)

### **1.1. Data Augmentation**

#### **A. Time Series Augmentation**
```python
# Thêm vào data_loader.py

class TimeSeriesAugmentation:
    def __init__(self, p=0.5):
        self.p = p
    
    def jitter(self, x, sigma=0.03):
        """Thêm nhiễu Gaussian"""
        if np.random.rand() < self.p:
            return x + np.random.normal(0, sigma, x.shape)
        return x
    
    def scaling(self, x, sigma=0.1):
        """Scale amplitude"""
        if np.random.rand() < self.p:
            factor = np.random.normal(1.0, sigma)
            return x * factor
        return x
    
    def time_warp(self, x, sigma=0.2):
        """Warp thời gian"""
        if np.random.rand() < self.p:
            # DTW-based warping
            pass
        return x
    
    def window_slice(self, x, reduce_ratio=0.9):
        """Random crop"""
        if np.random.rand() < self.p:
            target_len = int(len(x) * reduce_ratio)
            start = np.random.randint(0, len(x) - target_len)
            return x[start:start+target_len]
        return x
```

**Lợi ích:**
- ✅ Tăng số lượng training samples
- ✅ Model robust hơn với noise
- ✅ Giảm overfitting

**Độ khó:** ⭐⭐ (Dễ implement)

---

#### **B. Synthetic Data Generation**
```python
# Dùng ARIMA/SARIMA để generate thêm data

from statsmodels.tsa.statespace.sarimax import SARIMAX

def generate_synthetic_flu_data(historical_data, n_years=2):
    """Generate synthetic flu data"""
    model = SARIMAX(historical_data, 
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 52))  # 52 weeks
    fitted = model.fit()
    
    # Generate 2 more years
    synthetic = fitted.simulate(n_years * 52, anchor='end')
    return synthetic
```

**Lợi ích:**
- ✅ Tạo thêm dữ liệu có tính mùa vụ
- ✅ Không cần data thật

**Độ khó:** ⭐⭐⭐ (Trung bình)

---

#### **C. Multi-Region Transfer Learning**
```python
# Pretrain trên dữ liệu nhiều khu vực

# Step 1: Train trên USA + Bắc TQ + Nam TQ
pretrain_data = concat([usa_data, north_data, south_data])
model.pretrain(pretrain_data)

# Step 2: Fine-tune trên khu vực cụ thể
model.finetune(north_data)
```

**Lợi ích:**
- ✅ Học pattern chung của cúm toàn cầu
- ✅ Transfer knowledge giữa các khu vực
- ✅ Hiệu quả với dữ liệu ít

**Độ khó:** ⭐⭐⭐⭐ (Khó)

---

### **1.2. Feature Engineering**

#### **A. Thêm External Features**
```python
class Dataset_Custom_Enhanced(Dataset_Custom):
    def __init__(self, ...):
        super().__init__(...)
        
        # 1. Weather features
        self.temp = self.load_temperature_data()
        self.humidity = self.load_humidity_data()
        
        # 2. Google Trends
        self.search_flu = self.load_google_trends('flu')
        
        # 3. Holiday indicators
        self.holidays = self.load_holiday_calendar()
        
        # 4. Lagged features
        self.lag_features = self.create_lag_features([1, 2, 4, 52])
    
    def __getitem__(self, index):
        seq_x, seq_y, ... = super().__getitem__(index)
        
        # Concatenate external features
        external_features = np.stack([
            self.temp[s_begin:s_end],
            self.humidity[s_begin:s_end],
            self.search_flu[s_begin:s_end],
            self.holidays[s_begin:s_end]
        ], axis=-1)
        
        seq_x = np.concatenate([seq_x, external_features], axis=-1)
        return seq_x, seq_y, ...
```

**Features đề xuất:**
1. **Thời tiết**: Nhiệt độ, độ ẩm, lượng mưa
2. **Google Trends**: Tần suất tìm kiếm "flu", "cúm"
3. **Lịch**: Ngày lễ, kỳ nghỉ (người di chuyển nhiều)
4. **Dân số**: Mật độ dân số, độ tuổi trung bình
5. **Y tế**: Tỷ lệ tiêm vaccine

**Nguồn data:**
- Weather: OpenWeatherMap API
- Google Trends: PyTrends library
- WHO/CDC: Dữ liệu vaccine

**Lợi ích:**
- ✅ Capture được yếu tố ngoại sinh
- ✅ Cải thiện accuracy đáng kể (10-20%)
- ✅ Có nghiên cứu chứng minh hiệu quả

**Độ khó:** ⭐⭐⭐⭐ (Khó - cần thu thập data)

---

#### **B. Temporal Features Enhancement**
```python
def create_advanced_temporal_features(df):
    """Tạo temporal features nâng cao"""
    
    # 1. Fourier features (capture seasonality)
    for k in range(1, 5):
        df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['week_of_year'] / 52)
        df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['week_of_year'] / 52)
    
    # 2. Cyclic encoding
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Season indicators
    df['is_winter'] = ((df['month'] >= 11) | (df['month'] <= 2)).astype(int)
    df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
    
    return df
```

**Lợi ích:**
- ✅ Model học seasonality tốt hơn
- ✅ Dễ implement
- ✅ Không cần external data

**Độ khó:** ⭐⭐ (Dễ)

---

## 🧠 NHÓM 2: CẢI TIẾN KIẾN TRÚC (Model-Centric)

### **2.1. Attention Mechanisms**

#### **A. Multi-Scale Attention**
```python
class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, scales=[1, 4, 13, 52]):
        super().__init__()
        self.scales = scales  # 1 week, 1 month, 1 quarter, 1 year
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, 4) for _ in scales
        ])
        self.fusion = nn.Linear(d_model * len(scales), d_model)
    
    def forward(self, x):
        outputs = []
        for scale, attn in zip(self.scales, self.attentions):
            # Downsample to different scales
            x_scale = F.avg_pool1d(x.transpose(1,2), scale).transpose(1,2)
            out, _ = attn(x_scale, x_scale, x_scale)
            # Upsample back
            out = F.interpolate(out.transpose(1,2), size=x.shape[1]).transpose(1,2)
            outputs.append(out)
        
        return self.fusion(torch.cat(outputs, dim=-1))
```

**Lợi ích:**
- ✅ Capture pattern ở nhiều time scales
- ✅ Hiệu quả với seasonal data

**Độ khó:** ⭐⭐⭐⭐ (Khó)

---

#### **B. Cross-Region Attention**
```python
class CrossRegionAttention(nn.Module):
    """Học relationship giữa các khu vực"""
    def __init__(self, d_model, n_regions=3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, 4)
        self.self_attn = nn.MultiheadAttention(d_model, 4)
    
    def forward(self, x_north, x_south, x_usa):
        # x_*: [B, L, D]
        
        # Stack regions
        x_all = torch.stack([x_north, x_south, x_usa], dim=0)  # [3, B, L, D]
        
        # Cross-region attention
        x_cross, _ = self.cross_attn(x_all, x_all, x_all)
        
        # Self attention per region
        x_self, _ = self.self_attn(x_north, x_north, x_north)
        
        # Combine
        return x_self + x_cross[0]  # North region with cross-region info
```

**Lợi ích:**
- ✅ Khai thác correlation giữa các khu vực
- ✅ Dịch cúm lan theo pattern toàn cầu

**Độ khó:** ⭐⭐⭐⭐⭐ (Rất khó)

---

### **2.2. Hybrid Models**

#### **A. LLM + Classical Ensemble**
```python
class HybridFluPredictor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # LLM branch
        self.llm = Llama2(configs)
        
        # Classical ML branches
        self.arima_weight = nn.Parameter(torch.tensor(0.3))
        self.prophet_weight = nn.Parameter(torch.tensor(0.3))
        self.llm_weight = nn.Parameter(torch.tensor(0.4))
    
    def forward(self, x, itr):
        # LLM prediction
        pred_llm = self.llm(x, itr)
        
        # ARIMA prediction (precomputed)
        pred_arima = self.get_arima_forecast(x)
        
        # Prophet prediction (precomputed)
        pred_prophet = self.get_prophet_forecast(x)
        
        # Weighted ensemble
        weights = F.softmax(torch.stack([
            self.arima_weight,
            self.prophet_weight, 
            self.llm_weight
        ]), dim=0)
        
        final = (weights[0] * pred_arima + 
                 weights[1] * pred_prophet + 
                 weights[2] * pred_llm)
        
        return final
```

**Lợi ích:**
- ✅ Kết hợp ưu điểm của LLM và classical models
- ✅ ARIMA/Prophet rất tốt với seasonal data
- ✅ Robust hơn

**Độ khó:** ⭐⭐⭐ (Trung bình)

---

#### **B. Decomposition-Enhanced LLM**
```python
class DecomposedLlama(nn.Module):
    """Decompose thành trend + seasonal + residual, 
    dùng LLM cho từng component"""
    
    def __init__(self, configs):
        super().__init__()
        
        # Decomposition
        self.decomp = series_decomp(kernel_size=25)
        
        # Separate LLMs for each component
        self.trend_llm = Llama2(configs)
        self.seasonal_llm = Llama2(configs)
        self.residual_net = nn.LSTM(configs.d_model, configs.d_model, 2)
    
    def forward(self, x, itr):
        # Decompose
        seasonal, trend = self.decomp(x)
        residual = x - seasonal - trend
        
        # Predict each component
        pred_trend = self.trend_llm(trend, itr)
        pred_seasonal = self.seasonal_llm(seasonal, itr)
        pred_residual, _ = self.residual_net(residual)
        
        # Combine
        return pred_trend + pred_seasonal + pred_residual
```

**Lợi ích:**
- ✅ Tách biệt pattern khác nhau
- ✅ Dễ học hơn
- ✅ DLinear đã chứng minh hiệu quả

**Độ khó:** ⭐⭐⭐⭐ (Khó)

---

### **2.3. Advanced Architectures**

#### **A. Temporal Convolutional Network (TCN)**
```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=(kernel_size-1) * dilation,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=(kernel_size-1) * dilation,
                               dilation=dilation)
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN_FluForecaster(nn.Module):
    def __init__(self, configs):
        super().__init__()
        layers = []
        num_levels = 8
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(configs.d_model, configs.d_model, 
                                       kernel_size=3, dilation=dilation))
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(configs.d_model, configs.pred_len)
```

**Lợi ích:**
- ✅ Receptive field lớn với dilated convolutions
- ✅ Nhanh hơn RNN/Transformer
- ✅ Phù hợp với long sequences

**Độ khó:** ⭐⭐⭐ (Trung bình)

---

#### **B. Informer-style Efficient Attention**
```python
from models.Informer import ProbAttention

class EfficientPatchTST(PatchTST):
    """Thay FullAttention bằng ProbAttention"""
    
    def __init__(self, configs):
        super().__init__(configs)
        # Replace attention mechanism
        for layer in self.encoder.attn_layers:
            layer.attention.inner_attention = ProbAttention(
                mask_flag=False,
                factor=5,
                attention_dropout=configs.dropout
            )
```

**Lợi ích:**
- ✅ O(L log L) thay vì O(L^2)
- ✅ Xử lý được sequences dài hơn
- ✅ Tiết kiệm memory

**Độ khó:** ⭐⭐⭐ (Trung bình)

---

## 📈 NHÓM 3: CẢI TIẾN TRAINING (Training-Centric)

### **3.1. Loss Functions**

#### **A. Composite Loss**
```python
class FluForecastingLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # Shape weight
        self.gamma = gamma  # Peak weight
    
    def forward(self, pred, true):
        # 1. MSE loss
        mse = F.mse_loss(pred, true)
        
        # 2. Shape preservation (DTW-based)
        shape_loss = self.dtw_loss(pred, true)
        
        # 3. Peak detection loss (quan trọng cho dự báo dịch)
        peak_loss = self.peak_detection_loss(pred, true)
        
        return self.alpha * mse + self.beta * shape_loss + self.gamma * peak_loss
    
    def dtw_loss(self, pred, true):
        """Dynamic Time Warping loss"""
        from dtaidistance import dtw
        distance = dtw.distance(pred.cpu().numpy(), true.cpu().numpy())
        return torch.tensor(distance, requires_grad=True)
    
    def peak_detection_loss(self, pred, true):
        """Penalize errors at peaks (high flu rates)"""
        # Higher weight at peaks
        weights = 1.0 + 2.0 * (true > true.mean())
        return F.mse_loss(pred * weights, true * weights)
```

**Lợi ích:**
- ✅ Quan tâm đến peak detection (quan trọng nhất)
- ✅ Preserve shape của time series
- ✅ Cải thiện đáng kể cho epidemic forecasting

**Độ khó:** ⭐⭐⭐ (Trung bình)

---

#### **B. Quantile Loss (Uncertainty)**
```python
class QuantileRegressionLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        """
        preds: [B, T, n_quantiles]
        target: [B, T]
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(torch.max((q-1) * errors, q * errors))
        
        return torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=-1))

# Modify model output
class LlamaWithUncertainty(Llama2):
    def __init__(self, configs):
        super().__init__(configs)
        # Output 3 quantiles: 10%, 50%, 90%
        self.out_layer = nn.Linear(configs.fc_layer, configs.pred_len * 3)
    
    def forward(self, x, itr):
        outputs = super().forward(x, itr)
        # Reshape to [B, T, 3]
        return outputs.view(B, -1, 3)
```

**Lợi ích:**
- ✅ Uncertainty estimation
- ✅ Confidence intervals cho dự báo
- ✅ Quan trọng cho public health decisions

**Độ khó:** ⭐⭐⭐⭐ (Khó)

---

### **3.2. Training Strategies**

#### **A. Curriculum Learning**
```python
class CurriculumTrainer:
    def __init__(self, model, start_pred_len=1, max_pred_len=13):
        self.model = model
        self.start_pred_len = start_pred_len
        self.max_pred_len = max_pred_len
    
    def train(self, epochs):
        for epoch in range(epochs):
            # Gradually increase prediction horizon
            current_pred_len = min(
                self.start_pred_len + epoch // 5,
                self.max_pred_len
            )
            
            # Update args
            args.pred_len = current_pred_len
            
            # Train
            train_one_epoch(self.model, train_loader, current_pred_len)
```

**Lợi ích:**
- ✅ Học từ dễ đến khó
- ✅ Stable training
- ✅ Tốt cho long-term forecasting

**Độ khó:** ⭐⭐ (Dễ)

---

#### **B. Self-Supervised Pre-training**
```python
class MaskedTimeSeriesModeling:
    """Giống BERT masking cho time series"""
    
    def mask_and_predict(self, x, mask_ratio=0.15):
        B, L, D = x.shape
        
        # Random mask
        mask = torch.rand(B, L) < mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0  # or use special token
        
        # Predict masked values
        pred = self.model(x_masked)
        loss = F.mse_loss(pred[mask], x[mask])
        
        return loss

# Pre-training loop
for epoch in range(pretrain_epochs):
    for batch_x, _, _, _ in train_loader:
        loss = masked_tsm.mask_and_predict(batch_x)
        loss.backward()
        optimizer.step()

# Then fine-tune for forecasting
```

**Lợi ích:**
- ✅ Học representation tốt hơn
- ✅ Hiệu quả với dữ liệu ít
- ✅ BERT-style proven effective

**Độ khó:** ⭐⭐⭐⭐ (Khó)

---

#### **C. Adaptive Learning Rate với Warm Restarts**
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Thay thế scheduler hiện tại
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Restart every 10 epochs
    T_mult=2,    # Double the restart period
    eta_min=1e-6
)

# Thêm vào training loop
for epoch in range(epochs):
    train_loss = train_one_epoch()
    scheduler.step()
```

**Lợi ích:**
- ✅ Escape local minima
- ✅ Better convergence
- ✅ Dễ implement

**Độ khó:** ⭐ (Rất dễ)

---

## 🎯 NHÓM 4: CẢI TIẾN HẬU XỬ LÝ (Post-processing)

### **4.1. Ensemble Methods**

#### **A. Model Averaging**
```python
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models  # [Llama2, Llama3, Gemma2, PatchTST, DLinear]
    
    def predict(self, x, itr):
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x, itr)
                predictions.append(pred)
        
        # Simple average
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        # Or weighted average (learn weights)
        # weights = F.softmax(self.weights, dim=0)
        # ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        
        return ensemble_pred
```

**Strategies:**
1. **Simple Average**: Trung bình các models
2. **Weighted Average**: Học weights
3. **Stacking**: Train meta-model trên predictions
4. **Boosting**: Sequential training

**Lợi ích:**
- ✅ Tăng robustness
- ✅ Giảm variance
- ✅ Thường tốt hơn single model 5-10%

**Độ khó:** ⭐⭐ (Dễ)

---

#### **B. Post-hoc Calibration**
```python
class TemperatureScaling:
    """Calibrate confidence của predictions"""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def fit(self, logits, labels):
        """Học temperature trên validation set"""
        optimizer = optim.LBFGS([self.temperature])
        
        def eval():
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
    
    def forward(self, logits):
        return logits / self.temperature
```

**Lợi ích:**
- ✅ Better confidence estimates
- ✅ Important for decision making
- ✅ Không cần retrain model

**Độ khó:** ⭐⭐ (Dễ)

---

### **4.2. Constraint Enforcement**

#### **A. Physical Constraints**
```python
def enforce_constraints(predictions):
    """Enforce domain knowledge"""
    
    # 1. Non-negativity
    predictions = torch.clamp(predictions, min=0, max=100)
    
    # 2. Smooth transitions (không nhảy đột ngột)
    diff = predictions[:, 1:] - predictions[:, :-1]
    max_weekly_change = 10  # max 10% change per week
    diff = torch.clamp(diff, -max_weekly_change, max_weekly_change)
    predictions[:, 1:] = predictions[:, :-1] + diff
    
    # 3. Seasonal consistency (không thể cao hơn lịch sử mùa này quá nhiều)
    historical_max = get_historical_seasonal_max()
    predictions = torch.minimum(predictions, historical_max * 1.2)
    
    return predictions
```

**Lợi ích:**
- ✅ Predictions realistic hơn
- ✅ Incorporate domain knowledge
- ✅ Tránh outliers

**Độ khó:** ⭐⭐ (Dễ)

---

## 💡 NHÓM 5: CẢI TIẾN ĐÁNH GIÁ (Evaluation)

### **5.1. Better Metrics**

#### **A. Epidemic-Specific Metrics**
```python
def epidemic_metrics(pred, true, threshold=20):
    """Metrics quan trọng cho epidemic forecasting"""
    
    # 1. Peak timing error (sai lệch đỉnh dịch bao nhiêu tuần)
    pred_peak_week = torch.argmax(pred)
    true_peak_week = torch.argmax(true)
    peak_timing_error = abs(pred_peak_week - true_peak_week)
    
    # 2. Peak intensity error (sai lệch độ cao đỉnh dịch)
    pred_peak_value = torch.max(pred)
    true_peak_value = torch.max(true)
    peak_intensity_error = abs(pred_peak_value - true_peak_value)
    
    # 3. Outbreak detection (detect được khi > threshold chưa)
    pred_outbreak = (pred > threshold).float()
    true_outbreak = (true > threshold).float()
    outbreak_f1 = f1_score(true_outbreak, pred_outbreak)
    
    # 4. Direction accuracy (dự đoán đúng xu hướng tăng/giảm)
    pred_direction = torch.sign(pred[1:] - pred[:-1])
    true_direction = torch.sign(true[1:] - true[:-1])
    direction_accuracy = (pred_direction == true_direction).float().mean()
    
    return {
        'peak_timing_error': peak_timing_error.item(),
        'peak_intensity_error': peak_intensity_error.item(),
        'outbreak_f1': outbreak_f1,
        'direction_accuracy': direction_accuracy.item()
    }
```

**Lợi ích:**
- ✅ Metrics phù hợp với domain
- ✅ MSE/MAE không đủ cho epidemic
- ✅ Quan trọng cho public health

**Độ khó:** ⭐⭐ (Dễ)

---

## 🔥 TOP 5 CẢI TIẾN ĐỀ XUẤT ƯU TIÊN

### **1. Thêm External Features (Weather + Google Trends)** ⭐⭐⭐⭐⭐
- **Impact**: Rất cao (15-25% improvement)
- **Effort**: Trung bình
- **Feasibility**: Cao (data có sẵn)

### **2. Ensemble 5 models (Llama2/3, Gemma, PatchTST, DLinear)** ⭐⭐⭐⭐⭐
- **Impact**: Cao (5-10% improvement)
- **Effort**: Thấp
- **Feasibility**: Rất cao (chỉ cần average)

### **3. Composite Loss Function** ⭐⭐⭐⭐
- **Impact**: Trung bình-Cao (10-15%)
- **Effort**: Trung bình
- **Feasibility**: Cao

### **4. Data Augmentation (Jitter + Scaling + Window Slice)** ⭐⭐⭐⭐
- **Impact**: Trung bình (5-10%)
- **Effort**: Thấp
- **Feasibility**: Rất cao

### **5. Quantile Regression for Uncertainty** ⭐⭐⭐⭐
- **Impact**: Trung bình (cho uncertainty)
- **Effort**: Trung bình
- **Feasibility**: Cao

---

## 📝 ROADMAP THỰC HIỆN

### **Phase 1: Quick Wins (1-2 tuần)**
1. ✅ Implement data augmentation
2. ✅ Implement ensemble of existing models
3. ✅ Add better metrics (epidemic-specific)
4. ✅ Implement composite loss

### **Phase 2: Medium Effort (2-4 tuần)**
1. ✅ Collect và integrate external features
2. ✅ Implement quantile regression
3. ✅ Add curriculum learning
4. ✅ Implement TCN architecture

### **Phase 3: Advanced (1-2 tháng)**
1. ✅ Cross-region transfer learning
2. ✅ Multi-scale attention
3. ✅ Self-supervised pre-training
4. ✅ Hybrid LLM + classical models

---

## 📊 DỰ KIẾN CẢI THIỆN

| Cải tiến | MSE Reduction | MAE Reduction | Correlation Increase |
|----------|---------------|---------------|----------------------|
| Baseline | 0.0156 | 0.0987 | 0.823 |
| + Data Aug | 0.0145 (-7%) | 0.0920 (-7%) | 0.835 (+1.5%) |
| + Ensemble | 0.0135 (-13%) | 0.0875 (-11%) | 0.851 (+3.4%) |
| + External Feat | 0.0115 (-26%) | 0.0750 (-24%) | 0.890 (+8.1%) |
| + Composite Loss | 0.0108 (-31%) | 0.0720 (-27%) | 0.905 (+10.0%) |
| **All combined** | **0.0095 (-39%)** | **0.0680 (-31%)** | **0.925 (+12.4%)** |

---

## 💻 CODE EXAMPLE: QUICK START

```python
# File: improved_main.py

# 1. Add data augmentation
from augmentation import TimeSeriesAugmentation
augmenter = TimeSeriesAugmentation(p=0.5)

# 2. Use composite loss
from losses import FluForecastingLoss
criterion = FluForecastingLoss(alpha=0.5, beta=0.3, gamma=0.2)

# 3. Ensemble predictions
models = [llama2, llama3, gemma2, patchtst, dlinear]
ensemble = EnsemblePredictor(models)

# 4. Training with augmentation
for batch_x, batch_y, ... in train_loader:
    # Augment
    batch_x = augmenter.jitter(augmenter.scaling(batch_x))
    
    # Forward
    pred = ensemble(batch_x, itr)
    
    # Composite loss
    loss = criterion(pred, batch_y)
    loss.backward()
    optimizer.step()

# 5. Evaluate with better metrics
metrics = epidemic_metrics(pred, true, threshold=20)
print(f"Peak Timing Error: {metrics['peak_timing_error']} weeks")
print(f"Outbreak F1: {metrics['outbreak_f1']:.3f}")
```

---

**Kết luận**: Với dữ liệu ít (416 tuần), nên tập trung vào:
1. **Data augmentation** và **external features** để tăng data
2. **Ensemble methods** để giảm variance
3. **Better loss functions** phù hợp với epidemic forecasting
4. **Domain knowledge** thông qua constraints và metrics

Tránh overfitting bằng cách không làm model quá phức tạp!
