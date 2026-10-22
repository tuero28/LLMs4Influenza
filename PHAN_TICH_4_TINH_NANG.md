# PHÂN TÍCH CHI TIẾT 4 TÍNH NĂNG CẢI TIẾN

## 📋 TÓM TẮT NHANH

| Tính năng | Độ khó | Ưu tiên | Rủi ro | Thời gian | Tác động |
|-----------|---------|---------|--------|-----------|----------|
| 1. Epidemic Metrics | ⭐ Rất dễ | 🔥 CAO | ✅ Không | 2-3 giờ | 📊 Hiểu rõ hơn |
| 2. Hậu xử lý không âm | ⭐ Rất dễ | 🔥 CAO | ✅ Không | 1 giờ | 🎯 Predictions hợp lý |
| 3. Cosine Warm Restarts | ⭐⭐ Dễ | 🟡 TRUNG BÌNH | ⚠️ Có | 2-3 giờ | 🎲 Không chắc chắn |
| 4. Temporal Features | ⭐⭐⭐⭐ KHÓ | 🟢 THẤP | ⚠️⚠️ CÓ | 1-2 ngày | 🚀 Tiềm năng cao |

---

## 1️⃣ EPIDEMIC METRICS (TÍNH NĂNG ĐẦU TIÊN - ƯU TIÊN CAO)

### 📊 Mô tả
Thêm các metrics chuyên biệt cho dự báo dịch bệnh, thay vì chỉ dùng MSE/MAE chung chung.

### ✅ VÌ SAO DỄ?
1. **Không thay đổi model**: Chỉ tính toán thêm trên kết quả dự báo đã có
2. **Không cần retrain**: Dùng ngay với predictions hiện tại
3. **Logic đơn giản**: Chỉ là các phép tính trên numpy arrays
4. **Không ảnh hưởng kết quả cũ**: Không đổi cách train hay test

### 🎯 METRICS ĐỀ XUẤT

#### **A. Peak Timing Error (PTError)**
```python
def peak_timing_error(pred, true):
    """
    Tính sai lệch tuần đạt đỉnh dịch.
    
    Args:
        pred: [B, T] hoặc [T] - Dự báo
        true: [B, T] hoặc [T] - Thực tế
    
    Returns:
        error: Số tuần sai lệch (càng nhỏ càng tốt)
    
    Example:
        true đạt đỉnh ở tuần 5, pred đạt đỉnh ở tuần 7 → PTError = 2 tuần
    """
    pred_peak_idx = np.argmax(pred, axis=-1)
    true_peak_idx = np.argmax(true, axis=-1)
    
    error = np.abs(pred_peak_idx - true_peak_idx)
    return np.mean(error)  # Trung bình nếu có nhiều samples
```

**Ý nghĩa**: 
- Quan trọng nhất cho public health: biết khi nào đỉnh dịch để chuẩn bị
- MSE/MAE không phản ánh được sai lệch timing

**Threshold tốt**: 
- Xuất sắc: ≤ 1 tuần
- Tốt: ≤ 2 tuần
- Chấp nhận được: ≤ 3 tuần
- Kém: > 3 tuần

---

#### **B. Peak Intensity Error (PIError)**
```python
def peak_intensity_error(pred, true):
    """
    Tính sai lệch độ cao đỉnh dịch (%).
    
    Example:
        true peak = 30%, pred peak = 35% → PIError = 5%
    """
    pred_peak_value = np.max(pred, axis=-1)
    true_peak_value = np.max(true, axis=-1)
    
    # Absolute error
    abs_error = np.abs(pred_peak_value - true_peak_value)
    
    # Relative error (%)
    rel_error = abs_error / (true_peak_value + 1e-8) * 100
    
    return np.mean(abs_error), np.mean(rel_error)
```

**Ý nghĩa**:
- Biết đỉnh dịch cao bao nhiêu để phân bổ nguồn lực y tế
- Dự báo quá cao → lãng phí; quá thấp → thiếu hụt

**Threshold tốt**:
- Xuất sắc: ≤ 2% absolute error
- Tốt: ≤ 5% absolute error
- Chấp nhận được: ≤ 10% absolute error

---

#### **C. Direction Accuracy (DirAcc)**
```python
def direction_accuracy(pred, true):
    """
    Tính độ chính xác dự đoán hướng tăng/giảm.
    
    Example:
        Tuần này: 20%, tuần sau: 25% (tăng) → True direction = +1
        Nếu pred cũng tăng → Đúng
    """
    # Calculate differences (tăng/giảm từ tuần này sang tuần sau)
    pred_diff = pred[..., 1:] - pred[..., :-1]
    true_diff = true[..., 1:] - true[..., :-1]
    
    # Sign: +1 (tăng), -1 (giảm), 0 (không đổi)
    pred_direction = np.sign(pred_diff)
    true_direction = np.sign(true_diff)
    
    # Accuracy
    correct = (pred_direction == true_direction)
    accuracy = np.mean(correct) * 100  # Percentage
    
    return accuracy
```

**Ý nghĩa**:
- Quan trọng cho early warning: dịch đang tăng hay giảm?
- Hữu ích khi giá trị tuyệt đối sai nhưng trend đúng

**Threshold tốt**:
- Xuất sắc: ≥ 80%
- Tốt: ≥ 70%
- Chấp nhận được: ≥ 60%
- Kém: < 60%

---

#### **D. KHÔNG NÊN: Outbreak F1 với ngưỡng tự đặt**
```python
# ❌ KHÔNG LÀM NHƯ NÀY - Ngưỡng 20% không có cơ sở
def outbreak_detection_WRONG(pred, true, threshold=20):
    pred_outbreak = (pred > threshold).astype(float)
    true_outbreak = (true > threshold).astype(float)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(true_outbreak.flatten(), pred_outbreak.flatten())
    return f1
```

**Vấn đề**:
- Ngưỡng 20% là tự đặt, không có cơ sở khoa học
- Ngưỡng khác nhau cho từng khu vực (Bắc TQ khác Nam TQ khác USA)
- F1 score nhạy cảm với ngưỡng

**✅ CÁCH TỐT HƠN: Học ngưỡng từ dữ liệu**
```python
def learn_outbreak_threshold_from_train(train_data, percentile=75):
    """
    Học ngưỡng outbreak từ dữ liệu train.
    
    Args:
        train_data: Dữ liệu train [N, T]
        percentile: Phân vị để định nghĩa "outbreak" (mặc định 75%)
    
    Returns:
        threshold: Ngưỡng outbreak cho region này
    """
    # Ngưỡng = percentile 75% của tất cả giá trị trong train
    threshold = np.percentile(train_data, percentile)
    
    print(f"Learned outbreak threshold: {threshold:.2f}%")
    return threshold

# Sau đó mới tính F1
def outbreak_detection_CORRECT(pred, true, train_data):
    threshold = learn_outbreak_threshold_from_train(train_data, percentile=75)
    
    pred_outbreak = (pred > threshold).astype(float)
    true_outbreak = (true > threshold).astype(float)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(true_outbreak.flatten(), pred_outbreak.flatten())
    
    return f1, threshold
```

---

### 💻 CODE HOÀN CHỈNH

```python
# File: utils/epidemic_metrics.py

import numpy as np
import os
from sklearn.metrics import f1_score


def peak_timing_error(pred, true):
    """Sai lệch tuần đạt đỉnh."""
    pred_peak_idx = np.argmax(pred, axis=-1)
    true_peak_idx = np.argmax(true, axis=-1)
    error = np.abs(pred_peak_idx - true_peak_idx)
    return np.mean(error)


def peak_intensity_error(pred, true):
    """Sai lệch độ cao đỉnh."""
    pred_peak_value = np.max(pred, axis=-1)
    true_peak_value = np.max(true, axis=-1)
    
    abs_error = np.abs(pred_peak_value - true_peak_value)
    rel_error = abs_error / (true_peak_value + 1e-8) * 100
    
    return np.mean(abs_error), np.mean(rel_error)


def direction_accuracy(pred, true):
    """Độ chính xác dự đoán hướng tăng/giảm."""
    pred_diff = pred[..., 1:] - pred[..., :-1]
    true_diff = true[..., 1:] - true[..., :-1]
    
    pred_direction = np.sign(pred_diff)
    true_direction = np.sign(true_diff)
    
    correct = (pred_direction == true_direction)
    accuracy = np.mean(correct) * 100
    
    return accuracy


def learn_outbreak_threshold(train_data, percentile=75):
    """Học ngưỡng outbreak từ train data."""
    if train_data is None or len(train_data) == 0:
        raise ValueError("Train data is required to learn outbreak threshold!")
    
    threshold = np.percentile(train_data, percentile)
    return threshold


def outbreak_detection_f1(pred, true, train_data=None, threshold=None):
    """
    Outbreak detection F1 score.
    
    Args:
        pred: Predictions
        true: Ground truth
        train_data: Train data để học ngưỡng (ưu tiên)
        threshold: Ngưỡng cố định (chỉ dùng nếu không có train_data)
    
    Returns:
        f1: F1 score
        threshold: Ngưỡng đã dùng
    """
    if threshold is None:
        if train_data is not None:
            threshold = learn_outbreak_threshold(train_data, percentile=75)
        else:
            raise ValueError(
                "Must provide either 'train_data' to learn threshold "
                "or 'threshold' value!"
            )
    
    pred_outbreak = (pred > threshold).astype(float)
    true_outbreak = (true > threshold).astype(float)
    
    f1 = f1_score(true_outbreak.flatten(), pred_outbreak.flatten(), zero_division=0)
    
    return f1, threshold


def epidemic_metrics(pred, true, train_data=None, pred_file_path=None):
    """
    Tính toàn bộ epidemic metrics.
    
    Args:
        pred: Predictions [B, T] hoặc [T]
        true: Ground truth [B, T] hoặc [T]
        train_data: Train data để học outbreak threshold (optional)
        pred_file_path: Path đến file prediction (để kiểm tra)
    
    Returns:
        dict: Dictionary chứa tất cả metrics
    
    Raises:
        FileNotFoundError: Nếu pred_file_path được cung cấp nhưng không tồn tại
    """
    # Kiểm tra file tồn tại nếu được cung cấp
    if pred_file_path is not None:
        if not os.path.exists(pred_file_path):
            raise FileNotFoundError(
                f"Prediction file not found: {pred_file_path}\n"
                "Cannot compute epidemic metrics without predictions!"
            )
    
    # Ensure 2D arrays
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if true.ndim == 1:
        true = true.reshape(1, -1)
    
    # Compute metrics
    pt_error = peak_timing_error(pred, true)
    pi_abs_error, pi_rel_error = peak_intensity_error(pred, true)
    dir_acc = direction_accuracy(pred, true)
    
    results = {
        'peak_timing_error_weeks': float(pt_error),
        'peak_intensity_abs_error': float(pi_abs_error),
        'peak_intensity_rel_error_pct': float(pi_rel_error),
        'direction_accuracy_pct': float(dir_acc),
    }
    
    # Outbreak F1 (nếu có train data)
    if train_data is not None:
        try:
            f1, threshold = outbreak_detection_f1(pred, true, train_data=train_data)
            results['outbreak_f1'] = float(f1)
            results['outbreak_threshold'] = float(threshold)
        except Exception as e:
            print(f"Warning: Could not compute outbreak F1: {e}")
    
    return results


def print_epidemic_metrics(metrics):
    """In metrics đẹp."""
    print("\n" + "="*60)
    print("EPIDEMIC-SPECIFIC METRICS")
    print("="*60)
    
    print(f"\n📍 Peak Timing Error: {metrics['peak_timing_error_weeks']:.2f} weeks")
    if metrics['peak_timing_error_weeks'] <= 1:
        print("   ✅ Xuất sắc!")
    elif metrics['peak_timing_error_weeks'] <= 2:
        print("   ✓ Tốt")
    elif metrics['peak_timing_error_weeks'] <= 3:
        print("   ~ Chấp nhận được")
    else:
        print("   ❌ Cần cải thiện")
    
    print(f"\n📊 Peak Intensity Error:")
    print(f"   Absolute: {metrics['peak_intensity_abs_error']:.2f}%")
    print(f"   Relative: {metrics['peak_intensity_rel_error_pct']:.2f}%")
    if metrics['peak_intensity_abs_error'] <= 2:
        print("   ✅ Xuất sắc!")
    elif metrics['peak_intensity_abs_error'] <= 5:
        print("   ✓ Tốt")
    else:
        print("   ❌ Cần cải thiện")
    
    print(f"\n↗️ Direction Accuracy: {metrics['direction_accuracy_pct']:.2f}%")
    if metrics['direction_accuracy_pct'] >= 80:
        print("   ✅ Xuất sắc!")
    elif metrics['direction_accuracy_pct'] >= 70:
        print("   ✓ Tốt")
    elif metrics['direction_accuracy_pct'] >= 60:
        print("   ~ Chấp nhận được")
    else:
        print("   ❌ Cần cải thiện")
    
    if 'outbreak_f1' in metrics:
        print(f"\n🔔 Outbreak Detection F1: {metrics['outbreak_f1']:.4f}")
        print(f"   (threshold: {metrics['outbreak_threshold']:.2f}%)")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Test với dữ liệu giả
    np.random.seed(42)
    
    # Tạo dữ liệu giả: 10 samples, 13 weeks
    true = np.random.rand(10, 13) * 30 + 10  # 10-40%
    pred = true + np.random.randn(10, 13) * 2  # Thêm noise
    
    # Train data để học threshold
    train_data = np.random.rand(100, 52) * 30 + 10
    
    # Tính metrics
    metrics = epidemic_metrics(pred, true, train_data=train_data)
    
    # In kết quả
    print_epidemic_metrics(metrics)
```

---

### 📝 CÁCH SỬ DỤNG TRONG DỰ ÁN

#### **Bước 1: Thêm vào file testing**
```python
# Trong file exp/exp_main.py hoặc tương tự

from utils.epidemic_metrics import epidemic_metrics, print_epidemic_metrics

def test(self, setting, test=0):
    # ... existing code ...
    
    # Sau khi có preds và trues
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    
    # Standard metrics
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    
    # 🆕 Epidemic metrics
    try:
        epidemic_metrics_results = epidemic_metrics(
            pred=preds,
            true=trues,
            train_data=self.train_data,  # Cần lưu train_data trong __init__
            pred_file_path=folder_path + 'pred.npy'
        )
        
        print_epidemic_metrics(epidemic_metrics_results)
        
        # Lưu vào file
        np.save(folder_path + 'epidemic_metrics.npy', epidemic_metrics_results)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Skipping epidemic metrics.")
    except Exception as e:
        print(f"⚠️ Warning: Could not compute epidemic metrics: {e}")
    
    # ... existing code ...
```

#### **Bước 2: Lưu train_data trong Exp_Main**
```python
# Trong __init__ của Exp_Main class

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        
        # 🆕 Lưu train data để tính outbreak threshold
        self.train_data = None
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        
        # 🆕 Lưu train data
        if flag == 'train':
            # Lấy tất cả train data
            self.train_data = data_set.data_x  # hoặc data_set.data tùy implementation
        
        return data_set, data_loader
```

---

### 🎯 KẾT QUẢ MONG ĐỢI

Sau khi implement, bạn sẽ có output như:

```
==============================================================
EPIDEMIC-SPECIFIC METRICS
==============================================================

📍 Peak Timing Error: 1.35 weeks
   ✓ Tốt

📊 Peak Intensity Error:
   Absolute: 2.87%
   Relative: 9.45%
   ✓ Tốt

↗️ Direction Accuracy: 76.92%
   ✓ Tốt

🔔 Outbreak Detection F1: 0.8571
   (threshold: 24.32%)
==============================================================

Standard Metrics:
mae:0.0987, mse:0.0156, rmse:0.1249
```

---

### ✅ CHECKLIST IMPLEMENTATION

- [ ] Tạo file `utils/epidemic_metrics.py`
- [ ] Test functions riêng lẻ với dữ liệu giả
- [ ] Thêm vào exp_main.py
- [ ] Lưu train_data trong Exp_Main.__init__
- [ ] Chạy test và kiểm tra output
- [ ] Lưu epidemic metrics vào file .npy
- [ ] So sánh metrics giữa các models (Llama2 vs PatchTST vs DLinear)

---

## 2️⃣ HẬU XỬ LÝ KHÔNG ÂM (TÍNH NĂNG THỨ HAI)

### 📊 Mô tả
Clamp predictions về [0, +∞) trước khi tính metrics.

### ✅ VÌ SAO DỀ?
- Chỉ 1 dòng code: `pred = np.clip(pred, 0, None)`
- Không cần retrain
- Không ảnh hưởng quá trình training

### 🎯 IMPLEMENTATION

```python
# File: utils/post_processing.py

import numpy as np
import torch


def clip_predictions(pred, min_value=0.0, max_value=None):
    """
    Clip predictions để đảm bảo trong khoảng hợp lý.
    
    Args:
        pred: Predictions (numpy array hoặc torch tensor)
        min_value: Giá trị min (mặc định 0 - không âm)
        max_value: Giá trị max (mặc định None - không giới hạn trên)
    
    Returns:
        pred_clipped: Predictions sau khi clip
    
    ⚠️ LƯU Ý: 
    - BỎ max=100: Có khu vực có thể > 100% (đếm nhiều subtypes)
    - BỎ giới hạn 10%/tuần: Quá rigid, có thể sai
    - BỎ "seasonal max": Khó định nghĩa, dễ sai
    """
    is_torch = isinstance(pred, torch.Tensor)
    
    if is_torch:
        if max_value is not None:
            return torch.clamp(pred, min=min_value, max=max_value)
        else:
            return torch.clamp(pred, min=min_value)
    else:
        return np.clip(pred, min_value, max_value)


def analyze_clipping_impact(pred_original, pred_clipped, true):
    """
    Phân tích tác động của clipping lên metrics.
    
    Returns:
        dict: Metrics trước và sau clip
    """
    from utils.metrics import metric
    
    # Metrics trước clip
    mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig, smape_orig, nd_orig = \
        metric(pred_original, true)
    
    # Metrics sau clip
    mae_clip, mse_clip, rmse_clip, mape_clip, mspe_clip, smape_clip, nd_clip = \
        metric(pred_clipped, true)
    
    # So sánh
    results = {
        'before_clip': {
            'mae': mae_orig,
            'mse': mse_orig,
            'rmse': rmse_orig,
        },
        'after_clip': {
            'mae': mae_clip,
            'mse': mse_clip,
            'rmse': rmse_clip,
        },
        'improvement': {
            'mae': mae_orig - mae_clip,
            'mse': mse_orig - mse_clip,
            'rmse': rmse_orig - rmse_clip,
        },
        'n_negative_values': int(np.sum(pred_original < 0)),
        'min_value_before': float(np.min(pred_original)),
        'n_clipped': int(np.sum(pred_original != pred_clipped)),
    }
    
    return results


def print_clipping_analysis(analysis):
    """In phân tích clipping đẹp."""
    print("\n" + "="*60)
    print("CLIPPING ANALYSIS")
    print("="*60)
    
    print(f"\n🔍 Detected {analysis['n_negative_values']} negative predictions")
    print(f"   Min value: {analysis['min_value_before']:.4f}")
    print(f"   Total clipped: {analysis['n_clipped']} values")
    
    print("\n📊 Metrics Comparison:")
    print(f"   MAE:  {analysis['before_clip']['mae']:.6f} → {analysis['after_clip']['mae']:.6f} "
          f"(Δ {analysis['improvement']['mae']:+.6f})")
    print(f"   MSE:  {analysis['before_clip']['mse']:.6f} → {analysis['after_clip']['mse']:.6f} "
          f"(Δ {analysis['improvement']['mse']:+.6f})")
    print(f"   RMSE: {analysis['before_clip']['rmse']:.6f} → {analysis['after_clip']['rmse']:.6f} "
          f"(Δ {analysis['improvement']['rmse']:+.6f})")
    
    if analysis['n_negative_values'] > 0:
        if analysis['improvement']['mae'] < 0:
            print("\n✅ Clipping improved metrics!")
        else:
            print("\n⚠️ Clipping made metrics slightly worse (but predictions more realistic)")
    else:
        print("\n✓ No negative values found - clipping has no effect")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Test
    np.random.seed(42)
    
    # Predictions with some negative values
    pred = np.random.randn(10, 13) * 5 + 20  # Some may be negative
    true = np.random.rand(10, 13) * 30 + 10  # Always positive
    
    print(f"Original predictions range: [{pred.min():.2f}, {pred.max():.2f}]")
    
    # Clip
    pred_clipped = clip_predictions(pred, min_value=0.0)
    
    print(f"Clipped predictions range: [{pred_clipped.min():.2f}, {pred_clipped.max():.2f}]")
    
    # Analyze
    analysis = analyze_clipping_impact(pred, pred_clipped, true)
    print_clipping_analysis(analysis)
```

### 📝 CÁCH TÍCH HỢP VÀO DỰ ÁN

```python
# Trong exp/exp_main.py

from utils.post_processing import (
    clip_predictions, 
    analyze_clipping_impact, 
    print_clipping_analysis
)

def test(self, setting, test=0):
    # ... existing code to get preds and trues ...
    
    # 🆕 Analyze clipping
    pred_original = preds.copy()
    pred_clipped = clip_predictions(preds, min_value=0.0)
    
    clipping_analysis = analyze_clipping_impact(
        pred_original, pred_clipped, trues
    )
    print_clipping_analysis(clipping_analysis)
    
    # 🆕 Compute metrics cho CẢ HAI
    print("\n" + "="*60)
    print("METRICS - BEFORE CLIPPING")
    print("="*60)
    mae, mse, rmse, mape, mspe, smape, nd = metric(pred_original, trues)
    print(f'mae:{mae:.6f}, mse:{mse:.6f}, rmse:{rmse:.6f}')
    
    print("\n" + "="*60)
    print("METRICS - AFTER CLIPPING (RECOMMENDED)")
    print("="*60)
    mae, mse, rmse, mape, mspe, smape, nd = metric(pred_clipped, trues)
    print(f'mae:{mae:.6f}, mse:{mse:.6f}, rmse:{rmse:.6f}')
    
    # 🆕 Lưu cả hai versions
    np.save(folder_path + 'pred_original.npy', pred_original)
    np.save(folder_path + 'pred_clipped.npy', pred_clipped)
    np.save(folder_path + 'clipping_analysis.npy', clipping_analysis)
    
    # Use clipped version cho epidemic metrics
    epidemic_metrics_results = epidemic_metrics(
        pred=pred_clipped,  # Dùng clipped version
        true=trues,
        train_data=self.train_data
    )
```

---

## 3️⃣ COSINE WARM RESTARTS (TÍNH NĂNG THỨ BA)

### 📊 Mô tả
Thay learning rate scheduler hiện tại bằng `CosineAnnealingWarmRestarts`.

### ⚠️ LƯU Ý QUAN TRỌNG

**ĐÂY LÀ THAY ĐỔI TRAINING → CÓ RỦI RO!**

1. **Phải chạy riêng**: Không trộn với augmentation, ensemble
2. **Phải fixed seed**: So sánh công bằng với baseline
3. **Có thể không cải thiện**: Tùy dataset và model
4. **Cần validation**: Kiểm tra trên val set trước

### ✅ VÌ SAO TƯƠ��G ĐỐI DỄ?
- Chỉ thay 2-3 dòng code
- PyTorch có sẵn
- Không cần sửa model architecture

### 🎯 IMPLEMENTATION

```python
# File: exp/exp_main.py

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def _select_optimizer(self):
    model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    return model_optim

def _select_scheduler(self, optimizer):
    """
    Select learning rate scheduler.
    
    Options:
    - 'cosine_warm_restarts': CosineAnnealingWarmRestarts (NEW)
    - 'reduce_on_plateau': ReduceLROnPlateau (OLD)
    - 'step': StepLR
    - None: No scheduler
    """
    scheduler_type = getattr(self.args, 'scheduler', 'cosine_warm_restarts')
    
    if scheduler_type == 'cosine_warm_restarts':
        # Cosine Annealing with Warm Restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.args.cosine_T0,        # Restart every T_0 epochs
            T_mult=self.args.cosine_Tmult,  # Multiply T_0 after each restart
            eta_min=self.args.learning_rate * 0.01,  # Min LR = 1% of initial
            last_epoch=-1
        )
        print(f"✓ Using CosineAnnealingWarmRestarts(T_0={self.args.cosine_T0}, T_mult={self.args.cosine_Tmult})")
        
    elif scheduler_type == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        print("✓ Using ReduceLROnPlateau")
        
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        print("✓ Using StepLR")
        
    else:
        scheduler = None
        print("✓ No scheduler")
    
    return scheduler

def train(self, setting):
    # ... existing code ...
    
    model_optim = self._select_optimizer()
    scheduler = self._select_scheduler(model_optim)  # 🆕
    
    for epoch in range(self.args.train_epochs):
        train_loss = []
        self.model.train()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # ... training code ...
            train_loss.append(loss.item())
        
        # ... validation ...
        
        # 🆕 Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(vali_loss)
            else:
                scheduler.step()  # CosineAnnealingWarmRestarts
        
        # 🆕 Log current LR
        current_lr = model_optim.param_groups[0]['lr']
        print(f"Epoch {epoch} | LR: {current_lr:.6f}")
```

### 📝 THÊM ARGUMENTS

```python
# File: run.py hoặc argparse setup

parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts',
                    help='LR scheduler: cosine_warm_restarts, reduce_on_plateau, step, none')
parser.add_argument('--cosine_T0', type=int, default=10,
                    help='Restart interval for CosineAnnealingWarmRestarts')
parser.add_argument('--cosine_Tmult', type=int, default=2,
                    help='Multiply factor for restart interval')
```

### 🧪 CÁCH THỰC NGHIỆM ĐÚNG

```bash
# Baseline (không có warm restarts)
python main.py \
  --model Llama2 \
  --data flu \
  --data_path dataset/Flu_in_NorthChina.csv \
  --scheduler none \
  --seed 2024 \
  --itr 3

# With warm restarts (CÙNG SEED)
python main.py \
  --model Llama2 \
  --data flu \
  --data_path dataset/Flu_in_NorthChina.csv \
  --scheduler cosine_warm_restarts \
  --cosine_T0 10 \
  --cosine_Tmult 2 \
  --seed 2024 \
  --itr 3

# So sánh kết quả
python compare_results.py --baseline results_baseline --warmrestarts results_warmrestarts
```

### 📊 HYPERPARAMETER TUNING

| T_0 | T_mult | Ý nghĩa | Khi nào dùng |
|-----|--------|---------|--------------|
| 5 | 1 | Restart mỗi 5 epochs | Dataset nhỏ, train nhanh |
| 10 | 2 | Restart 10, 30, 70... epochs | **Khuyến nghị cho flu data** |
| 20 | 2 | Restart 20, 60, 140... epochs | Dataset lớn, train lâu |
| 10 | 1 | Restart mỗi 10 epochs | Muốn restart thường xuyên |

### ⚠️ RỦI RO VÀ GIẢI PHÁP

| Rủi ro | Giải pháp |
|---------|-----------|
| Không cải thiện | So sánh với baseline, giữ cái tốt hơn |
| Training unstable | Giảm learning rate ban đầu |
| Overfitting | Tăng regularization (dropout, weight decay) |
| Restart quá sớm | Tăng T_0 (từ 5 → 10 → 20) |
| Không converge | Giảm T_mult hoặc tăng epochs |

---

## 4️⃣ TEMPORAL FEATURES (FOURIER/CYCLIC) - KHÓ NHẤT

### 📊 Mô tả
Thêm các features theo thời gian: sin/cos của tuần trong năm, tháng, v.v.

### ⚠️ VÌ SAO KHÓ?

1. **Phải kiểm tra input shape hiện tại**
   - Model hiện nhận bao nhiêu channels?
   - Có hỗ trợ multivariate không?

2. **Có thể phải sửa model architecture**
   - Embedding layer
   - Input projection
   - Positional encoding

3. **Phải sửa data loader**
   - Đọc và parse datetime
   - Tạo thêm features
   - Concatenate đúng axis

4. **Risk cao**
   - Sai 1 chỗ → Toàn bộ pipeline lỗi
   - Phải retrain từ đầu
   - Có thể không cải thiện

### 🔍 BƯỚC 1: KIỂM TRA HIỆN TRẠNG

```python
# File: check_current_setup.py

import torch
from data_provider.data_factory import data_provider
from models import Llama2, PatchTST, DLinear

def check_input_features():
    """Kiểm tra số features đầu vào hiện tại."""
    
    # Mock args
    class Args:
        model = 'Llama2'
        data = 'custom'
        root_path = './dataset/'
        data_path = 'Flu_in_NorthChina.csv'
        features = 'S'  # univariate
        target = 'positive_rate'
        freq = 'w'
        seq_len = 336
        label_len = 18
        pred_len = 8
        batch_size = 32
        
    args = Args()
    
    # Get data
    train_data, train_loader = data_provider(args, 'train')
    
    # Get one batch
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        print("="*60)
        print("CURRENT INPUT SHAPE")
        print("="*60)
        print(f"batch_x shape: {batch_x.shape}")  # [B, T, C]
        print(f"batch_y shape: {batch_y.shape}")
        print(f"batch_x_mark shape: {batch_x_mark.shape}")  # Time features
        print(f"batch_y_mark shape: {batch_y_mark.shape}")
        print()
        print(f"Number of input features (C): {batch_x.shape[-1]}")
        print(f"Number of time features: {batch_x_mark.shape[-1]}")
        print()
        
        # Kiểm tra time features hiện có
        print("Current time features in batch_x_mark:")
        print(batch_x_mark[0, :5])  # First 5 timesteps
        
        break
    
    # Kiểm tra model
    print("\n" + "="*60)
    print("MODEL INPUT EXPECTATION")
    print("="*60)
    
    # Llama2
    from models.Llama2 import Model as Llama2
    model = Llama2(args)
    try:
        out = model(batch_x, batch_x_mark, None, None, 0)
        print(f"✓ Llama2 accepts input: {batch_x.shape}")
        print(f"  Output shape: {out.shape}")
    except Exception as e:
        print(f"✗ Llama2 error: {e}")
    
    # PatchTST
    try:
        from models.PatchTST import Model as PatchTST
        model = PatchTST(args)
        out = model(batch_x, batch_x_mark, None, None, 0)
        print(f"✓ PatchTST accepts input: {batch_x.shape}")
        print(f"  Output shape: {out.shape}")
    except Exception as e:
        print(f"✗ PatchTST error: {e}")
    
    # Kiểm tra có hỗ trợ multivariate không
    print("\n" + "="*60)
    print("MULTIVARIATE SUPPORT CHECK")
    print("="*60)
    
    # Tạo input với nhiều features
    batch_x_multi = torch.randn(32, 336, 5)  # 5 features
    
    try:
        args.enc_in = 5
        model = Llama2(args)
        out = model(batch_x_multi, batch_x_mark, None, None, 0)
        print(f"✓ Llama2 supports multivariate: {batch_x_multi.shape} → {out.shape}")
    except Exception as e:
        print(f"✗ Llama2 multivariate error: {e}")
        print("  → PHẢI SỬA EMBEDDING LAYER!")

if __name__ == "__main__":
    check_input_features()
```

**Chạy script này TRƯỚC KHI QUYẾT ĐỊNH triển khai temporal features!**

### 📝 KẾT QUẢ DỰ KIẾN

```
==============================================================
CURRENT INPUT SHAPE
==============================================================
batch_x shape: torch.Size([32, 336, 1])
batch_y shape: torch.Size([32, 26, 1])
batch_x_mark shape: torch.Size([32, 336, 4])
batch_y_mark shape: torch.Size([32, 26, 4])

Number of input features (C): 1
Number of time features: 4

Current time features in batch_x_mark:
tensor([[0.0000, 0.5000, 0.2500, 0.0000],
        [0.0192, 0.5000, 0.2500, 0.0000],
        ...])
```

**Phân tích**:
- ✅ `batch_x`: 1 feature (chỉ có positive_rate)
- ✅ `batch_x_mark`: 4 features (month, day, weekday, hour - nhưng không dùng)
- ⚠️ Model hiện CHỈ nhận 1 feature → Cần kiểm tra có thể thêm không

### 🎯 IMPLEMENTATION (NẾU ĐƯỢC)

```python
# File: data_provider/temporal_features.py

import numpy as np
import pandas as pd


def create_fourier_features(dates, n_harmonics=4, period=52):
    """
    Tạo Fourier features từ dates.
    
    Args:
        dates: Array of datetime objects
        n_harmonics: Số harmonics (4 → 8 features: 4 sin + 4 cos)
        period: Chu kỳ (52 weeks cho flu)
    
    Returns:
        fourier_features: [T, n_harmonics*2] array
    """
    # Convert to week of year
    if isinstance(dates, pd.DatetimeIndex):
        week_of_year = dates.isocalendar().week.values
    else:
        week_of_year = pd.DatetimeIndex(dates).isocalendar().week.values
    
    features = []
    for k in range(1, n_harmonics + 1):
        # Sin and cos for each harmonic
        sin_k = np.sin(2 * np.pi * k * week_of_year / period)
        cos_k = np.cos(2 * np.pi * k * week_of_year / period)
        features.extend([sin_k, cos_k])
    
    fourier_features = np.stack(features, axis=-1)  # [T, n_harmonics*2]
    return fourier_features


def create_cyclic_features(dates):
    """
    Tạo cyclic encoding cho temporal features.
    
    Returns:
        cyclic_features: [T, 6] array
          - week_sin, week_cos (weekly cycle)
          - month_sin, month_cos (monthly cycle)
          - quarter_sin, quarter_cos (quarterly cycle)
    """
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    
    # Week of year (1-52)
    week = dates.isocalendar().week.values
    week_sin = np.sin(2 * np.pi * week / 52)
    week_cos = np.cos(2 * np.pi * week / 52)
    
    # Month (1-12)
    month = dates.month.values
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Quarter (1-4)
    quarter = dates.quarter.values
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)
    
    cyclic_features = np.stack([
        week_sin, week_cos,
        month_sin, month_cos,
        quarter_sin, quarter_cos
    ], axis=-1)  # [T, 6]
    
    return cyclic_features


def create_all_temporal_features(dates, include_fourier=True, include_cyclic=True, 
                                  n_harmonics=4):
    """
    Tạo tất cả temporal features.
    
    Returns:
        features: [T, D] array where D depends on options
    """
    all_features = []
    
    if include_fourier:
        fourier = create_fourier_features(dates, n_harmonics=n_harmonics)
        all_features.append(fourier)
        print(f"✓ Added {fourier.shape[-1]} Fourier features")
    
    if include_cyclic:
        cyclic = create_cyclic_features(dates)
        all_features.append(cyclic)
        print(f"✓ Added {cyclic.shape[-1]} cyclic features")
    
    if all_features:
        features = np.concatenate(all_features, axis=-1)
        print(f"→ Total temporal features: {features.shape[-1]}")
        return features
    else:
        return None


# Example
if __name__ == "__main__":
    # Test
    dates = pd.date_range('2011-01-01', periods=104, freq='W')
    
    features = create_all_temporal_features(
        dates, 
        include_fourier=True, 
        include_cyclic=True,
        n_harmonics=4
    )
    
    print(f"\nFeature shape: {features.shape}")
    print(f"Sample features (first 5 timesteps):")
    print(features[:5])
```

### ⚠️ QUYẾT ĐỊNH TRIỂN KHAI

**TRƯỚC KHI IMPLEMENT, CHẠY:**

```bash
python check_current_setup.py
```

**NẾU KẾT QUẢ LÀ:**
- ✅ Model hỗ trợ multivariate → Có thể thêm temporal features
- ❌ Model CHỈ hỗ trợ univariate → **KHÔNG NÊN LÀM** (phải sửa model)

**NẾU QUYẾT ĐỊNH KHÔNG LÀM:**
- Temporal features đã CÓ trong `batch_x_mark` rồi
- Model có thể tự học seasonality từ data
- Rủi ro cao, lợi ích không chắc chắn

---

## 📊 BẢNG QUYẾT ĐỊNH CUỐI CÙNG

| Tính năng | Làm không? | Khi nào làm? |
|-----------|------------|--------------|
| **1. Epidemic Metrics** | ✅ **LÀM NGAY** | Hôm nay (2-3 giờ) |
| **2. Hậu xử lý không âm** | ✅ **LÀM NGAY** | Hôm nay (1 giờ) |
| **3. Cosine Warm Restarts** | ⚠️ **THỬ NGHIỆM** | Tuần sau (sau khi có baseline) |
| **4. Temporal Features** | ❌ **CHƯA LÀM** | Chỉ khi check_current_setup.py OK |

---

## 🚀 ROADMAP 1 TUẦN

### **Ngày 1 (Hôm nay)**
- [x] Đọc và hiểu phân tích này
- [ ] Implement Epidemic Metrics (2-3h)
- [ ] Implement Hậu xử lý không âm (1h)
- [ ] Test với predictions hiện có

### **Ngày 2-3**
- [ ] Chạy tất cả models hiện có với metrics mới
- [ ] So sánh epidemic metrics: Llama2 vs PatchTST vs DLinear
- [ ] Tạo báo cáo so sánh

### **Ngày 4-5**
- [ ] Chạy `check_current_setup.py`
- [ ] Quyết định có làm temporal features không
- [ ] Nếu KHÔNG: Bỏ qua, focus vào cải tiến khác
- [ ] Nếu CÓ: Implementation (1-2 ngày)

### **Ngày 6-7**
- [ ] Thử nghiệm Cosine Warm Restarts
- [ ] So sánh với baseline (cùng seed)
- [ ] Quyết định giữ hay bỏ

---

## 📝 KẾT LUẬN

### ✅ KHUYẾN NGHỊ ƯU TIÊN

1. **Epidemic Metrics**: LÀM NGAY - dễ, quan trọng, không rủi ro
2. **Hậu xử lý không âm**: LÀM NGAY - rất dễ, hợp lý
3. **Cosine Warm Restarts**: THỬ NGHIỆM - dễ nhưng có rủi ro, cần validation
4. **Temporal Features**: KIỂM TRA TRƯỚC - khó, rủi ro cao, cần phân tích kỹ

### 🎯 MỤC TIÊU TUẦN NÀY

- ✅ Hoàn thành metrics tốt hơn cho đánh giá dự báo dịch
- ✅ Predictions hợp lý hơn (không âm)
- ⚠️ (Optional) Thử warm restarts nếu có thời gian

### 💡 BƯỚC TIẾP THEO SAU 4 TÍNH NĂNG NÀY

1. **Data Augmentation** (từ PHUONG_HUONG_CAI_TIEN.md)
2. **Ensemble Methods** (average các models hiện có)
3. **External Features** (weather, Google Trends - cần thu thập data)

---

**Chúc bạn implement thành công! 🚀**
