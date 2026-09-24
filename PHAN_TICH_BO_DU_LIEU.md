# PHÂN TÍCH CHI TIẾT BỘ DỮ LIỆU LLMs4Influenza

## 📊 TỔNG QUAN BỘ DỮ LIỆU

Dự án sử dụng **dữ liệu time series về dịch cúm** từ 3 khu vực địa lý khác nhau để dự báo xu hướng bệnh dịch.

---

## 📁 CÁC FILE DỮ LIỆU

### 1. **Dữ liệu gốc (Raw Data)** - 469 dòng mỗi file

#### A. **Flu_in_NorthChina.csv** - Dữ liệu cúm Bắc Trung Quốc
```
Khoảng thời gian: 2011-04-04 đến 2020-03-23
Số tuần: 469 tuần (~9 năm)
Cột dữ liệu:
  - date: Ngày (theo tuần)
  - positive_rate: Tỷ lệ dương tính với virus cúm (%)

Ví dụ:
date,positive_rate
2011-04-04,4.83754896271468
2011-04-11,5.00789840876734
...
2020-03-23,1.87518214659929
```

**Đặc điểm:**
- Tỷ lệ dương tính dao động từ 0% đến ~49%
- Có mùa vụ rõ rệt: Cao vào mùa đông (12-3), thấp vào mùa hè (6-8)
- Đỉnh dịch cao nhất: Tháng 1/2018 (48.93%), 1/2019 (44.39%)

---

#### B. **Flu_in_SouthChina.csv** - Dữ liệu cúm Nam Trung Quốc
```
Khoảng thời gian: 2011-04-04 đến 2020-03-23
Số tuần: 469 tuần
Cột dữ liệu:
  - date: Ngày (theo tuần)
  - positive_rate: Tỷ lệ dương tính (%)
```

**Đặc điểm:**
- Khác với Bắc TQ, Nam TQ có 2 mùa dịch:
  - Mùa chính: Tháng 12-2
  - Mùa phụ: Tháng 6-8 (mùa hè vẫn có dịch)
- Pattern khác biệt do khí hậu nhiệt đới/cận nhiệt đới

---

#### C. **Flu_in_USA.csv** - Dữ liệu cúm Hoa Kỳ
```
Khoảng thời gian: 2011-04-04 đến 2020-03-23
Số tuần: 469 tuần
Cột dữ liệu:
  - date: Ngày (theo tuần)
  - positive_rate: Tỷ lệ dương tính (%)
```

**Đặc điểm:**
- Mùa cúm rõ rệt: Tháng 11-3
- Tỷ lệ cao nhất vào tháng 2-3 hàng năm
- Pattern tương tự Bắc TQ (cùng vĩ độ ôn đới)

---

#### D. **ILI_in_NorthChina.csv** - Dữ liệu ILI Bắc Trung Quốc
```
ILI = Influenza-Like Illness (Bệnh giống cúm)
Khoảng thời gian: 2011-04-04 đến 2020-03-23
Số tuần: 469 tuần
Cột dữ liệu:
  - date: Ngày (theo tuần)
  - positive_rate: Số ca ILI (không phải %, là SỐ TUYỆT ĐỐI)

Ví dụ:
date,positive_rate
2011-04-04,17331
2011-04-11,17542
...
2019-12-23,93740    ← Đỉnh cao nhất
2020-03-23,18509
```

**Đặc điểm:**
- Đây là **số ca bệnh**, không phải tỷ lệ %
- Số ca dao động: 14,000 - 93,000 ca/tuần
- Đỉnh dịch cực cao: Tháng 12/2019 (93,740 ca) - COVID-19 bắt đầu
- Đỉnh dịch cúm: Tháng 12/2017 (84,603 ca)

---

#### E. **ILI_in_SouthChina.csv** - Dữ liệu ILI Nam Trung Quốc
```
Tương tự ILI_in_NorthChina.csv nhưng cho khu vực Nam Trung Quốc
```

---

### 2. **Dữ liệu đã xử lý - Differencing** - 416 dòng mỗi file

Các file `*_diff.csv` là dữ liệu sau khi áp dụng **differencing** (lấy hiệu số giữa 2 tuần liên tiếp).

#### **Tại sao cần Differencing?**
- **Loại bỏ trend**: Time series cúm có trend theo mùa
- **Stationarity**: Làm dữ liệu dừng (stationary) để model học tốt hơn
- **Công thức**: `diff[t] = value[t] - value[t-1]`

#### Ví dụ: **NorthChina_diff.csv**
```
date,positive_rate
2012-04-09,-4.73384511870036    ← Giảm 4.73% so với tuần trước
2012-04-16,-0.03348409396126    ← Giảm 0.03%
2012-04-23,-3.71678244044018    ← Giảm 3.71%
...
```

**Đặc điểm dữ liệu diff:**
- Giá trị dương (+): Tăng so với tuần trước
- Giá trị âm (-): Giảm so với tuần trước
- Số dòng: 416 (= 469 - 53 dòng đầu để training)
- Thời gian: Bắt đầu từ 2012-04-09 (sau 52 tuần đầu)

---

## 🎯 MỤC ĐÍCH SỬ DỤNG

### **Training Data (70%)**
```
Split trong code (data_loader.py line 360-363):
- Train: 70% dữ liệu đầu
- Validation: 10% 
- Test: 20% cuối
```

### **Input cho Model**
```python
--seq_len 52      # Dùng 52 tuần (1 năm) dữ liệu quá khứ
--pred_len 8-13   # Dự đoán 8-13 tuần tương lai
--label_len 18    # Overlap giữa input và output
```

### **Ví dụ cụ thể:**
```
Input:  Tuần 1-52   (52 tuần = 1 năm dữ liệu cúm)
        |--------------------------------|
Label:  Tuần 35-52 (18 tuần overlap)
        |------------------|
Output:                     Tuần 53-60 (8 tuần dự đoán)
                            |--------|
```

---

## 📈 CẤU TRÚC DỮ LIỆU CHI TIẾT

### **Format CSV**
```csv
date,positive_rate
YYYY-MM-DD,float_value
```

### **Đặc tính Time Series**
1. **Frequency**: Weekly (hàng tuần)
2. **Missing values**: Một số tuần có giá trị 0.0 (đặc biệt mùa hè)
3. **Seasonality**: Mùa vụ rõ rệt (12-52 tuần/cycle)
4. **Trend**: Có xu hướng tăng theo năm ở một số khu vực

### **Thống kê cơ bản - Flu_in_NorthChina.csv**
```
Min:  0.00%      (mùa hè 2013, 2016)
Max:  48.93%     (tháng 1/2018)
Mean: ~10-15%
Std:  ~10-12%

Mùa cao điểm (> 30%):
- Mùa đông 2011-2012: 33.19%
- Mùa đông 2013-2014: 34.39%
- Mùa đông 2014-2015: 34.70%
- Mùa đông 2015-2016: 40.83%
- Mùa đông 2017-2018: 48.93% ← Cao nhất
- Mùa đông 2018-2019: 44.39%
- Mùa đông 2019-2020: 38.33%
```

---

## 🔬 NGUỒN DỮ LIỆU

### **Paper gốc:**
[COVID-19, flu, and RSV surveillance](https://www.nature.com/articles/s41467-021-23440-1)
- **Nature Communications**
- Published: 2021
- Tác giả: Nghiên cứu từ CDC Trung Quốc và CDC Mỹ

### **Các nguồn chính:**
1. **China CDC** (Chinese Center for Disease Control and Prevention)
   - Dữ liệu từ hệ thống giám sát bệnh truyền nhiễm quốc gia
   - Chia thành Bắc/Nam theo đường vĩ tuyến

2. **US CDC** (Centers for Disease Control and Prevention)
   - FluView surveillance system
   - WHO Collaborating Laboratories

### **Phương pháp thu thập:**
- **Lấy mẫu**: Bệnh nhân có triệu chứng cúm tại các bệnh viện sentinel
- **Xét nghiệm**: PCR test cho virus influenza A, B
- **Tổng hợp**: Theo tuần, báo cáo tỷ lệ dương tính
- **ILI**: Số ca bệnh giống cúm (sốt + ho/đau họng) từ các phòng khám

---

## 🎓 ĐẶC ĐIỂM QUAN TRỌNG ĐỂ TRAIN MODEL

### 1. **Seasonality (Tính mùa vụ)**
```
Bắc TQ, Mỹ:   Một mùa dịch (tháng 11-3)
Nam TQ:       Hai mùa dịch (12-2 và 6-8)
```
→ Model cần học được pattern theo mùa

### 2. **Trend (Xu hướng)**
```
- Xu hướng tăng nhẹ qua các năm
- Đỉnh dịch 2017-2018 và 2018-2019 rất cao
- Giảm mạnh sau COVID-19 (đầu 2020)
```
→ Cần differencing để loại bỏ trend

### 3. **Variability (Biến động)**
```
- Biến động cao trong mùa dịch
- Ổn định thấp trong mùa hè
- Standard deviation: ~10-12%
```
→ Model cần robust với noise

### 4. **Lag Effect (Hiệu ứng trễ)**
```
- Dịch cúm lan chậm: 1-2 tuần
- Tỷ lệ dương tính phản ánh mức độ lây lan
```
→ Cần seq_len đủ dài (52 tuần)

---

## 📊 PHÂN TÍCH DATA SPLIT

### **Thực tế khi chạy với NorthChina_diff.csv**
```python
Total rows: 416 tuần
- Train: 70% = 291 tuần (~5.6 năm)
- Val:   10% = 42 tuần (~0.8 năm)
- Test:  20% = 83 tuần (~1.6 năm)

Với seq_len=52, pred_len=8:
- Mỗi sample cần 60 tuần (52 input + 8 output)
- Train samples: ~239 samples
- Val samples:   ~0 (không đủ dữ liệu, chia lại)
- Test samples:  ~25 samples
```

### **Lưu ý:**
```python
# Code trong data_loader.py tự động điều chỉnh
# Nếu percent=10 → Chỉ dùng 10% train data
# Nếu percent=100 → Dùng toàn bộ train data
```

---

## 🎯 CÁC FILE ĐƯỢC DÙNG TRONG SCRIPTS

### **Scripts PatchTST/DLinear:**
```bash
# Bắc TQ - Flu
--data_path NorthChina_diff.csv
--target positive_rate

# Nam TQ - Flu
--data_path SouthChina_diff.csv
--target positive_rate

# Mỹ - Flu
--data_path USA_diff.csv
--target positive_rate

# Bắc TQ - ILI
--data_path ILI_NorthChina_diff.csv
--target positive_rate

# Nam TQ - ILI
--data_path ILI_SouthChina_diff.csv
--target positive_rate
```

### **Scripts LLMs (Llama2/3, Gemma2):**
- Dùng các file tương tự
- Thêm pretrained LLM để học pattern phức tạp hơn

---

## 🔍 KẾT LUẬN

### **Bộ dữ liệu này phù hợp cho:**
1. ✅ Time series forecasting
2. ✅ Seasonal pattern learning
3. ✅ Epidemic prediction
4. ✅ Multi-region comparison
5. ✅ Transfer learning (giữa các khu vực)

### **Thách thức:**
1. ⚠️ Dữ liệu ít (chỉ ~9 năm, 469 tuần)
2. ⚠️ Nhiễu cao trong mùa hè (giá trị rất thấp)
3. ⚠️ COVID-19 làm gián đoạn pattern (2020)
4. ⚠️ Missing data ở một số tuần
5. ⚠️ Distribution khác nhau giữa train/test nếu có pandemic

### **Điểm mạnh:**
1. ✅ Dữ liệu thực tế, chất lượng cao từ CDC
2. ✅ Multi-region cho phép transfer learning
3. ✅ Seasonality rõ ràng, dễ học
4. ✅ Đã published trên Nature → tin cậy
5. ✅ Có cả Flu và ILI (2 loại data)

---

## 📚 CÁCH SỬ DỤNG

### **1. Dự báo cúm Bắc Trung Quốc:**
```bash
python main.py \
    --root_path ./dataset/ \
    --data_path NorthChina_diff.csv \
    --target positive_rate \
    --seq_len 52 \
    --pred_len 8 \
    --model PatchTST
```

### **2. Dự báo ILI (số ca bệnh):**
```bash
python main.py \
    --root_path ./dataset/ \
    --data_path ILI_NorthChina_diff.csv \
    --target positive_rate \
    --seq_len 52 \
    --pred_len 13
```

### **3. So sánh nhiều khu vực:**
Chạy lần lượt với:
- `NorthChina_diff.csv`
- `SouthChina_diff.csv`
- `USA_diff.csv`

So sánh metrics (MSE, MAE, Correlation) để thấy model hoạt động tốt ở khu vực nào.

---

## 💡 TIPS

1. **Dùng dữ liệu diff**: Hiệu quả hơn raw data
2. **seq_len=52**: Bao phủ cả chu kỳ 1 năm
3. **pred_len=8-13**: Dự đoán 2-3 tháng tương lai
4. **percent=100**: Dùng toàn bộ train data để không mất thông tin
5. **itr=3**: Chạy 3 lần để có kết quả ổn định (mean ± std)

---

**Tóm lại**: Bộ dữ liệu này là **time series về tỷ lệ dương tính virus cúm theo tuần**, được dùng để **dự báo xu hướng dịch cúm trong tương lai** bằng các mô hình deep learning và LLMs.
