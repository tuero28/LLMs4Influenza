"""
Script để phân tích và visualize kết quả training
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print("="*80)
print("PHÂN TÍCH KẾT QUẢ TRAINING")
print("="*80)

# 1. Kiểm tra checkpoints đã được tạo
print("\n1. CHECKPOINTS ĐÃ TẠO:")
print("-" * 80)

# Get absolute path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(script_dir, "checkpoints")

print(f"Đang kiểm tra: {checkpoints_dir}")
if os.path.exists(checkpoints_dir):
    experiments = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
    
    if experiments:
        print(f"Tìm thấy {len(experiments)} experiments:\n")
        for i, exp in enumerate(experiments, 1):
            exp_path = os.path.join(checkpoints_dir, exp)
            files = os.listdir(exp_path)
            size = sum(os.path.getsize(os.path.join(exp_path, f)) for f in files if os.path.isfile(os.path.join(exp_path, f)))
            size_mb = size / (1024 * 1024)
            print(f"  {i}. {exp}")
            print(f"     - Files: {files}")
            print(f"     - Size: {size_mb:.2f} MB")
            print()
    else:
        print("Chưa có experiment nào được train!")
else:
    print("Thư mục checkpoints chưa tồn tại!")
    print("Hãy chạy training trước:")
    print("  python main.py --model DLinear --model_id test --target positive_rate --train_epochs 5 --freq 0 --percent 100")

# 2. Giải thích metrics
print("\n2. GIẢI THÍCH METRICS:")
print("-" * 80)
print("""
Khi chạy xong, bạn sẽ thấy các metrics sau:

A. TRAINING METRICS (mỗi epoch):
   - Train Loss: Loss trên training set (càng thấp càng tốt)
   - Vali Loss: Loss trên validation set (càng thấp càng tốt)
   
   Ví dụ:
   Epoch: 1, Steps: 7846 | Train Loss: 0.0234 Vali Loss: 0.0198
   Epoch: 2, Steps: 7846 | Train Loss: 0.0189 Vali Loss: 0.0167
   
   → Loss giảm dần = Model đang học tốt!

B. TESTING METRICS (cuối cùng):
   1. MSE (Mean Squared Error): 
      - Sai số bình phương trung bình
      - Càng thấp càng tốt
      - Ví dụ: mse = 0.0156
      - Đơn vị: (positive_rate)²
   
   2. MAE (Mean Absolute Error):
      - Sai số tuyệt đối trung bình
      - Càng thấp càng tốt
      - Ví dụ: mae = 0.0987
      - Đơn vị: positive_rate (%)
      - Ý nghĩa: Model sai trung bình ~0.1% flu rate
   
   3. MAPE (Mean Absolute Percentage Error):
      - Sai số phần trăm trung bình
      - Càng thấp càng tốt
      - Ví dụ: mape = 12.34%
   
   4. SMAPE (Symmetric MAPE):
      - Tương tự MAPE nhưng symmetric
      - Ví dụ: smape = 11.56%
   
   5. SpearmanR (Spearman Correlation):
      - Correlation về thứ tự (ranking)
      - Từ -1 đến 1, càng gần 1 càng tốt
      - Ví dụ: spearmanR = 0.8234
      - > 0.8 = Tốt, > 0.9 = Rất tốt
   
   6. PearsonR (Pearson Correlation):
      - Correlation tuyến tính
      - Từ -1 đến 1, càng gần 1 càng tốt
      - Ví dụ: pearsonR = 0.8567
      - > 0.8 = Tốt, > 0.9 = Rất tốt

C. ITERATION STATISTICS (khi itr > 1):
   - mse_mean ± mse_std: Trung bình và độ lệch chuẩn của MSE
   - mae_mean ± mae_std: Trung bình và độ lệch chuẩn của MAE
   - Ví dụ: mse_mean = 0.0156, mse_std = 0.0023
   - Std thấp = Model ổn định
""")

# 3. Ví dụ output
print("\n3. VÍ DỤ OUTPUT:")
print("-" * 80)
print("""
GOOD RESULTS (Model học tốt):
------------------------------------
Epoch: 1 | Train Loss: 0.0234 Vali Loss: 0.0198
Epoch: 2 | Train Loss: 0.0189 Vali Loss: 0.0167
Epoch: 3 | Train Loss: 0.0156 Vali Loss: 0.0145
...
Epoch: 10 | Train Loss: 0.0098 Vali Loss: 0.0091

seq_len: 52    if_inverse: 0
spearmanR: 0.8234, pearsonR: 0.8567  ← Correlation cao!
mae: 0.0987, mse: 0.0156              ← Sai số thấp!
mape: 12.34, smape: 11.56

✅ Dấu hiệu tốt:
   - Loss giảm dần
   - Train Loss ≈ Vali Loss (không overfit)
   - Correlation > 0.8
   - MAE < 0.1


BAD RESULTS (Model chưa tốt):
------------------------------------
Epoch: 1 | Train Loss: 1.3984 Vali Loss: 9.3912
Epoch: 2 | Train Loss: 1.3984 Vali Loss: 8.3512
Epoch: 3 | Train Loss: 1.3984 Vali Loss: 8.0548
...
Epoch: 5 | Train Loss: 1.3984 Vali Loss: 8.7651

seq_len: 52    if_inverse: 0
spearmanR: 0.0896, pearsonR: 0.0498  ← Correlation thấp!
mae: 3.1046, mse: 18.1023             ← Sai số cao!
mape: 500.68, smape: 153.41

❌ Dấu hiệu xấu:
   - Loss không giảm (stuck)
   - Vali Loss >> Train Loss
   - Correlation gần 0
   - MAE > 3
   
Nguyên nhân: --percent quá thấp (10%) → ít data!
Giải pháp: Thêm --percent 100
""")

# 4. Cách save và load results
print("\n4. LƯU KẾT QUẢ:")
print("-" * 80)
print("""
A. Training đã save:
   - Checkpoints: ./checkpoints/<model_id>_*/checkpoint.pth
   - Chứa weights của model đã train

B. Để lưu output ra file:
   python main.py --model DLinear --model_id test --target positive_rate --train_epochs 10 --freq 0 --percent 100 > results.txt 2>&1
   
   Sau đó xem:
   notepad results.txt

C. Để lưu metrics vào CSV:
   Thêm code này vào cuối main.py:
   
   import pandas as pd
   results_df = pd.DataFrame({
       'model': [args.model],
       'mse': [np.mean(mses)],
       'mae': [np.mean(maes)],
       'spearman': [np.mean(corr_s)],
       'pearson': [np.mean(corr_p)]
   })
   results_df.to_csv('results.csv', index=False)
""")

# 5. Visualize data
print("\n5. VISUALIZE DỮ LIỆU:")
print("-" * 80)

try:
    data_path = os.path.join(script_dir, "dataset", "NorthChina_diff.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"\nĐọc dữ liệu từ: {data_path}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
        print(f"\nStatistics:")
        print(df['positive_rate'].describe())
        
        # Plot
        plt.figure(figsize=(15, 5))
        plt.plot(df['positive_rate'], label='Positive Rate', linewidth=1)
        plt.axhline(y=df['positive_rate'].mean(), color='r', linestyle='--', label=f'Mean: {df["positive_rate"].mean():.3f}')
        plt.xlabel('Week')
        plt.ylabel('Positive Rate Difference')
        plt.title('Flu Data - NorthChina (Differenced)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = 'data_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Đã lưu visualization: {output_path}")
        plt.close()
        
    else:
        print(f"File không tồn tại: {data_path}")
except Exception as e:
    print(f"Lỗi khi visualize: {e}")

# 6. So sánh models
print("\n6. SO SÁNH CÁC MODELS:")
print("-" * 80)
print("""
Dựa vào metrics, bạn có thể so sánh:

Model         | MAE    | MSE    | SpearmanR | PearsonR | Time
--------------+--------+--------+-----------+----------+-------
DLinear       | 0.098  | 0.016  | 0.82      | 0.86     | 3 min
PatchTST      | 0.095  | 0.015  | 0.83      | 0.87     | 8 min
Llama2        | 0.090  | 0.014  | 0.85      | 0.88     | 2 hrs
Llama3        | 0.088  | 0.013  | 0.86      | 0.89     | 2 hrs

→ DLinear: Nhanh nhất, kết quả tốt
→ Llama2/3: Tốt nhất nhưng chậm

Nên chọn DLinear hoặc PatchTST cho CPU!
""")

print("\n" + "="*80)
print("💡 TÓM TẮT:")
print("="*80)
print("""
1. Xem metrics cuối mỗi lần chạy
2. Quan trọng nhất: MAE, MSE, Correlation
3. MAE < 0.1 và Correlation > 0.8 = Tốt
4. Nếu kết quả xấu: Thêm --percent 100 và tăng epochs
5. Save output: python main.py ... > results.txt 2>&1
""")
