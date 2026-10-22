@echo off
echo ========================================
echo TEST ALL MODELS WITH NEW METRICS
echo ========================================
echo.

echo ========================================
echo 1. DLinear - Fast baseline
echo ========================================
python main.py ^
  --model DLinear ^
  --model_id DLinear_with_metrics ^
  --data_path NorthChina_diff.csv ^
  --root_path ./dataset/ ^
  --target positive_rate ^
  --seq_len 52 ^
  --pred_len 8 ^
  --label_len 18 ^
  --train_epochs 10 ^
  --batch_size 16 ^
  --features S ^
  --itr 3 ^
  --freq 0 ^
  --percent 100

echo.
echo ========================================
echo 2. PatchTST - Transformer baseline
echo ========================================
python main.py ^
  --model PatchTST ^
  --model_id PatchTST_with_metrics ^
  --data_path NorthChina_diff.csv ^
  --root_path ./dataset/ ^
  --target positive_rate ^
  --seq_len 52 ^
  --pred_len 8 ^
  --label_len 18 ^
  --train_epochs 20 ^
  --batch_size 16 ^
  --features S ^
  --itr 3 ^
  --freq 0 ^
  --percent 100

echo.
echo ========================================
echo 3. Llama2 - LLM-based (requires GPU)
echo ========================================
python main.py ^
  --model Llama2 ^
  --model_id Llama2_with_metrics ^
  --data_path NorthChina_diff.csv ^
  --root_path ./dataset/ ^
  --data custom ^
  --seq_len 52 ^
  --label_len 18 ^
  --pred_len 13 ^
  --batch_size 8 ^
  --learning_rate 0.001 ^
  --train_epochs 10 ^
  --d_model 4096 ^
  --n_heads 4 ^
  --d_ff 768 ^
  --llama_layers 8 ^
  --itr 3 ^
  --pretrain 1 ^
  --freeze 1 ^
  --features S ^
  --target positive_rate ^
  --freq 0

echo.
echo ========================================
echo ALL MODELS COMPLETED!
echo Check console output for:
echo - Clipping Analysis
echo - Epidemic Metrics
echo ========================================
pause
