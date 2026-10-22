@echo off
REM Run Ensemble Predictor
REM Combines DLinear, PatchTST, and Llama2 predictions

echo ===============================================
echo ENSEMBLE PREDICTOR
echo ===============================================
echo.

REM Default: Equal weights 0.33, 0.33, 0.34
python ensemble_predictor.py ^
    --checkpoints_dir ./checkpoints/ ^
    --dlinear_name DLinear_optimized ^
    --patchtst_name PatchTST_fast ^
    --llama2_name Llama2_raw_augmented ^
    --ensemble_weights 0.3,0.3,0.4 ^
    --root_path ./dataset/ ^
    --data_path Flu_in_NorthChina.csv ^
    --data custom ^
    --seq_len 52 ^
    --label_len 18 ^
    --pred_len 8 ^
    --d_model 4096 ^
    --n_heads 16 ^
    --e_layers 3 ^
    --d_ff 2048 ^
    --llama_layers 8 ^
    --fc_layer 256 ^
    --pretrain 1 ^
    --freeze 1 ^
    --is_gpt 1 ^
    --batch_size 8 ^
    --features S ^
    --target positive_rate ^
    --freq 0 ^
    --percent 100

echo.
echo ===============================================
echo ENSEMBLE COMPLETED
echo ===============================================
pause
