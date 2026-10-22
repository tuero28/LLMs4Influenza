@echo off
REM Run Ensemble with different weight combinations

echo ===============================================
echo ENSEMBLE EXPERIMENTS - TESTING WEIGHTS
echo ===============================================
echo.

echo [1/5] Equal weights (0.33, 0.33, 0.34)...
python ensemble_predictor.py --ensemble_weights 0.33,0.33,0.34 --checkpoints_dir ./checkpoints/ --root_path ./dataset/ --data_path Flu_in_NorthChina.csv --data custom --seq_len 52 --label_len 18 --pred_len 8 --d_model 4096 --llama_layers 8 --fc_layer 256 --features S --target positive_rate --freq 0
echo.

echo [2/5] Favor Llama2 (0.2, 0.2, 0.6)...
python ensemble_predictor.py --ensemble_weights 0.2,0.2,0.6 --checkpoints_dir ./checkpoints/ --root_path ./dataset/ --data_path Flu_in_NorthChina.csv --data custom --seq_len 52 --label_len 18 --pred_len 8 --d_model 4096 --llama_layers 8 --fc_layer 256 --features S --target positive_rate --freq 0
echo.

echo [3/5] Favor DLinear (0.5, 0.3, 0.2)...
python ensemble_predictor.py --ensemble_weights 0.5,0.3,0.2 --checkpoints_dir ./checkpoints/ --root_path ./dataset/ --data_path Flu_in_NorthChina.csv --data custom --seq_len 52 --label_len 18 --pred_len 8 --d_model 4096 --llama_layers 8 --fc_layer 256 --features S --target positive_rate --freq 0
echo.

echo [4/5] Favor PatchTST (0.2, 0.6, 0.2)...
python ensemble_predictor.py --ensemble_weights 0.2,0.6,0.2 --checkpoints_dir ./checkpoints/ --root_path ./dataset/ --data_path Flu_in_NorthChina.csv --data custom --seq_len 52 --label_len 18 --pred_len 8 --d_model 4096 --llama_layers 8 --fc_layer 256 --features S --target positive_rate --freq 0
echo.

echo [5/5] Balanced (0.3, 0.3, 0.4)...
python ensemble_predictor.py --ensemble_weights 0.3,0.3,0.4 --checkpoints_dir ./checkpoints/ --root_path ./dataset/ --data_path Flu_in_NorthChina.csv --data custom --seq_len 52 --label_len 18 --pred_len 8 --d_model 4096 --llama_layers 8 --fc_layer 256 --features S --target positive_rate --freq 0
echo.

echo ===============================================
echo ALL EXPERIMENTS COMPLETED
echo ===============================================
pause
