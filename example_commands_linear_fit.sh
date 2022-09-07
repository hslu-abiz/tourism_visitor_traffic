python3 scripts/sklearn_fit.py --base_path .. --dataset dataset --num_train_years 4 -v
python3 scripts/evaluate.py   --base_path .. --dataset dataset --num_train_years 4 --model linear --time_steps 1 --batch_size 4096 --optimizer sklearn --epochs 0 -v
python3 scripts/predict.py    --base_path .. --dataset dataset --num_train_years 4 --model linear --time_steps 1 --batch_size 4096 --optimizer sklearn --epochs 0 -v --show
