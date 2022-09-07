python3 scripts/train.py    --base_path .. --dataset dataset --num_train_years 4 --model linear --time_steps 1 --batch_size 2048 --optimizer SGD --epochs 10 -v
python3 scripts/evaluate.py --base_path .. --dataset dataset --num_train_years 4 --model linear --time_steps 1 --batch_size 2048 --optimizer SGD --epochs 10 -v
python3 scripts/predict.py  --base_path .. --dataset dataset --num_train_years 4 --model linear --time_steps 1 --batch_size 2048 --optimizer SGD --epochs 10 -v --show
