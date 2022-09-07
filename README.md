# Tourism Project

Workflow to train models

## Build system

Build docker for debugging and building

```
docker build -t tourism_docker -f ./docker/Dockerfile-dev ./
```

**Call the train function**
```
docker run -v "$(pwd)"/data:/app/data -v "$(pwd)/results":/app/results -v "$(pwd)/tourism_workflow":/app/main -it tourism_docker /bin/bash -c "cd /app/main && python -m scripts.train --help"
```

**Call nvidia-docker for gpu 1**
```
docker run --gpus "device=1" -v "$(pwd)"/data:/app/data -v "$(pwd)/results":/app/results -v "$(pwd)/tourism_workflow":/app/main -it tapfaeff-tourism-gpu-stack
```

## Preprocess scripts

Call if using the sources as python library.
```
python -m scripts.preprocess_dataset --base_path /data --dataset dataset --num_train_years 4 -v
```


## Plot scripts

To plot the target value for a dataset use
```
python -m plotting.plot_data --dataset_path /app/data/dataset/dataset.csv --dataset_delimiter ',' --column_information_path /app/data/dataset/dataset.csv --column_information_delimiter ";" --save ../dataset_plot.png
```
