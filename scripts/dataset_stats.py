import pandas as pd

DATASETS = ['dataset', 'dataset']
TARGETS = ['target', 'target']


dataset_stats = []
for dataset, target in zip(DATASETS, TARGETS):
    df = pd.read_csv(
        f'../data/processed/{dataset}_train_10ty_0vy.csv',
        sep=';',
        index_col='x_c_dat1',
    )
    targets = df[target]
    mean = targets.mean()
    std = targets.std()
    dataset_stats.append({'dataset': dataset, 'mean': mean, 'std': std})
dataset_stats_df = pd.DataFrame(data=dataset_stats)
print(dataset_stats_df.to_latex(float_format='%.1f', index=False))
