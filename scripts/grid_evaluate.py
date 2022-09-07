# Copyright 2020 HSLU. All Rights Reserved.
#
# Created by ialionet on 04.02.2020.
#
import itertools
import os
from typing import Dict, Any

FILENAME = 'find_optimizer/linear_sgd_{}.in'

FIXED = {
    'dataset': 'dataset',
    'num_train_years': range(4, 9),
    'optimizer': 'SGD',
    'model': 'linear',
    'outputs': 1,
    'time_steps': 1,
    'batch_size': 4096,
    'epochs': 1000,
}

GRID = {
    'target_names': ('target', 'target'),
    'momentum': (0.0, 0.9),
    'learning_rate': (0.1, 0.01, 0.001, 0.0001),
    'seed': range(1, 9, 1),
}


def get_all_dicts(grid_spec: Dict[str, Any]):
    keys = grid_spec.keys()
    value_grid = [c for c in itertools.product(*grid_spec.values())]
    return [dict(zip(keys, values)) for values in value_grid]


def main():
    all_dicts = get_all_dicts(GRID)
    total_runs = len(all_dicts)
    for i, a_dict in enumerate(all_dicts):
        print('-'*100)
        print(f'Run {i+1} of {total_runs}, with parameters {a_dict}')
        a_dict.update(FIXED)
        filename = FILENAME.format(i+1)
        with open(filename, 'w') as file:
            for k, v in a_dict.items():
                file.write(k + ' ' + str(v) + '\n')
        os.system('python3 scripts/evaluate.py --base_path .. @'+filename)
        print()


if __name__ == '__main__':
    main()
