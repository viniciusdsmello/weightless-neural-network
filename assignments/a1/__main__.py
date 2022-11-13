#!/usr/bin/env python
import wandb
from assignments.a1.train import train

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {
            'name': 'mean_val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'training_type': {
                'values': ['kfold']
            },
            'binarization_strategy': {
                'values': ['basic_bin', 'simple_thermometer', 'circular_thermometer', 'sauvola', 'niblack', 'adaptive_thresh_mean', 'adaptive_thresh_gaussian']
            },
            'binarization_threshold': {
                'values': [128, 64, 32, 16, 8, 4, 2]
            },
            'binarization_resolution': {
                'values': [20, 40, 60, 80, 100]
            },
            'binarization_window_size': {
                'values': [3, 5, 7, 9, 11]
            },
            'binarization_constant_c': {
                'values': [2, 4, 6, 8, 10]
            },
            'binarizaton_constant_k': {
                'values': [0.2, 0.4, 0.6, 0.8, 1.0]
            },
            'wsd_address_size': {
                'values': [2, 4, 8, 16, 32, 64, 128, 256]
            },
            'wsd_ignore_zero': {
                'values': [True, False]
            },
        }
    }
    sweep_id = wandb.sweep(
        entity="viniciusdsmello",
        project="wnn",
        sweep=sweep_config
    )
    wandb.agent(sweep_id, function=train)
