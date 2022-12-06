#!/usr/bin/env python
import wandb
import argparse
from assignments.a2.train import train


def main():
    # Define the parser
    parser = argparse.ArgumentParser(description='Wandb Sweep')

    # Add the arguments
    parser.add_argument('--sweep', help='Run sweep', action='store_true')
    parser.add_argument('--sweep_id', help='Sweep ID')
    parser.add_argument('--agent', help='Run agent', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the argument sweep is True
    if args.sweep:
        sweep_config = {
            'method': 'grid', # bayes, grid, random 
            'name': 'sweep',
            'metric': {
                'name': 'mean_val_sp_index',
                'goal': 'maximize'
            },
            'parameters': {
                'training_type': {
                    'values': ['kfold']
                },
                'preprocessing_lofar_spectrum_bins_left': {
                    'values': [400, 200]
                },
                'data_normalization': {
                    'values': ['mapminmax', 'mapstd']
                },
                'binarization_strategy': {
                    'values': [
                        'basic_bin',
                        'simple_thermometer',
                        'circular_thermometer'
                    ]
                },
                'binarization_threshold': {
                    'values': [-0.5, -0.25, 0, 0.25, 0.5]
                },
                'binarization_resolution': {
                    'values': [2 ** x for x in range(1, 5)]
                },
                'binarization_window_size': {
                    'values': [2 ** x for x in range(1, 5)]
                },
                'wsd_address_size': {
                    'values': [2 ** x for x in range(1, 8)]
                },
                'wsd_ignore_zero': {
                    'values': [False]
                },
                'wsd_bleaching_activated': {
                    'values': [True, False]
                },
            }
        }
        sweep_id = wandb.sweep(
            entity="viniciusdsmello",
            project="wnn-sonar",
            sweep=sweep_config
        )
        print("Sweep ID: ", sweep_id)
    if args.agent and args.sweep_id:
        # Run the agent
        wandb.agent(args.sweep_id, entity='viniciusdsmello', project='wnn-sonar', function=train)


if __name__ == '__main__':
    main()
