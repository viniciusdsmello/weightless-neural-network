#!/usr/bin/env python
import wandb
import argparse
from assignments.a1.train import train


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
            'method': 'bayes',
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
                    'values': [
                        # 'basic_bin',
                        'simple_thermometer',
                        # 'circular_thermometer',
                        # 'sauvola',
                        # 'niblack',
                        # 'adaptive_thresh_mean',
                        # 'adaptive_thresh_gaussian'
                    ]
                },
                'binarization_threshold': {
                    'values': [None]
                },
                'binarization_resolution': {
                    'values': [
                        20,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        100,
                        110,
                        120,
                        130,
                        140,
                        150
                    ]
                },
                'binarization_window_size': {
                    'values': [None]
                },
                'binarization_constant_c': {
                    'values': [None]
                },
                'binarization_constant_k': {
                    'values': [None]
                },
                'wsd_address_size': {
                    'values': [24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
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
            project="wnn",
            sweep=sweep_config
        )
        print("Sweep ID: ", sweep_id)
    if args.agent and args.sweep_id:
        # Run the agent
        wandb.agent(args.sweep_id, entity='viniciusdsmello', project='wnn', function=train)


if __name__ == '__main__':
    main()
