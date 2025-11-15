#!/usr/bin/env python3
"""Train CNN models on MNIST fashion data from the following repo:

https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion


"""

from experiment import run_experiment


def parse_args():
    """Parse and type cast any command line arguments.

    All arguments are optional.  They are as follows:
    
    - model: Choose between BaseNet (default) or ResNet.
    - variant: Select number of convs/module (default 10 -> 2 convs/model).
    - optimizer: Either sgd or adam (default).
    - batch_size: Number of data samples to run at once (default 50).
    - epochs: Number of times to train (default 30).
    - lr: Learning rate (default 0.1).
    - data_root: Local data store and where to download (default './data').
    - subset_train_samples: How much of the training set to use (default 10000).
    - save_dir: Where to save the runs (default './runs').
    - num_workers: How much parallelization to use (default 4).
    - seed: Determines logarithmic determinism (default 42).
    
    Args:
        None.
        
    Returns:
        Parsed args.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['base', 'resnet'], default='base')
    parser.add_argument('--variant', type=int, choices=[10, 16], default=10,
                        help='For BaseNet: 10 -> 2 convs/module, 16 -> 4 convs/module')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--subset_train_samples', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./runs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    models = ["base", "resnet"]
    optimizers = ["sgd", "adam"]
    variants = [10, 16]


    args = parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
