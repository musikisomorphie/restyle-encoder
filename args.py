import random
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        # default=0,
                        help='global seed (for weight initialization, data sampling, etc.). '
                        'If not specified it will be randomized (and printed on the log)')
    parser.add_argument('--mode',
                        default='recon',
                        choices=('recon', 'synth'))

    parser.add_argument('--decoder',
                        default='style2',
                        choices=('style2', 'style3'))
    parser.add_argument('--method',
                        default='psp',
                        choices=('psp', 'e4e'))

    parser.add_argument('--data_name',
                        type=str,
                        choices=['rxrx19a_HRCE', 'rxrx19b_VERO', 'rxrx19b', 'ham10k'],
                        help='experiments to run')
    parser.add_argument('--data_splt',
                        type=str,
                        choices=['official', '012', '120', '201'],
                        help='official for rxrx1, the rests for scrc')
    parser.add_argument('--data_path',
                        type=Path,
                        help='path to the data root.')

    parser.add_argument('--ckpt_path',
                        type=Path,
                        help='Path to checkpoint for the auto-encoder')
    parser.add_argument('--save_path',
                        type=Path,
                        help='path to output visual images')

    parser.add_argument('--n_eval',
                        type=int,
                        default=16,
                        help='evaluation batch size')
    parser.add_argument('--n_work',
                        type=int,
                        default=8,
                        help='number of data loader workers')

    args = parser.parse_args()

    assert args.save_path is not None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args
