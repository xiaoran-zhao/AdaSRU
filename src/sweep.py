from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--num_eps', type=int, default=6)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps_per_epoch', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--device', type=str, default='auto')

    return p.parse_args()

def geomspace(start: float, stop: float, num: int) -> list[float]:

    if num == 1:
        return [start]

    logs = [math.log(start) + i * (math.log(stop) - math.log(start)) / (num - 1) for i in range(num)]
    return [math.exp(x) for x in logs]

def main() -> None:

    args = parse_args()
    eps_list = geomspace(1e-1, 1e+1, args.num_eps)

    eps_str = ','.join(f'{x:.8f}' for x in eps_list)

    cmd = [
        sys.executable,
        '-m', 'src.unlearn',
        '--run_dir', args.run_dir,
        '--output_dir', args.output_dir,
        '--epochs', str(args.epochs),
        '--steps_per_epoch', str(args.steps_per_epoch),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
        '--eps_list', eps_str,
    ]

    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
