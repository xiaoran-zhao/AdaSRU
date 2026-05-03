from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


# 定义命令行参数解析函数
def parse_args() -> argparse.Namespace:
    # 创建一个参数解析器对象
    p = argparse.ArgumentParser()
    # 添加参数：训练结果目录，必须提供
    p.add_argument('--run_dir', type=str, required=True)
    # 添加参数：输出目录，必须提供
    p.add_argument('--output_dir', type=str, required=True)
    # 添加参数：eps 一共采样多少个值，默认 6 个
    p.add_argument('--num_eps', type=int, default=6)
    # 添加参数：每个 eps 对应跑多少个 epoch，默认 30
    p.add_argument('--epochs', type=int, default=30)
    # 添加参数：每个 epoch 内跑多少步，默认 64
    p.add_argument('--steps_per_epoch', type=int, default=64)
    # 添加参数：批大小，默认 2048
    p.add_argument('--batch_size', type=int, default=2048)
    # 添加参数：学习率，默认 5e-4
    p.add_argument('--lr', type=float, default=5e-4)
    # 添加参数：运行设备，默认 auto，表示自动选择
    p.add_argument('--device', type=str, default='auto')
    # 解析命令行参数并返回
    return p.parse_args()

# 定义一个几何间隔采样函数
# 用来在 start 和 stop 之间生成 num 个按对数均匀分布的数
def geomspace(start: float, stop: float, num: int) -> list[float]:
    # 如果只需要一个数，就直接返回起点
    if num == 1:
        return [start]
    # 在对数空间中均匀划分 num 个点
    # 先取 log，再线性插值
    logs = [math.log(start) + i * (math.log(stop) - math.log(start)) / (num - 1) for i in range(num)]
    # 再通过 exp 把对数空间的点还原回原始数值空间
    return [math.exp(x) for x in logs]

# 主函数
def main() -> None:
    # 读取命令行参数
    args = parse_args()
    # 生成 eps 值列表，从 1e-3 到 5e-1，按几何间隔取样
    eps_list = geomspace(1e-1, 1e+1, args.num_eps)
    # 将 eps 列表格式化为字符串，并用逗号连接
    # 例如 "0.00100000,0.00346572,..."
    eps_str = ','.join(f'{x:.8f}' for x in eps_list)
    # 构造将要执行的命令
    cmd = [
        # 当前 Python 解释器路径，例如 /usr/bin/python
        sys.executable,
        # 表示以模块方式运行
        '-m', 'src.unlearn',
        # 下面依次传入 src.unlearn 所需的参数
        '--run_dir', args.run_dir,
        '--output_dir', args.output_dir,
        '--epochs', str(args.epochs),
        '--steps_per_epoch', str(args.steps_per_epoch),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
        '--eps_list', eps_str,
    ]
    # 打印即将执行的完整命令，方便调试和查看
    print('Running:', ' '.join(cmd))
    # 执行该命令
    # 如果子进程返回非 0 状态码，会直接抛出异常
    subprocess.check_call(cmd)


# 如果当前脚本是直接运行，而不是被别的文件 import
if __name__ == '__main__':
    # 执行主函数
    main()