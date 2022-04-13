import argparse
import random

import numpy
import numpy as np
import torch

# parser = argparse.ArgumentParser(description='单任务模式')
# parser.add_argument('--num_epochs', '-n', type=int, default=2000,help="迭代次数")
# parser.add_argument('--mode', '-m', choices=('train', 'test','train_and_test'), default='train_and_test',help="模式选择")
# parser.add_argument('--lr', '-a', type=float, default=0.001,help="学习率")
# parser.add_argument('--batch_size', '-b', type=int, default=200,help="batch大小")
# parser.add_argument('--hidden_size_1', '-h1', type=int, default=500,help="第二层神经元数量")
# parser.add_argument('--hidden_size_2', '-h2', type=int, default=256,help="第三层神经元数量")
# parser.add_argument('--hidden_size_3', '-h3', type=int, default=128,help="第四层神经元数量")
#
# # parser.add_argument('--sigma', '-s', type=float, default=100.0)
# # parser.add_argument('--sigma', '-s', type=float, default=100.0)
# # parser.add_argument('--sigma', '-s', type=float, default=100.0)
# # parser.add_argument('--sigma', '-s', type=float, default=100.0)
# # parser.add_argument('--sigma', '-s', type=float, default=100.0)
# args = parser.parse_args()
# print(args.hidden_size_3)


def f(*x):
    print(x)

f((1,2,3))

