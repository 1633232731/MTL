import random

import numpy
import numpy as np
import torch

# def get_tensor_data(dataset_name):
#     file_name = dataset_name.split(".")[0]
#     b = numpy.load("tensor_data/{}_tensor.npy".format(file_name), allow_pickle=True)
#     c = [torch.tensor(x) for x in b]
#     return c[0],c[1],c[2],c[3]
#
# datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv"]
#
# for dataset_name in datasets_name:
#     X,Y,x_test,y_test = get_tensor_data(dataset_name)
#
#     print(X)
#     print(Y)
#     print(x_test)
#     print(y_test)
#     print(len(X))
#     print(len(Y))
#     print(len(x_test))
#     print(len(y_test))
#     print()

# def judge_empty(dataloads):
#     for dataload in dataloads:
#         if len(dataload) != 0:
#             return False
#     return True
#
# batch_list = []
# a = [[1,2],[3,4],[5,6,7,8],[9,10,11],[12]]
# while not judge_empty(a):
#     for index, dataloader in enumerate(a):
#         if len(dataloader) != 0:
#             batch_list.append({index:dataloader.pop()})
#
# print(batch_list)
#
# for key in batch_list:
#     print(tuple(key.values())[0])
#
# d = [1,2,3,4,5]
# random.shuffle(d)
# print(d)

# from AutomaticWeightedLoss import AutomaticWeightedLoss
#
# awl = AutomaticWeightedLoss(2)	# we have 2 losses
# loss1 = 1
# loss2 = 2
# loss_sum = awl(loss1, loss2)
#
# print(loss_sum)
# import cv2
# import random
# import colorsys
# import numpy as np
# from core.config import cfg
