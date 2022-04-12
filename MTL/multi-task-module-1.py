import random
import time
from typing import List

import numpy
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
from dataset import MyData
from pretrain_utils import split

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from sklearn.utils import shuffle as reset
from sklearn.metrics import mean_squared_error
import math
import torch
from torch.utils.data.sampler import RandomSampler
from uncertainty_weight_loss import UncertaintyWeightLoss

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4


def random_mini_batches(XE, R1E, R2E, mini_batch_size=10, seed=42):
    # Creating the mini-batches
    np.random.seed(seed)
    m = XE.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_XE = XE[permutation, :]
    shuffled_X1R = R1E[permutation]
    shuffled_X2R = R2E[permutation]
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = shuffled_XE[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        mini_batch_X1R = shuffled_X1R[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_X2R = shuffled_X2R[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
        mini_batches.append(mini_batch)
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m / mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = shuffled_XE[Lower: Lower + Upper, :]
        mini_batch_X1R = shuffled_X1R[Lower: Lower + Upper]
        mini_batch_X2R = shuffled_X2R[Lower: Lower + Upper]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
        mini_batches.append(mini_batch)

    return mini_batches


def get_all_dataset() -> List[str]:
    """
    获取 dataset 文件名
    :return: dataset_name -> String
    """
    dataset = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv", "esol.csv",
               "freesolv.csv", "lipo.csv"]

    return dataset


def get_y_column_title(dataset_name) -> str:
    """
    返回每个数据集需要smiles对应的值的列名
    :param dataset_name: 数据集文件名
    :return:
    """
    dataset_y = {
        "bace.csv": "Class",  # 不确定
        "bbbp.csv": "p_np",
        "clintox.csv": "FDA_APPROVED",  # 不确定
        "HIV.csv": "HIV_active",
        "muv.csv": "MUV-466",  # 不确定
        "tox21.csv": "NR-AR",  # 不确定
        "sider.csv": "Hepatobiliary disorders",  # 不确定
        "esol.csv": "measured log solubility in mols per litre",  # 不确定,但有实验
        "freesolv.csv": "expt",  # 不确定,但有实验
        "lipo.csv": "exp"
    }
    try:
        return dataset_y[dataset_name]
    except KeyError:
        return ""


def get_score_method(dataset_name) -> int:
    """
    获取评分标准
    :param dataset_name:
    :return: int : 0是prc_auc，1是roc_auc，2是rmse（回归）
    """
    prc_auc = ["muv.csv"]
    roc_auc = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "tox21.csv", "sider.csv"]
    rmse = ["esol.csv", "freesolv.csv", "lipo.csv"]

    if dataset_name in prc_auc:
        return 0
    elif dataset_name in roc_auc:
        return 1
    elif dataset_name in rmse:
        return 2
    else:
        return 0


def get_split_length(dataset_name) -> int:
    """
    获取划分训练集合测试集的数量，一般给大数据集测试集 留 0.1%数量，小数据集留 40个测试
    :param dataset_name:
    :return: 划分数量
    """
    t = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv", "esol.csv",
         "freesolv.csv", "lipo.csv"]

    if dataset_name == "bace.csv":
        return 100
    elif dataset_name == "bbbp.csv":
        return 130
    elif dataset_name == "clintox.csv":
        return 190
    elif dataset_name == "HIV.csv":
        return 218
    elif dataset_name == "muv.csv":
        return 62
    elif dataset_name == "tox21.csv":
        return 190
    elif dataset_name == "sider.csv":
        return 218
    elif dataset_name == "esol.csv":
        return 48
    elif dataset_name == "freesolv.csv":
        return 48
    elif dataset_name == "lipo.csv":
        return 84


def task_type(dataset_name) -> int:
    """
    返回任务的种类，0为 分类，1为回归，-1 为未知
    :param dataset_name: 数据集文件名
    :return: type_code -> int: 0为 分类，1为回归,-1 为未知
    """
    classification_task = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
    regression_task = ["esol.csv", "freesolv.csv", "lipo.csv"]

    if dataset_name in classification_task:
        return 0
    elif dataset_name in regression_task:
        return 1
    else:
        return -1


def get_dataset_detail() -> dict:
    """
    获取所有数据集的信息，包括 类型（分类或回归）、y_title（输出层的列名）、评价标准
    :return:
    """
    datasets = get_all_dataset()

    dataset_detail = {}

    for dataset in datasets:
        dataset_detail[dataset] = {}

        # type 0为 分类，1为回归，-1 为未知
        dataset_detail[dataset]['type'] = task_type(dataset)

        dataset_detail[dataset]['y_title'] = get_y_column_title(dataset)

        # 0是prc_auc，1是roc_auc，2是rmse（回归）
        dataset_detail[dataset]['score_method'] = get_score_method(dataset)

        dataset_detail[dataset]['split_length'] = get_split_length(dataset)

    return dataset_detail


def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        # print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1] * len(ids)
    padding = [pad_index] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg


def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


class MTL_regression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MTL_regression, self).__init__()
        self.share_layer = nn.Sequential(
            # 第一个隐含层
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, tower_h1),
            nn.ReLU(),
            # nn.Linear(tower_h1, tower_h2),
            # nn.ReLU(),
        )
        # 回归预测层
        self.tower1 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, 1)
        )
        self.tower2 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, 1)
        )
        self.tower3 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, 1)
        )
        # self.predict = nn.Linear(hidden_size, 1)

    # 定义网络前向传播路径
    def forward(self, x):
        share_layer_output = (self.share_layer(x))

        out1 = self.tower1(share_layer_output)
        out2 = self.tower2(share_layer_output)
        out3 = self.tower3(share_layer_output)
        out = []
        out.append(out1[:, 0])
        out.append(out2[:, 0])
        out.append(out3[:, 0])
        # 输出一个一维向量
        return out


class MTL_classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super().__init__()

        self.share_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Linear(hidden_size, tower_h1),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Dropout(),
            # nn.Linear(tower_h1, tower_h2),
            # nn.ReLU(),
        )
        self.tower1 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )
        self.tower2 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )
        self.tower3 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )
        self.tower4 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )
        self.tower5 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )
        self.tower6 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )
        self.tower7 = nn.Sequential(
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Linear(tower_h2, num_classes)
        )

    def forward(self, x):
        share_layer_output = self.share_layer(x)
        out1 = self.tower1(share_layer_output)
        out2 = self.tower2(share_layer_output)
        out3 = self.tower3(share_layer_output)
        out4 = self.tower4(share_layer_output)
        out5 = self.tower5(share_layer_output)
        out6 = self.tower6(share_layer_output)
        out7 = self.tower7(share_layer_output)
        out = []
        out.append(out1)
        out.append(out2)
        out.append(out3)
        out.append(out4)
        out.append(out5)
        out.append(out6)
        out.append(out7)

        return out


def load_vocal():
    return WordVocab.load_vocab('vocab/vocab.pkl')


def load_transformer(vocab):
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('.save/trfm_12_23000.pkl'))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))
    return trfm


def train_test_split(data, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test


def prepare_data(dataset_name):
    seed = 30
    dataset_detail = get_dataset_detail()[dataset_name]
    df = pd.read_csv('dataset/{}'.format(dataset_name))
    if dataset_name == "bace.csv":
        df.rename(columns={"mol": "smiles"}, inplace=True)
    # 丢弃为 nan 的行
    df = df.dropna(axis=0, subset=[dataset_detail['y_title']])

    # # 取比较长的smiles 50个作为测试数据
    # df_train = df[np.array(list(map(len, df['smiles']))) <= dataset_detail["split_length"]]
    # df_test = df[np.array(list(map(len, df['smiles']))) > dataset_detail["split_length"]]
    #
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)

    x_split = [split(sm) for sm in df_train['smiles'].values]
    xid, _ = get_array(x_split)

    # X 为 1024 列的tensor
    X = trfm.encode(torch.t(xid))

    # Y 为 1024 列的tensor
    Y = df_train[dataset_detail["y_title"]].values

    x_split = [split(sm) for sm in df_test['smiles'].values]
    xid, _ = get_array(x_split)
    X_test = trfm.encode(torch.t(xid))
    Y_test = df_test[dataset_detail["y_title"]].values

    return X, Y, X_test, Y_test


def load_dataset(dataset_name):
    X, Y, X_test, Y_test = get_tensor_data(dataset_name)
    dataset_detail = get_dataset_detail()[dataset_name]

    return MyData(dataset_name, dataset_detail, trfm, X, Y)


def load_train_set_test_set(dataset_name):
    """

    :param dataset_name:
    :return: 返回DataLoader型的训练集和测试集
    """
    # 将文件 处理为 1024列的tensor

    # 需要重新改变种子获取tensor的时候使用这行
    # X, Y, X_test, Y_test = prepare_data(dataset_name)

    X, Y, X_test, Y_test = get_tensor_data(dataset_name)
    dataset_detail = get_dataset_detail()[dataset_name]
    # 构建训练和测试集
    dataloader = DataLoader(MyData(dataset_name, dataset_detail, trfm, X, Y), batch_size=batch_size)
    testloader = DataLoader(MyData(dataset_name, dataset_detail, trfm, X_test, Y_test), batch_size=batch_size)
    return dataloader, testloader


# def train_classificition(dataset_name1, dataset_name2, dataloader1, dataloader2):
#     """
#     对一个数据集进行训练
#     :param dataset_name:
#     :return: 这个数据集的测试集
#     """
#     print("Start Training {} and {}".format(dataset_name1.split(".")[0], dataset_name2.split(".")[0]))
#
#     # loss和优化器
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(mtl_classification.parameters(), lr=learning_rate, weight_decay=0.01)
#
#     dataloader = [dataloader1, dataloader2]
#     # 开始迭代
#     loss_list = []
#     for epoch in range(num_epochs):
#         # 每一次迭代都对所有数据集训练
#         loss = 0
#         for index,tt in enumerate(dataloader):
#             # 对每一batch的数据训练
#
#             for i, (data, label) in enumerate(tt):
#                 output1, output2 = mtl_classification(data.to(device))
#                 label = label.to(device)
#                 label = label.to(torch.int64)
#                 label_false = torch.zeros_like(label).to(torch.int64)
#                 if index == 0:
#                     loss1 = criterion(output1, label)
#                     loss2 = criterion(output2, label_false)
#                 else:
#                     loss1 = criterion(output1, label_false)
#                     loss2 = criterion(output2, label)
#                 loss = (loss1 + loss2) / 2
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#         loss_list.append(loss)
#         if epoch % 100 == 0:
#             print(
#                 'Epoch [{}/{}],loss: {:.8f}'.format(epoch , num_epochs, loss.item()))
#     return loss_list

# 所有数据集一起训练.每次加载一个数据集
# def train_regression(datasets_name,dataloaders):
#     """
#     对所有数据集进行训练
#     :param datasets_name:
#     :return: 所有数据集的loss
#     """
#     print("Start Training {}".format(datasets_name))
#     # loss和优化器
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.SGD(mtl_regression.parameters(), lr=0.001)
#     # 开始迭代
#     loss_list = []
#     for epoch in tqdm(range(num_epochs),colour="#29b7cb"):
#
#         train_loss = 0
#         for index,dataloader in enumerate(dataloaders):
#             # 对训练数据的加载器进行迭代计算
#             for step, (X, Y) in enumerate(dataloader):
#                 X = X.to(device)
#                 Y = Y.to(device)
#                 Y = Y.to(torch.float32)
#                 out_list = mtl_regression(X)  # MLP在训练batch上的输出
#                 '''
#                 5000 epoch
#                 esol 0.70
#                 freesolv 1.19
#                 lipo 4.51
#                 '''
#                 if index == 2:
#                     out_list[index] /= 5
#                 elif index == 1:
#                     out_list[index] /= 1.2
#                 loss = criterion(out_list[index], Y)  # 均方根损失函数
#
#
#                 optimizer.zero_grad()  # 每次迭代梯度初始化0
#                 loss.backward()  # 反向传播，计算梯度
#                 optimizer.step()  # 使用梯度进行优化
#                 train_loss = loss.item()
#         if train_loss < 0.001:
#             break
#         if epoch % 100 == 0:
#             print("Epoch {} / {} loss {}".format(epoch,num_epochs,train_loss))
#             loss_list.append(train_loss)
#     return loss_list



def train_regression(datasets_name,dataloaders):
    """
    对所有数据集进行训练
    :param datasets_name:
    :return: 所有数据集的loss
    """
    print("Start Training {}".format(datasets_name))
    # loss和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(mtl_regression.parameters(), lr=0.001)
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs),colour="#29b7cb"):
        train_loss = 0
        index_batch_list = mix_dataload_into_batches(dataloaders)
        for order_dict in index_batch_list:
            # 对训练数据的加载器进行迭代计算
            (data, label) = tuple(order_dict.values())[0]
            index = tuple(order_dict.keys())[0]
            data = data.to(device)
            label = label.to(device)
            label = label.to(torch.float32)
            out_list = mtl_regression(data)  # MLP在训练batch上的输出
            '''
            5000 epoch
            esol 0.70
            freesolv 1.19
            lipo 4.51
            '''
            # if index == 2:
            #     out_list[index] /= 5
            # elif index == 1:
            #     out_list[index] /= 1.2
            loss = criterion(out_list[index], label)  # 均方根损失函数


            optimizer.zero_grad()  # 每次迭代梯度初始化0
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss = loss.item()
        # if train_loss < 0.001:
        #     break
        if epoch % 100 == 0:
            print("Epoch {} / {} loss {}".format(epoch,num_epochs,train_loss))
            loss_list.append(train_loss)
    return loss_list
# 所有数据集一起训练.每次加载一个数据集
# def train_classificition(datasets_name, dataloaders):
#     """
#     所有数据集一起训练.每次加载一个数据集
#     :param dataset_name:
#     :return: 这个数据集的测试集
#     """
#     print("Start Training {} ".format(datasets_name))
#
#     # loss和优化器
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(mtl_classification.parameters(), lr=learning_rate)
#
#     # 开始迭代
#     loss_list = []
#     for epoch in tqdm(range(num_epochs),colour="#41ae3c"):
#         # 每一次迭代都对所有数据集训练
#         train_loss = 0
#         for index, dataloader in enumerate(dataloaders):
#             # 对每一batch的数据训练
#             for i, (data, label) in enumerate(dataloader):
#
#                 out_list = mtl_classification(data.to(device))
#                 label = label.to(device)
#                 label = label.to(torch.int64)
#                 # label_false = torch.zeros_like(label).to(torch.int64)
#                 if index == 0:
#                     out_list[index] *= 6
#                 elif index == 1:
#                     out_list[index] *= 4
#                 elif index == 2:
#                     out_list[index] *= 20
#                 elif index == 3:
#                     out_list[index] *= 1
#                 elif index == 4:
#                     out_list[index] *= 200
#                 elif index == 5:
#                     out_list[index] *= 4
#                 elif index == 6:
#                     out_list[index] *= 1
#                 loss = criterion(out_list[index], label)
#                 '''
#                 epoch 5000
#                 bace 0.03
#                 bbbp 0.05
#                 clintox 0.01
#                 HIV 0.17
#                 muv 0.001
#                 tox21 0.05
#                 sider 0.18
#                 '''
#
#                 # if index == 0:
#                 #     l = criterion(out_list[i + 1], label)
#                 #     # loss2 = criterion(output2, label_false)
#                 #     # loss3 = criterion(output3, label_false)
#                 # elif index == 1:
#                 #     # loss1 = criterion(output1, label_false)
#                 #     l = criterion(output2, label)
#                 #     # loss3 = criterion(output3, label_false)
#                 # else:
#                 #     # loss1 = criterion(output1, label_false)
#                 #     # loss2 = criterion(output2, label_false)
#                 #     l = criterion(output3, label)
#                 # loss = (loss1 + loss2 + loss3 * 2) / 4
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 train_loss = loss.item()
#         if train_loss < 0.002:
#             break
#         if epoch % 100 == 0:
#             print('Epoch [{}/{}],loss: {:.8f}'.format(epoch, num_epochs, train_loss))
#             loss_list.append(train_loss)
#
#     return loss_list

def judge_empty(dataloads):

    """
    判断嵌套列表是否全空
    exp:
    [[1,2],[3,4],[5,6,7,8],[9,10,11],[12]] return False
    [[],[]] return True
    :param dataloads:
    :return:
    """
    for dataload in dataloads:
        if len(dataload) != 0:
            return False
    return True

def mix_dataload_into_batches(dataloaders):
    '''
    将所有dataloads的数据按batch从第一个数据集开始依次排列

    :param dataloaders:
    :return: 返回batch字典,key为原数据集的index,value为batch
    '''

    # 列表,元素为字典,key为 dataloads的下标,value为对应batch
    index_batch_list = []
    new_dataloads = []
    for index, dataloader in enumerate(dataloaders):
        t = list(dataloader)
        new_dataloads.append(t)
    while not judge_empty(new_dataloads):
        for index,dataloader in enumerate(new_dataloads):
            if len(dataloader) != 0:
                index_batch_list.append({index:dataloader.pop()})
    # random.shuffle(index_batch_list)
    return index_batch_list

# 所有数据集一起训练.每次迭代一个batch一个数据集
def train_classificition(datasets_name, dataloaders):
    """
    所有数据集一起训练.每次迭代一个batch一个数据集
    :param dataset_name:
    :return: 这个数据集的测试集
    """
    print("Start Training {} ".format(datasets_name))

    # loss和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(mtl_classification.parameters(), lr=learning_rate)
    awl = UncertaintyWeightLoss(7)
    optimizer = torch.optim.Adam([
        {'params': mtl_classification.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ])
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs),colour="#41ae3c"):
        # 每一次迭代都对所有数据集训练
        train_loss = 0

        # index_batch_list exp:[{0: 2}, {1: 4}, {2: 8}, {3: 11}, {4: 12}, {0: 1}, {1: 3}, {2: 7}, {3: 10}, {2: 6}, {3: 9}, {2: 5}]
        # key 为index,value为batch
        index_batch_list = mix_dataload_into_batches(dataloaders)
        for order_dict in index_batch_list:

            (data, label) = tuple(order_dict.values())[0]
            index = tuple(order_dict.keys())[0]
            # 对每一batch的数据训练

            out_list = mtl_classification(data.to(device))
            label = label.to(device)
            label = label.to(torch.int64)
            # label_false = torch.zeros_like(label).to(torch.int64)

            # if index == 0:
            #     out_list[index] *= 6
            # elif index == 1:
            #     out_list[index] *= 4
            # elif index == 2:
            #     out_list[index] *= 20
            # elif index == 3:
            #     out_list[index] *= 1
            # elif index == 4:
            #     out_list[index] *= 200
            # elif index == 5:
            #     out_list[index] *= 4
            # elif index == 6:
            #     out_list[index] *= 1
            # loss_all = criterion(out_list[index], label)

            # loss = criterion(out_list[index], label)
            loss_temp = []
            for i in range(7):
                if i != index:
                    loss_temp.append(criterion(out_list[i],torch.zeros_like(label).to(torch.int64)))
                else:
                    loss_temp.append(criterion(out_list[index], label))
            loss = awl(loss_temp[0], loss_temp[1], loss_temp[2], loss_temp[3], loss_temp[4], loss_temp[5], loss_temp[6])
            # if index == 0:
            #     loss *= 6
            # elif index == 1:
            #     loss *= 4
            # elif index == 2:
            #     loss *= 20
            # elif index == 3:
            #     loss *= 1
            # elif index == 4:
            #     loss *= 200
            # elif index == 5:
            #     loss *= 4
            # elif index == 6:
            #     loss *= 1
            '''
            epoch 5000 
            bace 0.03
            bbbp 0.05
            clintox 0.01
            HIV 0.17
            muv 0.001
            tox21 0.05
            sider 0.18
            '''

            # if index == 0:
            #     l = criterion(out_list[i + 1], label)
            #     # loss2 = criterion(output2, label_false)
            #     # loss3 = criterion(output3, label_false)
            # elif index == 1:
            #     # loss1 = criterion(output1, label_false)
            #     l = criterion(output2, label)
            #     # loss3 = criterion(output3, label_false)
            # else:
            #     # loss1 = criterion(output1, label_false)
            #     # loss2 = criterion(output2, label_false)
            #     l = criterion(output3, label)
            # loss = (loss1 + loss2 + loss3 * 2) / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
        # if train_loss < 0.00002 :
        #     break
        if epoch % 2 == 0:
            print('Epoch [{}/{}],loss: {:.8f}'.format(epoch, num_epochs, train_loss))
            loss_list.append(train_loss)

    return loss_list

def get_regression_RMSE(testloader,model_name,index):
    model = torch.load('regression_model/regression_{}.pt'.format(model_name))
    model.eval()
    with torch.no_grad():
        prob_all = []
        label_all = []
        for (data, label) in (testloader):
            prob = model(data.to(device))[index]  # 表示模型的预测输出

            prob_all.extend(prob)

            label_all.extend(label)
        prob_all = [float(i) for i in prob_all]
        label_all = [float(i) for i in label_all]

        RMSE = np.sqrt(mean_squared_error(label_all, prob_all))

    return RMSE

def get_classification_accuracy(testloader, dataset_name, index):
    model = torch.load('classification_model/classification_{}.pt'.format(dataset_name))
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in testloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _, predicted = torch.max(output[index].data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
    return accuracy


def get_classification_auc_roc(testloader, dataset_name, index):
    model = torch.load('classification_model/classification_{}.pt'.format(dataset_name))
    model.eval()

    with torch.no_grad():
        prob_all = []
        label_all = []
        for (data, label) in (testloader):
            prob = model(data.to(device))[index]  # 表示模型的预测输出
            prob_all.extend(
                prob[:,1].cpu().numpy())  # prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
            label_all.extend(label)
        auc = roc_auc_score(label_all, prob_all)

    return auc


def get_tensor_data(dataset_name):
    file_name = dataset_name.split(".")[0]
    b = numpy.load("tensor_data/{}_tensor.npy".format(file_name), allow_pickle=True)
    c = [torch.tensor(x) for x in b]
    print("加载 {} tensor 数据成功".format(file_name))
    return c[0], c[1], c[2], c[3]


if __name__ == "__main__":
    test = {}
    testloader_list = []
    model_result = {}

    input_size = 1024
    hidden_size = 500
    num_classes = 2
    learning_rate = 0.001
    batch_size = 2000
    num_epochs = 1000
    tower_h1 = 200
    tower_h2 = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab = load_vocal()
    trfm = load_transformer(vocab)

    t = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv", "esol.csv",
         "freesolv.csv", "lipo.csv"]
    datasets_name = get_all_dataset()
    datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]

    # # 多任务模式1
    # multi_tasks = ["bbbp.csv", "clintox.csv"]
    #
    #
    # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
    #
    # mtl_classification.train()
    #
    # data_temp1, test_temp1 = load_train_set_test_set(multi_tasks[0])
    # data_temp2, test_temp2 = load_train_set_test_set(multi_tasks[1])
    # loss_list = train_classificition(multi_tasks[0], multi_tasks[1], data_temp1, data_temp2)
    # torch.save(mtl_classification, 'classification_{}.pt'.format("mtk4"))
    # plt.plot(loss_list)
    # plt.title("mtk4")
    # plt.show()

    # regression_model_name = "mtl16"
    # # 回归多任务模式2
    # multi_tasks = ["esol.csv","freesolv.csv", "lipo.csv"]
    # mtl_regression = MTL_regression(input_size, hidden_size).to(device)
    # mtl_regression.train()
    # datas = []
    # for index, multi_task in enumerate(multi_tasks):
    #     data, test = load_train_set_test_set(multi_task)
    #     datas.append(data)
    # loss_list = train_regression(multi_tasks, datas)
    # plt.plot(loss_list)
    # plt.title("regression_{}".format(regression_model_name))
    # plt.show()
    # torch.save(mtl_regression, 'regression_model/regression_{}.pt'.format(regression_model_name))


    classfication_model_name = "mtl37"
    # 分类多任务模式2
    multi_tasks = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
    mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
    mtl_classification.train()

    datas = []
    for index, multi_task in enumerate(multi_tasks):
        data, test = load_train_set_test_set(multi_task)
        datas.append(data)
    loss_list = train_classificition(multi_tasks, datas)
    plt.plot(loss_list)
    plt.title("classification_{}".format(classfication_model_name))
    plt.show()
    torch.save(mtl_classification, 'classification_model/classification_{}.pt'.format(classfication_model_name))


    with open("classification_model/result_new.txt", "a+", encoding="utf-8") as f:
        print(classfication_model_name,file=f)
        print(classfication_model_name)
        # 测试分类
        test_datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
        # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
        for index, test_dataset_name in enumerate(test_datasets_name):
            print("Start eval {}".format(test_dataset_name))
            print("Start eval {}".format(test_dataset_name),file=f)
            _,testloader = load_train_set_test_set(test_dataset_name)
            accuracy = get_classification_accuracy(testloader, classfication_model_name, index)
            auc_roc = get_classification_auc_roc(testloader, classfication_model_name, index)
            print("{} %".format(accuracy))
            print("{} %".format(accuracy),file=f)

            print("AUC:{:.4f}".format(auc_roc))
            print("AUC:{:.4f}".format(auc_roc),file=f)
        print(file=f)
    # 测试回归
    # with open("result1.txt","a+",encoding="utf-8") as f:
    #     print(regression_model_name, file=f)
    #     print(regression_model_name)
    #     test_datasets_name = ["esol.csv", "freesolv.csv", "lipo.csv"]
    #     # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
    #     for index,test_dataset_name in enumerate(test_datasets_name):
    #         file_name = test_dataset_name.split(".")[0]
    #         print("Start eval {}".format(file_name))
    #         print("Start eval {}".format(file_name),file=f)
    #         _,testloader = load_train_set_test_set(test_dataset_name)
    #         RMSE = get_regression_RMSE(testloader, regression_model_name, index)
    #         print("{} ".format(RMSE))
    #         print("{} ".format(RMSE),file=f)
    #     print(file=f)
'''

1. mtl3 是有["bace.csv", "bbbp.csv", "clintox.csv"]训练出的,其在bace auc效果略好.只计算一个loss,每个epoch训练所有数据集的batch
2. mtl4 所有分类,只带输出层,效果很差
3. mtl5 所有分类,每个任务输出层前加了一层,效果很差
4. mtl6 在 3 基础上,根据单个模型损失添加除法权重,1000次迭代,很差
5. mtl7 在 4 基础上,根据单个模型损失添加乘法权重,1000次迭代,很差,loss一直降不下去
6. mtl8 在 5 基础上,根据单个模型损失添加乘法权重,1000次迭代,删除L2正则化,效果不错!
7. mtl9 在 6 基础上,根据单个模型损失添加乘法权重,5000次迭代,删除L2正则化,效果一般!
8. mtl10 在 7 基础上,根据单个模型损失添加乘法权重,迭代到loss小于0.002,删除L2正则化,效果不错!
9. mtl11 在 8 基础上,根据单个模型损失添加乘法权重,迭代到loss小于0.002,删除L2正则化,按照每个数据集每个batch训练,不错!
10. mtl12 在 9 基础上,根据单个模型损失添加乘法权重,删除L2正则化,按照每个数据集每个batch训练,不错!
11. mtl13 在 10 基础上,根据单个模型损失添加乘法权重,迭代到loss小于0.00003,删除L2正则化,按照每个数据集每个batch训练,删除了一层,相当不错!!!!
12. mtl14 在 11 基础上,根据单个模型损失添加乘法权重,迭代到loss小于0.00003,删除L2正则化,按照每个数据集每个batch训练,删除了一层,一般
13. mtl15 在 12 基础上,根据单个模型损失添加乘法权重,迭代到loss小于0.00003,删除L2正则化,按照每个数据集每个batch训练,删除了一层,一般
14. mtl16 在 12 基础上,根据单个模型损失添加乘法权重,迭代到loss小于0.00003,删除L2正则化,按照每个数据集每个batch训练(随机),不好


1. mtl20 根据单个模型损失添加乘法权重,迭代2000次loss 0.00007680,删除L2正则化,按照每个数据集每个batch训练,删除了一层(1+2),一般
2. mtl22 根据单个模型损失添加乘法权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,(2+2),第一层drop,很差,说明不能drop
3. mtl23 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,(2+2),一般
4. mtl24 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到500(2+2),还行 sider好一点
5. mtl25 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),结果还不错,对于有的数据集比较好!!!
6. mtl26 根据单个模型损失添加乘法权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),一般
7. mtl27 根据单个模型损失添加乘法权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到5000(2+2),一般
8. mtl28 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到5000(2+2),一般
9. mtl29 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到10000(2+2),一般
10. mtl30 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2100(2+2)
11. mtl31 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2100(1+2),也不错
32. mtl32 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(1+3),hiddensize700,一般

33. mtl33 增加权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),hiddensize500,不错!!
34. mtl34 删除权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),hiddensize500,不错!

35. mtl35 loss计算所有数据集,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),hiddensize500,一般
36. mtl36 loss计算所有数据集带权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),hiddensize500,很差
37. mtl37 loss计算所有数据集带自动计算权重,迭代2000次,删除L2正则化,按照每个数据集每个batch训练,把batch加到2000(2+2),hiddensize500



'''

'''
分类不错的结果:
25, 31,33

'''

'''
1. mtl1 3个回归数据集,1000次迭代,效果正常
2. mtl2 3个回归数据集,5000次迭代
3. mtl3 3个回归数据集,2000次迭代,batch 200,很差

4. mtl4删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,相当不错
6. mtl6 删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 100,不如batch 200
7. mtl7 删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 300,不如batch 200
8. mtl8 删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 230,和200差不多
9. mtl9 删除权重,迭代10000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,不如迭代5000次
10. mtl10删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,相当不错
11. mtl11删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为3+1,还行
12. mtl12删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为3+1,hiddensize调成300,还行
13. mtl13删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为3+1,hiddensize调成700,最好的结果
14. mtl14删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为3+1,hiddensize调成800,一般
14. mtl14删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为3+1,hiddensize调成800,tower1 500,一般
15. mtl15删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为2+2,hiddensize调成700,一般
16. mtl16删除权重,迭代5000次,删除L2正则化,按照每个数据集每个batch训练,batch 200,改为2+2,hiddensize调成700,tower1 300,一般


'''

'''
回归不错的结果:
10,13
'''
