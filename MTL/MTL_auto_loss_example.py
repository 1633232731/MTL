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
        # self.tower3 = nn.Sequential(
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     nn.Linear(tower_h2, num_classes)
        # )
        # self.tower4 = nn.Sequential(
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     nn.Linear(tower_h2, num_classes)
        # )
        # self.tower5 = nn.Sequential(
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     nn.Linear(tower_h2, num_classes)
        # )
        # self.tower6 = nn.Sequential(
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     nn.Linear(tower_h2, num_classes)
        # )
        # self.tower7 = nn.Sequential(
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     nn.Linear(tower_h2, num_classes)
        # )

    def forward(self, x):
        share_layer_output = self.share_layer(x)
        out1 = self.tower1(share_layer_output)
        out2 = self.tower2(share_layer_output)
        # out3 = self.tower3(share_layer_output)
        # out4 = self.tower4(share_layer_output)
        # out5 = self.tower5(share_layer_output)
        # out6 = self.tower6(share_layer_output)
        # out7 = self.tower7(share_layer_output)
        out = []
        out.append(out1)
        out.append(out2)
        # out.append(out3)
        # out.append(out4)
        # out.append(out5)
        # out.append(out6)
        # out.append(out7)

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


def train_regression(mtl_regression, datasets_name, dataloaders):
    """
    对所有数据集进行训练
    :param datasets_name:
    :return: 所有数据集的loss
    """
    print("Start Training {}".format(datasets_name))
    # loss和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(mtl_regression.parameters(), lr=learning_rate)

    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs), colour="#29b7cb"):
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
            # loss = criterion(out_list[index], label)  # 均方根损失函数
            loss_temp = []

            # # 将不对应batch的label由0改为label的均值
            # label_average = sum(label) / len(label)
            # t = []
            # for i in range(len(label)):
            #     t.append(label_average.item())
            # a = np.array(t)
            # average = torch.from_numpy(a).to(device)
            #
            # not_match_label = average
            not_match_label = torch.zeros_like(label)

            for i in range(3):
                if i != index:
                    loss_temp.append(criterion(out_list[i], not_match_label.to(torch.float32)))
                else:
                    loss_temp.append(criterion(out_list[index], label))
            # loss = (loss_temp[0] * 4  + loss_temp[1] + loss_temp[2] * 6)/3
            loss = (loss_temp[0] + loss_temp[1] + loss_temp[2]) / 3

            optimizer.zero_grad()  # 每次迭代梯度初始化0
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss = loss.item()
        # if train_loss < 0.001:
        #     break
        if epoch % 100 == 0:
            print("Epoch {} / {} loss {}".format(epoch, num_epochs, train_loss))
            loss_list.append(train_loss)
    return loss_list


def train_regression_with_auto_loss(mtl_regression, datasets_name, dataloaders):
    """
    对所有数据集进行训练
    :param datasets_name:
    :return: 所有数据集的loss
    """
    print("Start Training {}".format(datasets_name))
    # loss和优化器
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(mtl_regression.parameters(), lr=0.001)
    awl = UncertaintyWeightLoss(len(datasets_name))
    optimizer = torch.optim.Adam([
        {'params': mtl_regression.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=learning_rate)
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs), colour="#29b7cb"):
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
            loss_temp = []
            for i in range(len(datasets_name)):
                if i != index:
                    loss_temp.append(criterion(out_list[i], torch.zeros_like(label).to(torch.float32)))
                else:
                    loss_temp.append(criterion(out_list[index], label))
            loss = awl(loss_temp[0], loss_temp[1], loss_temp[2])

            optimizer.zero_grad()  # 每次迭代梯度初始化0
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss = loss.item()
        # if train_loss < 0.001:
        #     break
        if epoch % 100 == 0:
            print("Epoch {} / {} loss {}".format(epoch, num_epochs, train_loss))
            loss_list.append(train_loss)
    return loss_list


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
        for index, dataloader in enumerate(new_dataloads):
            if len(dataloader) != 0:
                index_batch_list.append({index: dataloader.pop()})
    # random.shuffle(index_batch_list)
    return index_batch_list


# 所有数据集一起训练.每次迭代一个batch一个数据集
def train_classificition(mtl_classification, datasets_name, dataloaders):
    """
    所有数据集一起训练.每次迭代一个batch一个数据集
    :param dataset_name:
    :return: 这个数据集的测试集
    """
    print("Start Training {} ".format(datasets_name))

    # loss和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mtl_classification.parameters(), lr=learning_rate)
    # awl = AutomaticWeightedLoss(1)
    # optimizer = torch.optim.Adam([
    #     {'params': mtl_classification.parameters()},
    #     {'params': awl.parameters(), 'weight_decay': 0},
    # ], lr=learning_rate)
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs), colour="#41ae3c"):
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

            loss_temp = []
            for i in range(2):
                if i != index:
                    loss_temp.append(criterion(out_list[i], torch.zeros_like(label).to(torch.int64)))
                else:
                    loss_temp.append(criterion(out_list[index], label))
            # loss = (loss_temp[0] * 17 + loss_temp[1]) / 2
            if index == 0:
                loss = (criterion(out_list[index], label)) * 100
            else:
                loss = (criterion(out_list[index], label)) * 0.01
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


def train_classificition_with_auto_loss(mtl_classification, datasets_name, dataloaders):
    """
    所有数据集一起训练.每次迭代一个batch一个数据集
    :param dataset_name:
    :return: 这个数据集的测试集
    """
    print("Start Training {} ".format(datasets_name))

    # loss和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(mtl_classification.parameters(), lr=learning_rate)
    awl = UncertaintyWeightLoss(2)
    optimizer = torch.optim.Adam([
        {'params': mtl_classification.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0},
    ], lr=learning_rate)
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs), colour="#41ae3c"):
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

            loss_temp = []
            for i in range(2):
                if i != index:
                    loss_temp.append(criterion(out_list[i], torch.zeros_like(label).to(torch.int64)))
                else:
                    loss_temp.append(criterion(out_list[index], label))
            loss = awl(loss_temp[0], loss_temp[1])
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


def get_regression_RMSE(testloader, model_name, index):
    model = torch.load('regression_auto_loss_model/regression_{}.pt'.format(model_name))
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
    model = torch.load('classification_auto_loss_model/classification_{}.pt'.format(dataset_name))
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
    model = torch.load('classification_auto_loss_model/classification_{}.pt'.format(dataset_name))
    model.eval()

    with torch.no_grad():
        prob_all = []
        label_all = []
        for (data, label) in (testloader):
            prob = model(data.to(device))[index]  # 表示模型的预测输出
            prob_all.extend(
                prob[:, 1].cpu().numpy())  # prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
            label_all.extend(label)
        auc = roc_auc_score(label_all, prob_all)

    return auc


def get_tensor_data(dataset_name):
    file_name = dataset_name.split(".")[0]
    b = numpy.load("tensor_data/{}_tensor.npy".format(file_name), allow_pickle=True)
    c = [torch.tensor(x) for x in b]
    print("加载 {} tensor 数据成功".format(file_name))
    return c[0], c[1], c[2], c[3]


def regression_mode(multi_tasks):
    # 回归多任务模式2
    mtl_regression = MTL_regression(input_size, hidden_size).to(device)
    mtl_regression.train()
    datas = []
    for index, multi_task in enumerate(multi_tasks):
        data, test = load_train_set_test_set(multi_task)
        datas.append(data)
    loss_list = train_regression(mtl_regression,multi_tasks, datas)
    # loss_list = train_regression_with_auto_loss(mtl_regression, multi_tasks, datas)
    plt.plot(loss_list)
    plt.title("regression_{}".format(regression_model_name))
    plt.show()
    torch.save(mtl_regression, 'regression_auto_loss_model/regression_{}.pt'.format(regression_model_name))


def regression_test(test_datasets_name):
    # 测试回归
    with open("regression_auto_loss_model/result1.txt", "a+", encoding="utf-8") as f:
        print(regression_model_name, file=f)
        print(regression_model_name)

        # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
        for index, test_dataset_name in enumerate(test_datasets_name):
            file_name = test_dataset_name.split(".")[0]
            print("Start eval {}".format(file_name))
            print("Start eval {}".format(file_name), file=f)
            _, testloader = load_train_set_test_set(test_dataset_name)
            RMSE = get_regression_RMSE(testloader, regression_model_name, index)
            print("{} ".format(RMSE))
            print("{} ".format(RMSE), file=f)
        print(file=f)


def classification_mode(multi_tasks):
    # 分类多任务模式2

    mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
    mtl_classification.train()

    datas = []
    for index, multi_task in enumerate(multi_tasks):
        data, test = load_train_set_test_set(multi_task)
        datas.append(data)
    loss_list = train_classificition(mtl_classification,multi_tasks, datas)
    # loss_list = train_classificition_with_auto_loss(mtl_classification, multi_tasks, datas)
    # loss_list = train_classificition_with_grad_norm_loss(mtl_classification,multi_tasks, datas)
    plt.plot(loss_list)
    plt.title("classification_{}".format(classfication_model_name))
    plt.show()
    torch.save(mtl_classification,
               'classification_auto_loss_model/classification_{}.pt'.format(classfication_model_name))


def classification_test(test_datasets_name):
    with open("classification_auto_loss_model/result_new.txt", "a+", encoding="utf-8") as f:
        print(classfication_model_name, file=f)
        print(classfication_model_name)
        # 测试分类

        # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
        for index, test_dataset_name in enumerate(test_datasets_name):
            print("Start eval {}".format(test_dataset_name))
            print("Start eval {}".format(test_dataset_name), file=f)
            _, testloader = load_train_set_test_set(test_dataset_name)
            accuracy = get_classification_accuracy(testloader, classfication_model_name, index)
            auc_roc = get_classification_auc_roc(testloader, classfication_model_name, index)
            print("{} %".format(accuracy))
            print("{} %".format(accuracy), file=f)

            print("AUC:{:.4f}".format(auc_roc))
            print("AUC:{:.4f}".format(auc_roc), file=f)
        print(file=f)


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

    # # # 回归
    # regression_model_name = "mtl_f-3-1"
    # # multi_tasks = ["freesolv.csv", "lipo.csv","esol.csv"]
    # multi_tasks = ["freesolv.csv", "lipo.csv", "esol.csv"]
    # regression_mode(multi_tasks)
    # # test_datasets_name = ["freesolv.csv", "lipo.csv","esol.csv"]
    # test_datasets_name = ["freesolv.csv", "lipo.csv", "esol.csv"]
    # regression_test(test_datasets_name)

    # 分类
    classfication_model_name = "mtl4-6"
    # multi_tasks = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
    multi_tasks = ["clintox.csv","bbbp.csv"]
    classification_mode(multi_tasks)
    # test_datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
    test_datasets_name = ["clintox.csv", "bbbp.csv"]
    classification_test(test_datasets_name)

'''
分类:
["clintox.csv","sider.csv"]
1. 平均分loss, epoch 1000,batch2000,差不多
2. 自动loss, epoch 1000,batch2000,差不多
3. 按比例手动loss, epoch 1000,batch2000,差一点

4. 平均分loss, epoch 2000,batch2000,差不多
5. 自动loss, epoch 2000,batch2000,差不多
6. 按比例手动loss, epoch 2000,batch2000,差一点

7. 平均分loss, epoch 1000,batch200,差不多
8. 自动loss, epoch 1000,batch200,无结果
9. 按比例手动loss, epoch 1000,batch200,差一点

得出结论 
epoch 1000,batch2000

["clintox.csv","bbbp.csv"]
2-1. 平均分loss, epoch 1000,batch200比较好
2-2. 按比例手动loss, epoch 1000,batch200比较差
2-3. 自动loss, epoch 1000,batch200,不出结果

2-4. 平均分loss, epoch 1000,batch2000,还行
2-5. 按比例手动loss, epoch 1000,batch2000,还行
2-6. 自动loss, epoch 1000,batch2000,最好
2-6-1. 自动loss, epoch 1000,batch2000,重做一下上面的.还行
2-6-2. 自动loss, epoch 1000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,最好!!
2-6-3. 自动loss, epoch 1000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,加个dropout试图改善loss突然上升,不如不加.还行
2-6-4. 自动loss, epoch 1000,batch12,重做一下上面的. 把网络从7个输出改为2个,优化器改为2.还行
2-6-5. 自动loss, epoch 1000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2.还行
2-6-6. 自动loss, epoch 1000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,看了人家的经验将学习率修改为0.003,还行
2-6-7. 自动loss, epoch 1000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,看了人家的经验将学习率修改为0.0005,不错
2-6-8. 自动loss, epoch 2000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,看了人家的经验将学习率修改为0.0001,不错
2-6-9. 自动loss, epoch 5000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,看了人家的经验将学习率修改为0.0001,不错
2-6-10. 自动loss, epoch 5000,batch2000,重做一下上面的. 把网络从7个输出改为2个,优化器改为2,看了人家的经验将学习率修改为0.00015,还行


["clintox.csv","HIV.csv"]
3-1 自己的loss,最好
3-2 平均分loss,HIV略好,clintox差得远
3-3 手动比例分loss,都不好
3-4 自动loss,epoch 2000,batch2000,学习率0.001,HIV略好,clintox差得远
3-5 自动loss,epoch 2000,batch2000,学习率0.0005,HIV略好,clintox差得远

不错的自动结果
2-6-2
2-6-7


说明自动loss 好
131415

'''

'''
回归

1-1 自己的loss,不错
2-1 所有loss平均,很差
2-2 所有loss加权平均,比上面好一点,也很差
2-3 所有loss自动平均,非常差
2-4 所有loss加权平均,标准不用0,而是用label的平均值,非常差


input_size = 1024
hidden_size = 500
num_classes = 2
learning_rate = 0.001
batch_size = 2000
num_epochs = 2000
tower_h1 = 200
tower_h2 = 50

mtl_f-1-1 三个任务一起,人工配loss

mtl_f-2-1 三个任务一起,平均 loss

mtl_f-3-1 三个任务一起,autoweight loss
'''

'''
1. 多任务 自己的loss的问题
2. 目前完成了两种均衡loss 的方案,其中在回归问题上 gradnorm比较好,在分类问题上



'''


'''
["clintox.csv","bbbp.csv"]
4-1 多任务自己的loss,结果非常好
4-2 多任务自动loss,一般
4-3 多任务自己的loss + autoweight,还行
4-4 多任务自己的loss 加个测试比例测试这个loss 的比例有没有意义
4-5 多任务自己的loss 加个测试比例测试这个loss 的比例有没有意义,和上面一致
4-6 多任务自己的loss 加个测试比例测试这个loss 的比例有没有意义,和上面比例反过来



'''