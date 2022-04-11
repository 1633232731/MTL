import argparse
import os
import time
from typing import List

import numpy
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
from dataset import MyData
from utils import split
from sklearn.metrics import mean_squared_error, precision_recall_curve, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from tqdm import tqdm

from sklearn.utils import shuffle as reset

import torch.nn.functional as F

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4


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
    def __init__(self, input_size, hidden_size_1,hidden_size_2,hidden_size_3):
        super(MTL_regression, self).__init__()
        self.regression = nn.Sequential(
            # 第一个隐含层
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, 1)
        )
    # 定义网络前向传播路径
    def forward(self, x):
        output = self.regression(x)
        # 输出一个一维向量
        return output[:, 0]


class MTL_classification(nn.Module):
    def __init__(self, input_size, hidden_size_1,hidden_size_2,hidden_size_3, num_classes) -> None:
        super().__init__()

        self.classification = nn.Sequential(
        nn.Linear(input_size, hidden_size_1),  # 输入层到隐藏层
        nn.ReLU(),
        nn.Linear(hidden_size_1, hidden_size_2), # 隐藏层到输出层
        nn.ReLU(),
        nn.Linear(hidden_size_2, hidden_size_3),
        nn.ReLU(),
        nn.Linear(hidden_size_3, num_classes)
        # nn.Dropout(p=0.01),  # dropout
        )

    def forward(self, x):
        out = self.classification(x)
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
    df_train,df_test = train_test_split(df,test_size=0.2,shuffle=True,random_state=seed)

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

    return X,Y,X_test,Y_test

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
    return dataloader,testloader


def train_regression(dataset_name,dataloader):
    """
    对一个数据集进行训练
    :param dataset_name:
    :return: 这个数据集的loss
    """
    print("Start Training {}".format(dataset_name.split(".")[0]))
    # loss和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(mtl_regression.parameters(), lr=0.001,weight_decay=0.0001)
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs),colour="#29b7cb"):
        train_loss = 0
        # 对训练数据的加载器进行迭代计算
        for step, (X, Y) in enumerate(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            Y = Y.to(torch.float32)
            output = mtl_regression(X)  # MLP在训练batch上的输出
            loss = criterion(output, Y)  # 均方根损失函数
            optimizer.zero_grad()  # 每次迭代梯度初始化0
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss += loss.item()
        loss_list.append(train_loss)
        if epoch % 100 == 0:
            print("Epoch {} / {} loss {}".format(epoch+1,num_epochs,train_loss))
    return loss_list


def train_classification(dataset_name, dataloader):
    """
    对一个数据集进行训练
    :param dataset_name:
    :return: 这个数据集的loss
    """
    print("Start Training {}".format(dataset_name.split(".")[0]))

    # loss和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mtl_classification.parameters(), lr=learning_rate,weight_decay=0.01)
    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs),colour="#41ae3c"):
        # 对每一batch的数据训练
        loss_temp = 0
        for i, (data, label) in enumerate(dataloader):
            output = mtl_classification(data.to(device))
            label = label.to(device)
            label = label.to(torch.int64)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_temp = loss.item()
        if epoch % 100 == 0:
            print('{} Epoch [{}/{}],loss: {:.8f}'.format(dataset_name,epoch + 1, num_epochs, loss_temp))
        loss_list.append(loss_temp)
    return loss_list




def get_classification_accuracy(testloader,dataset_name):
    model = torch.load('single_model/classification_single_model/classification_{}.pt'.format(dataset_name))
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in testloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
    return accuracy


def get_classification_auc_roc(testloader,dataset_name):
    model = torch.load('single_model/classification_single_model/classification_{}.pt'.format(dataset_name))
    model.eval()

    with torch.no_grad():
        prob_all = []
        label_all = []
        for (data, label) in (testloader):

            prob = model(data.to(device))  # 表示模型的预测输出
            prob_all.extend(
                prob[:, 1].cpu().numpy())  # prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
            label_all.extend(label)
        # print(label_all)
        # print("----------------------")
        # print(prob_all)
        auc = roc_auc_score(label_all, prob_all)

    return auc


def get_regression_RMSE(testloader,dataset_name):
    model = torch.load('single_model/regression_single_model/regression_{}.pt'.format(dataset_name))
    model.eval()
    with torch.no_grad():
        prob_all = []
        label_all = []
        for (data, label) in (testloader):
            prob = model(data.to(device))  # 表示模型的预测输出

            prob_all.extend(prob)

            label_all.extend(label)
        prob_all = [float(i) for i in prob_all]
        label_all = [float(i) for i in label_all]

        RMSE = np.sqrt(mean_squared_error(label_all, prob_all))

    return RMSE

def get_tensor_data(dataset_name):
    file_name = dataset_name.split(".")[0]
    b = numpy.load("tensor_data/{}_tensor.npy".format(file_name), allow_pickle=True)
    c = [torch.tensor(x) for x in b]
    print("加载 {} tensor 数据成功".format(file_name))
    return c[0],c[1],c[2],c[3]




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='单任务模式')
    parser.add_argument('--num_epochs', '-n', type=int, default=2000, help="迭代次数")
    parser.add_argument('--mode', '-m', choices=('train', 'test'), default='test',
                        help="模式选择")
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001, help="学习率")
    parser.add_argument('--batch_size', '-b', type=int, default=200, help="batch大小")
    parser.add_argument('--hidden_size_1', '-h1', type=int, default=500, help="第二层神经元数量")
    parser.add_argument('--hidden_size_2', '-h2', type=int, default=256, help="第三层神经元数量")
    parser.add_argument('--hidden_size_3', '-h3', type=int, default=128, help="第四层神经元数量")

    args = parser.parse_args()

    input_size = 1024
    hidden_size_1 = args.hidden_size_1
    hidden_size_2 = args.hidden_size_2
    hidden_size_3 = args.hidden_size_3
    num_classes = 2
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    mode = args.mode


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab = load_vocal()
    trfm = load_transformer(vocab)

    datasets_name = get_all_dataset()

    if mode == "train":
        t = input("Please save trained model somewhere else,press enter to continue.")

        datasets_name = ["esol.csv", "freesolv.csv", "lipo.csv"]
        # 单任务模式,回归
        for index, dataset_name in enumerate(datasets_name):
            file_name = dataset_name.split(".")[0]
            mtl_regression = MTL_regression(input_size, hidden_size_1,hidden_size_2,hidden_size_3).to(device)
            # 设置为 训练 模式
            mtl_regression.train()

            dataloader, testloader = load_train_set_test_set(dataset_name)

            loss_list = train_regression(dataset_name, dataloader)
            # testloader_list.append(testloader)
            # model_result["name"] = dataset_name
            plt.plot(loss_list)
            plt.title(dataset_name.split(".")[0])
            plt.show()
            print("save {} model".format(dataset_name))
            torch.save(mtl_regression, 'single_model/regression_single_model/regression_{}.pt'.format(file_name))

        datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
        # 单任务模式,分类
        for index,dataset_name in enumerate(datasets_name):
            file_name = dataset_name.split(".")[0]
            # 定义模型
            mtl_classification = MTL_classification(input_size, hidden_size_1,hidden_size_2,hidden_size_3, num_classes).to(device)
            # 设置为 训练 模式
            mtl_classification.train()

            dataloader, testloader = load_train_set_test_set(dataset_name)

            loss_list = train_classification(dataset_name, dataloader)
            # testloader_list.append(testloader)
            # model_result["name"] = dataset_name
            plt.plot(loss_list)
            plt.title(dataset_name.split(".")[0])
            plt.show()
            print("save {} model".format(dataset_name))
            torch.save(mtl_classification, 'single_model/classification_single_model/classification_{}.pt'.format(file_name))
            print("test {} model".format(dataset_name))
            # for index,testloader in enumerate(testloader_list):
            accuracy = get_classification_accuracy(testloader,file_name)
            auc_roc = get_classification_auc_roc(testloader,file_name)
            print("{} %".format(accuracy))
            print("AUC:{:.4f}".format(auc_roc))
    else:
        # 测试分类
        test_datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
        for test_dataset_name in test_datasets_name:
            file_name = test_dataset_name.split(".")[0]
            print("Start eval {}".format(file_name))
            _,testloader = load_train_set_test_set(test_dataset_name)
            accuracy = get_classification_accuracy(testloader, file_name)
            if test_dataset_name == "muv.csv":
                score = get_classification_auc_roc(testloader, file_name)
            else:
                score = get_classification_auc_roc(testloader, file_name)
            print("{} %".format(accuracy))
            print("AUC:{:.4f}".format(score))

        # 测试回归
        test_datasets_name = ["esol.csv", "freesolv.csv", "lipo.csv"]
        for test_dataset_name in test_datasets_name:
            file_name = test_dataset_name.split(".")[0]
            print("Start eval {}".format(file_name))
            _,testloader = load_train_set_test_set(test_dataset_name)
            RMSE = get_regression_RMSE(testloader, file_name)
            print("{} ".format(RMSE))

    os.system("pause")