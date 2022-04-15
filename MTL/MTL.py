import argparse
import os
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

from model import GradNormLossModel, GradNormLossTrain, MTLRegression, MTLClassification

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


def judge_empty(dataloaders):
    """
    判断嵌套列表是否全空
    exp:
    [[1,2],[3,4],[5,6,7,8],[9,10,11],[12]] return False
    [[],[]] return True
    :param dataloads:
    :return:
    """
    for dataloader in dataloaders:
        if len(dataloader) != 0:
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


def train_model(mtl_model, datasets_name, dataloaders, loss_type, mode):
    """
       对所有数据集进行训练
       :param mtl_model: 模型
       :param datasets_name: 任务名字列表
       :param dataloaders:
       :param loss_type: 损失函数的种类,0是计算自己的loss,1是全部loss平均加权,2是全部loss经验比例加权,3是uncertainty weight比例加权,4是在自己的loss 上使用 uncertainty weight
       :param mode 0是回归,1是分类
       :return: 所有数据集的loss
       """
    print("Start Training {}".format(datasets_name))
    '''
    回归
    5000 epoch
    esol 0.70
    freesolv 1.19
    lipo 4.51

    分类
    epoch 5000 
    bace 0.03
    bbbp 0.05
    clintox 0.01
    HIV 0.17
    muv 0.001
    tox21 0.05
    sider 0.18
    '''
    dataset_rate = {
        "bace.csv": 1 / 0.03,
        "bbbp.csv": 1 / 0.05,
        "clintox.csv": 1 / 0.01,
        "HIV.csv": 1 / 0.17,
        "muv.csv": 1 / 0.001,
        "tox21.csv": 1 / 0.05,
        "sider.csv": 1 / 0.18,
        "esol.csv": 1 / 0.7,
        "freesolv.csv": 1 / 1.19,
        "lipo.csv": 1 / 4.51,
    }
    t = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv", "esol.csv",
         "freesolv.csv", "lipo.csv"]


    num_tasks = len(datasets_name)
    # loss和优化器
    if mode == 0:
        # 回归
        loss_function = nn.MSELoss().to(device)
        if loss_type == 0 or loss_type == 1 or loss_type == 2:
            optimizer = torch.optim.SGD(mtl_model.parameters(), lr=learning_rate)
        else:
            awl = UncertaintyWeightLoss(num_tasks)
            optimizer = torch.optim.SGD([
                {'params': mtl_model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0},
            ], lr=learning_rate)
        color = "#29b7cb"
        label_type = torch.float32
    else:
        # 分类
        loss_function = nn.CrossEntropyLoss().to(device)
        if loss_type == 0 or loss_type == 1 or loss_type == 2:
            optimizer = torch.optim.Adam(mtl_model.parameters(), lr=learning_rate)
        else:
            awl = UncertaintyWeightLoss(num_tasks)
            optimizer = torch.optim.Adam([
                {'params': mtl_model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0},
            ], lr=learning_rate)
        color = "#41ae3c"
        label_type = torch.int64

    # 开始迭代
    loss_list = []
    for epoch in tqdm(range(num_epochs), colour=color):
        train_loss = 0
        index_batch_list = mix_dataload_into_batches(dataloaders)
        for order_dict in index_batch_list:
            # 对训练数据的加载器进行迭代计算
            (data, label) = tuple(order_dict.values())[0]
            index = tuple(order_dict.keys())[0]
            data = data.to(device)
            label = label.to(device)
            label = label.to(label_type)

            out_list = mtl_model(data)  # MLP在训练batch上的输出

            # loss_type: 损失函数的种类,0是计算自己的loss,1是全部loss平均加权,2是全部loss经验比例加权,3是uncertainty weight比例加权,4是在自己的loss 上使用 uncertainty weight


            if loss_type == 0:
                # 自己的loss
                loss = loss_function(out_list[index], label)
            else:
                # 除了自己的loss 都需要 not_match_label和loss_temp
                not_match_label = torch.zeros_like(label).to(label_type)
                loss_temp = []
                if loss_type == 1 or loss_type == 2 or loss_type == 3:
                    # 1是全部loss平均加权,2是全部loss经验比例加权
                    for i in range(num_tasks):
                        if i != index:
                            loss_temp.append(loss_function(out_list[i], not_match_label))
                        else:
                            loss_temp.append(loss_function(out_list[index], label))
                    if loss_type == 1:
                        loss = sum(loss_temp) / num_tasks
                    elif loss_type == 2:
                        loss = 0
                        for t,dataset_name in enumerate(datasets_name):
                            loss += loss_temp[t] * dataset_rate[dataset_name]
                        # loss = sum(loss_temp) / num_tasks
                    elif loss_type == 3:
                        loss = awl(loss_temp)
                else:
                    # TODO: 是在自己的 loss 上使用 uncertainty weight
                    loss = awl(loss_temp)

            optimizer.zero_grad()  # 每次迭代梯度初始化0
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss = loss.item()
        # if train_loss < 0.001:
        #     break
        if epoch % 100 == 0:
            print("Epoch {} / {} loss {:.10f}".format(epoch, num_epochs, train_loss))
            loss_list.append(train_loss)
    return loss_list


def get_regression_RMSE(testloader, model_name, index, model_save_path):
    model = torch.load('{}/regression_{}.pt'.format(model_save_path, model_name))
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


def get_classification_accuracy(testloader, dataset_name, index, model_save_path):
    model = torch.load('{}/classification_{}.pt'.format(model_save_path, dataset_name))
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


def get_classification_auc_roc(testloader, dataset_name, index, model_save_path):
    model = torch.load('{}/classification_{}.pt'.format(model_save_path, dataset_name))
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


def grad_norm_loss(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes, model_save_path,
                   model_name, datas):
    """

    :param multi_tasks:
    :param mode: 0 是回归,1是分类
    :return:
    """
    n_tasks = len(multi_tasks)

    # datas = []
    # for index, multi_task in enumerate(multi_tasks):
    #     data, test = load_train_set_test_set(multi_task)
    #     datas.append(data)

    model = GradNormLossModel(n_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes)

    train = GradNormLossTrain(model, mode, n_tasks)

    if torch.cuda.is_available():
        train.cuda()
        model.cuda()
    optimizer = torch.optim.Adam(train.parameters(), lr=learning_rate)
    weights = []
    task_losses = []
    loss_ratios = []
    grad_norm_losses = []
    if mode == 0:
        label_type = torch.float32
    else:
        label_type = torch.int64
    # run n_iter iterations of training
    for t in tqdm(range(num_epochs), colour="#29b7cb"):
        # get a single batch
        index_batch_list = mix_dataload_into_batches(datas)
        for order_dict in index_batch_list:
            # 对训练数据的加载器进行迭代计算
            (X, target) = tuple(order_dict.values())[0]
            index = tuple(order_dict.keys())[0]
            target = target.to(label_type)
            if torch.cuda.is_available():
                X = X.cuda()
                target = target.cuda()
            # evaluate each task loss L_i(t)
            task_loss = train(X, target,
                              index)  # this will do a forward pass in the model and will also evaluate the loss
            # compute the weighted loss w_i(t) * L_i(t)
            weighted_task_loss = torch.mul(train.weights, task_loss)
            # initialize the initial loss L(0) if t=0
            if t == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                initial_task_loss = initial_task_loss.numpy()
            # get the total loss
            loss = torch.sum(weighted_task_loss)
            # clear the gradients
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            # This is equivalent to compute each \nabla_W L_i(t)
            loss.backward(retain_graph=True)

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            # print('Before turning to 0: {}'.format(model.weights.grad))
            train.weights.grad.data = train.weights.grad.data * 0.0
            # print('Turning to 0: {}'.format(model.weights.grad))

            # get layer of shared weights
            W = train.get_last_shared_layer()

            # get the gradient norms for each of the tasks
            # G^{(i)}_w(t)
            norms = []
            for i in range(len(task_loss)):
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                # compute the norm
                norms.append(torch.norm(torch.mul(train.weights[i], gygw[0])))
            norms = torch.stack(norms)
            # print('G_w(t): {}'.format(norms))

            # compute the inverse training rate r_i(t)
            # \curl{L}_i
            if torch.cuda.is_available():
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            else:
                loss_ratio = task_loss.data.numpy() / initial_task_loss
            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)
            # print('r_i(t): {}'.format(inverse_train_rate))

            # compute the mean norm \tilde{G}_w(t)
            if torch.cuda.is_available():
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())
            # print('tilde G_w(t): {}'.format(mean_norm))

            # compute the GradNorm loss
            # this term has to remain constant
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
            if torch.cuda.is_available():
                constant_term = constant_term.cuda()
            # print('Constant term: {}'.format(constant_term))
            # this is the GradNorm loss itself
            grad_norm_loss = torch.as_tensor(torch.sum(torch.abs(norms - constant_term)))
            # print('GradNorm loss {}'.format(grad_norm_loss))

            # compute the gradient for the weights
            train.weights.grad = torch.autograd.grad(grad_norm_loss, train.weights)[0]

            # do a step with the optimizer
            optimizer.step()

        # renormalize
        normalize_coeff = n_tasks / torch.sum(train.weights.data, dim=0)
        train.weights.data = train.weights.data * normalize_coeff
        # record
        if torch.cuda.is_available():
            task_losses.append(task_loss.data.cpu().numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(train.weights.data.cpu().numpy())
            grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
        else:
            task_losses.append(task_loss.data.numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(train.weights.data.numpy())
            grad_norm_losses.append(grad_norm_loss.data.numpy())

        if t % 100 == 0:
            if torch.cuda.is_available():
                print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
                    t, num_epochs, loss_ratios[-1], train.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(),
                    grad_norm_loss.data.cpu().numpy()))
            else:
                print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
                    t, num_epochs, loss_ratios[-1], train.weights.data.numpy(), task_loss.data.numpy(),
                    grad_norm_loss.data.numpy()))
    task_losses = np.array(task_losses)
    weights = np.array(weights)

    fig = plt.figure()

    plt_list = []
    for i in range(n_tasks):
        plt_list.append(fig.add_subplot(4, 3, i + 1))
        plt_list[i].set_title('Task {} Loss'.format(i + 1))

    plt1 = fig.add_subplot(4, 3, n_tasks + 1)
    plt1.set_title("L_i(t) / L_i(0)")

    plt2 = fig.add_subplot(4, 3, n_tasks + 2)
    plt2.set_title('grad')

    plt3 = fig.add_subplot(4, 3, n_tasks + 3)
    plt3.set_title('Change of weights')
    for i in range(n_tasks):
        plt_list[i].plot(task_losses[:, i])

    plt1.plot(loss_ratios)
    plt2.plot(grad_norm_losses)
    plt3.plot(weights[:, 0])
    plt3.plot(weights[:, 1])
    plt3.plot(weights[:, 2])

    plt.show()
    if mode == 0:
        model_name = "regression_" + model_name
    else:
        model_name = "classification_" + model_name
    torch.save(model, '{}/{}.pt'.format(model_save_path, model_name))


def multi_task_learn(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes, model_save_path,
                     model_name, loss_type):
    """

    :param multi_tasks: 任务名字列表
    :param mode: 0是回归,1是分类
    :param input_size: 1024维输入
    :param hidden_size: 第一隐藏层神经元数量
    :param tower_h1: h1神经元数量
    :param tower_h2: h2神经元数量
    :param num_classes: 分类数量
    :param model_save_path: 模型保存路径
    :param model_name: 模型名字
    :param loss_type: 0是计算自己的loss,1是全部loss平均加权,2是全部loss经验比例加权,3是uncertainty weight比例加权,4是在自己的loss 上使用 uncertainty weight,5是grad norm loss
    :return:
    """
    datas = []
    for index, multi_task in enumerate(multi_tasks):
        data, test = load_train_set_test_set(multi_task)
        datas.append(data)
    num_tasks = len(multi_tasks)
    if loss_type == 0 or loss_type == 1 or loss_type == 2 or loss_type == 3 or loss_type == 4:
        if mode == 0:
            mode_name = "regression"
            mtl_model = MTLRegression(input_size, hidden_size, tower_h1, tower_h2, num_tasks).to(device)
            mtl_model.train()
            loss_list = train_model(mtl_model, multi_tasks, datas, loss_type, mode)
        else:
            mode_name = "classification"
            mtl_model = MTLClassification(input_size, hidden_size, num_classes, tower_h1, tower_h2, num_tasks).to(
                device)
            mtl_model.train()
            loss_list = train_model(mtl_model, multi_tasks, datas, loss_type, mode)

        plt.plot(loss_list)
        plt.title("{}_{}".format(mode_name, model_name))
        plt.show()
        torch.save(mtl_model, '{}/{}_{}.pt'.format(model_save_path, mode_name, model_name))
    else:
        # gradnorm loss
        grad_norm_loss(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes, model_save_path,
                       model_name, datas)


def regression_test(test_datasets_name, model_save_path, model_name, hyper_parameters):
    # 测试回归
    with open("{}/result.txt".format(model_save_path), "a+", encoding="utf-8") as f:
        print(model_name, file=f)
        print(file=f)
        print(model_name)
        for key in hyper_parameters.keys():
            print("{} : {}".format(key, hyper_parameters[key]), file=f)
            print("{} : {}".format(key, hyper_parameters[key]))
        print(file=f)
        # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
        for index, test_dataset_name in enumerate(test_datasets_name):
            file_name = test_dataset_name.split(".")[0]
            print("Start eval {}".format(file_name))
            print("Start eval {}".format(file_name), file=f)
            _, testloader = load_train_set_test_set(test_dataset_name)
            RMSE = get_regression_RMSE(testloader, model_name, index, model_save_path)
            print("{} ".format(RMSE))
            print("{} ".format(RMSE), file=f)
            print(file=f)
        print("---------------------------------", file=f)
        print(file=f)


def classification_test(test_datasets_name, model_save_path, model_name, hyper_parameters):
    with open("{}/result.txt".format(model_save_path), "a+", encoding="utf-8") as f:
        print(model_name, file=f)
        print(file=f)
        print(model_name)
        for key in hyper_parameters.keys():
            print("{} : {}".format(key, hyper_parameters[key]), file=f)
            print("{} : {}".format(key, hyper_parameters[key]))
        print(file=f)
        # 测试分类

        # mtl_classification = MTL_classification(input_size, hidden_size, num_classes).to(device)
        for index, test_dataset_name in enumerate(test_datasets_name):
            print("Start eval {}".format(test_dataset_name))
            print("Start eval {}".format(test_dataset_name), file=f)
            _, testloader = load_train_set_test_set(test_dataset_name)
            accuracy = get_classification_accuracy(testloader, model_name, index, model_save_path)
            auc_roc = get_classification_auc_roc(testloader, model_name, index, model_save_path)
            print("{} %".format(accuracy))
            print("{} %".format(accuracy), file=f)

            print("AUC:{:.4f}".format(auc_roc))
            print("AUC:{:.4f}".format(auc_roc), file=f)
            print(file=f)
        print("---------------------------------", file=f)
        print(file=f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='多任务模式')
    parser.add_argument('--num_epochs', '-n', type=int, default=2000, help="迭代次数")
    parser.add_argument('--mode', '-m', choices=(0, 1), default=1,
                        help="模式选择 0是回归,1是分类")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('--batch_size', '-b', type=int, default=200, help="batch大小")
    parser.add_argument('--hidden_size', '-h1', type=int, default=500, help="第二层神经元数量")
    parser.add_argument('--tower_h1', '-h2', type=int, default=200, help="第三层神经元数量")
    parser.add_argument('--tower_h2', '-h3', type=int, default=50, help="第四层神经元数量")
    parser.add_argument('--alpha', '-a', type=float, default=0.12, help="grad norm的超参数")
    parser.add_argument('--seed', '-s', type=int, default=30, help="需要重新生成tensor数据的随机种子")
    parser.add_argument('--model_name', '-mn', type=str, default="mtl1", help="模型名字(如已有则覆盖)")
    parser.add_argument('--loss_type', '-l', choices=(0, 1, 2, 3, 4, 5), type=int, default=0,
                        help="0是计算自己的loss,1是全部loss平均加权,2是全部loss经验比例加权,3是uncertainty weight比例加权,4是在自己的loss 上使用 uncertainty weight,5是grad norm loss")
    args = parser.parse_args()

    hidden_size = args.hidden_size
    tower_h1 = args.tower_h1
    tower_h2 = args.tower_h2
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model_name = args.model_name
    num_classes = 2
    input_size = 1024

    # 0是回归,1是分类
    mode = args.mode
    if mode == 0:
        num_classes = 1

    # grad norm的超参数
    alpha = args.alpha

    # loss_type 0是计算自己的loss,1是全部loss平均加权,2是全部loss经验比例加权,3是uncertainty weight比例加权,4是在自己的loss 上使用 uncertainty weight,5是grad norm loss
    loss_type = args.loss_type

    # 需要重新生成tensor数据的随机种子,修改后需要在 func:load_train_set_test_set() 中取消对prepare_data()的注释
    seed = args.seed
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    vocab = load_vocal()
    trfm = load_transformer(vocab)


    datasets_name = get_all_dataset()
    datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]



    hyper_parameters = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "tower_h1": tower_h1,
        "tower_h2": tower_h2,
        "alpha": alpha,
        "seed": seed
    }

    model_save_path = "mtl_model/"

    if mode == 0:
        hyper_parameters["mode"] = "regression"
        model_save_path += "regression/"
    else:
        hyper_parameters["mode"] = "classification"
        model_save_path += "classification/"

    if loss_type == 0:
        model_save_path += "self_loss_model"
        loss_type_hint = "计算自己的 loss"
    elif loss_type == 1:
        model_save_path += "average_loss_model"
        loss_type_hint = "全部 loss 平均加权"
    elif loss_type == 2:
        model_save_path += "experience_loss_model"
        loss_type_hint = "全部 loss 经验比例加权"
    elif loss_type == 3:
        model_save_path += "uncertainty_weight_loss_model"
        loss_type_hint = "uncertainty weight 比例加权"
    elif loss_type == 4:
        model_save_path += "self_loss_with_uncertainty_model"
        loss_type_hint = "在自己的 loss 上使用 uncertainty weight"
    else:
        model_save_path += "grad_norm_loss_model"
        loss_type_hint = "grad norm loss"

    hyper_parameters["loss_type"] = loss_type_hint

    if mode == 0:
        print("训练回归任务")
    else:
        print("训练分类任务")
    print(loss_type_hint)

    if mode == 0:
        # 回归
        # multi_tasks = ["freesolv.csv", "lipo.csv","esol.csv"]
        multi_tasks = ["freesolv.csv", "lipo.csv", "esol.csv"]
        # 普通多任务
        multi_task_learn(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes, model_save_path,
                         model_name, loss_type)
        # regression_mode(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes,model_save_path,regression_model_name,loss_type)
        # test_datasets_name = ["freesolv.csv", "lipo.csv","esol.csv"]
        test_datasets_name = ["freesolv.csv", "lipo.csv", "esol.csv"]
        regression_test(test_datasets_name, model_save_path, model_name, hyper_parameters)
    else:
        # 分类
        multi_tasks = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
        # multi_tasks = ["clintox.csv", "bbbp.csv"]
        # 普通多任务
        multi_task_learn(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes, model_save_path,
                         model_name, loss_type)
        # classification_mode(multi_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes,model_save_path,classfication_model_name,loss_type)
        test_datasets_name = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv"]
        # test_datasets_name = ["clintox.csv", "bbbp.csv"]
        classification_test(test_datasets_name, model_save_path, model_name, hyper_parameters)

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
["freesolv.csv", "lipo.csv"]
1-1 自己的loss,不错
2-1 所有loss平均,很差
2-2 所有loss加权平均,比上面好一点,也很差
2-3 所有loss自动平均,非常差
2-4 所有loss加权平均,标准不用0,而是用label的平均值,非常差

3-1 用 gradnorm 均衡loss,alpha=0.12,结果不错,一好一坏
3-2 用 gradnorm 均衡loss,重做上面,一般
3-3 用 gradnorm 均衡loss,将alpha调成1.5,batch5000,不好
3-4 用 gradnorm 均衡loss,将alpha调成1.5,batch2000,不好
3-5 用 gradnorm 均衡loss,将alpha调成0.12,batch2000,不好
3-6 用 gradnorm 均衡loss,将alpha调成0.12,batch1000,epoch2000,learning_rate 1e-4,不好
3-7 用 gradnorm 均衡loss,将alpha调成0.12,batch2000,epoch2000,learning_rate 1e-4,不好
3-8 用 gradnorm 均衡loss,将alpha调成0.12,batch5000,epoch2000,learning_rate 1e-4,还行
3-9 用 gradnorm 均衡loss,将alpha调成0.12,batch4000,epoch2000,learning_rate 1e-4,还行

3-10 三个任务一起,用3-9超参数,还行
3-11 三个任务一起,用3-9超参数 learning_rate 0.00005,epoch3000,很差
3-12 三个任务一起,用3-9超参数 learning_rate 0.0001,epoch1000,最好
3-13 三个任务一起,用3-9超参数 learning_rate 0.0001,epoch800,最好
3-14 三个任务一起,用3-9超参数 learning_rate 0.0001,epoch500,最好

'''
