from typing import List

import numpy
import pandas as pd
from sklearn.utils import shuffle as reset
import torch
from utils import split
from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
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


pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
vocab = load_vocal()
trfm = load_transformer(vocab)

t = ["bace.csv", "bbbp.csv", "clintox.csv", "HIV.csv", "muv.csv", "tox21.csv", "sider.csv", "esol.csv",
     "freesolv.csv", "lipo.csv"]
datasets_name = get_all_dataset()
# datasets_name = ["bace.csv"]

import numpy
for dataset_name in datasets_name:
    s = []
    X, Y, X_test, Y_test = prepare_data(dataset_name)
    file_name = dataset_name.split(".")[0]
    s.append(X)
    s.append(Y)
    s.append(X_test)
    s.append(Y_test)

    # s_numpy = [x.numpy() for x in s] #步骤1
    numpy.save("tensor_data/{}_tensor.npy".format(file_name),s)

# for dataset_name in datasets_name:
#     X, Y, X_test, Y_test = prepare_data(dataset_name)
#     file_name = dataset_name.split(".")[0]
#     print(X)
#     with open("tensor_data/{}/{}.tensor".format(file_name,"X"),"wb") as f:
#         torch.save(X,f)

# b = numpy.load("tensor_data/tensor.npy", allow_pickle=True)
# c = [torch.tensor(x) for x in b]
#
# for i in c:
#     print(len((i)))

# for dataset_name in datasets_name:
#     file_name = dataset_name.split(".")[0]
#
#     p = torch.load("tensor_data/{}.tensor".format(file_name))
#     print(p)
'''
[[-0.04837199  0.05349776 -0.27410927 ...  0.4407835   1.0344229
   0.5548961 ]
 [ 0.23367056 -0.07295024  0.04524554 ...  0.73599344  0.83701044
  -0.17228502]
 [-0.30589828  0.01457004 -0.37422922 ...  0.640108    0.6886574
   0.03235246]
 ...
 [-0.0219465  -0.02037244 -0.30890003 ...  0.25516704  1.2887658
   0.25078145]
 [-0.27812383  0.2022775  -0.41321993 ...  0.24608806  1.1947991
  -0.12421796]
 [ 0.04365214  0.0765794  -0.16466732 ...  0.5999204   1.0781609
   0.89489204]]
'''