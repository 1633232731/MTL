import torch

from torch.nn.modules.loss import MSELoss,CrossEntropyLoss

import torch.nn.functional as F

from torch import nn


class GradNormLossTrain(torch.nn.Module):

    def __init__(self, model, mode, num_tasks):
        """

        :param model:
        :param mode: 0是回顾,1是分类
        """

        super(GradNormLossTrain, self).__init__()
        self.model = model
        self.weights = torch.nn.Parameter(torch.ones(model.num_tasks).float())
        # 损失函数
        if mode == 0:
            self.loss_function = MSELoss()
        else:
            self.loss_function = CrossEntropyLoss()

        self.mode = mode
        self.n_tasks = num_tasks
    
    def forward(self, x, target, index):
        n_tasks = self.n_tasks
        mode = self.mode
        ys = self.model(x)
        task_loss = []
        for i in range(n_tasks):
            if index == i:
                # 是该任务的输出
                if mode == 0:
                    # 回归
                    task_loss.append(self.loss_function(ys[index].squeeze(dim=1), target))
                else:
                    # 分类
                    task_loss.append(self.loss_function(ys[index], target))
            else:
                # 不是该任务的输出
                if mode == 0:
                    # 回归
                    # task_loss.append(self.loss_function(ys[index].squeeze(dim=1), torch.zeros_like(ys[index].squeeze(dim=1))))
                    task_loss.append(self.loss_function(torch.zeros_like(ys[index].squeeze(dim=1)), target))
                else:
                    # 分类
                    task_loss.append(self.loss_function(ys[index], torch.zeros_like(ys[index][:,1]).to(torch.int64)))
        task_loss = torch.stack(task_loss)
        return task_loss


    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()



class GradNormLossModel(torch.nn.Module):

    def __init__(self, num_tasks, mode, input_size, hidden_size, tower_h1, tower_h2, num_classes):
        """

        :param num_tasks: 任务数
        :param mode: 0是回顾,1是分类
        :param input_size:
        :param hidden_size:
        :param tower_h1:
        :param tower_h2:
        :param num_classes: 分类的数量
        """
        super(GradNormLossModel, self).__init__()

        self.num_tasks = num_tasks
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, tower_h1)
        self.l3 = torch.nn.Linear(tower_h1, tower_h2)
        if mode == 0:
            for i in range(self.num_tasks):
                setattr(self, 'task_{}'.format(i), torch.nn.Linear(tower_h2, 1))
        else:
            for i in range(self.num_tasks):
                setattr(self, 'task_{}'.format(i), torch.nn.Linear(tower_h2, num_classes))

    
    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))

        outs = []
        for i in range(self.num_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(h))

        return outs


    def get_last_shared_layer(self):
        return self.l3




class MTLRegression(nn.Module):
    # 多任务回归 2 + 2 架构
    def __init__(self, input_size, hidden_size, tower_h1, tower_h2,num_tasks):
        super(MTLRegression, self).__init__()
        self.share_layer = nn.Sequential(
            # 第一个隐含层
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, tower_h1),
            nn.ReLU(),
            # nn.Linear(tower_h1, tower_h2),
            # nn.ReLU(),
        )
        self.num_tasks = num_tasks
        for i in range(self.num_tasks):
            setattr(self, 'task_separate_layer_{}'.format(i), torch.nn.Linear(tower_h1, tower_h2))

        for i in range(self.num_tasks):
            setattr(self, 'predict_{}'.format(i), torch.nn.Linear(tower_h2, 1))

    # 定义网络前向传播路径
    def forward(self, x):
        out = []
        t = []
        share_layer_output = (self.share_layer(x))

        for i in range(self.num_tasks):
            task_separate_layer = getattr(self, 'task_separate_layer_{}'.format(i))
            t.append(F.relu(task_separate_layer(share_layer_output)))

        for i in range(self.num_tasks):
            predict = getattr(self, 'predict_{}'.format(i))
            out.append(predict(t[i])[:, 0])

        # 输出一个一维向量
        return out

class MTLRegression_3p1(nn.Module):
    # 多任务回归 3 + 1 架构
    def __init__(self, input_size, hidden_size, tower_h1, tower_h2,num_tasks):
        super(MTLRegression_3p1, self).__init__()
        self.share_layer = nn.Sequential(
            # 第一个隐含层
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, tower_h1),
            nn.ReLU(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
        )
        self.num_tasks = num_tasks
        # for i in range(self.num_tasks):
        #     setattr(self, 'task_separate_layer_{}'.format(i), torch.nn.Linear(tower_h1, tower_h2))

        for i in range(self.num_tasks):
            setattr(self, 'predict_{}'.format(i), torch.nn.Linear(tower_h2, 1))

    # 定义网络前向传播路径
    def forward(self, x):
        out = []
        share_layer_output = (self.share_layer(x))

        # for i in range(self.num_tasks):
        #     task_separate_layer = getattr(self, 'task_separate_layer_{}'.format(i))
        #     t.append(F.relu(task_separate_layer(share_layer_output)))

        for i in range(self.num_tasks):
            predict = getattr(self, 'predict_{}'.format(i))
            out.append(predict(share_layer_output)[:, 0])

        # 输出一个一维向量
        return out

class MTLClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, tower_h1, tower_h2,num_tasks) -> None:
        super().__init__()

        self.share_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, tower_h1),
            nn.ReLU(),
            # nn.Linear(tower_h1, tower_h2),
            # nn.ReLU(),
        )
        self.num_tasks = num_tasks
        for i in range(self.num_tasks):
            setattr(self, 'task_separate_layer_{}'.format(i), torch.nn.Linear(tower_h1, tower_h2))

        for i in range(self.num_tasks):
            setattr(self, 'predict_{}'.format(i), torch.nn.Linear(tower_h2, num_classes))

    def forward(self, x):
        out = []
        t = []
        share_layer_output = (self.share_layer(x))

        for i in range(self.num_tasks):
            task_separate_layer = getattr(self, 'task_separate_layer_{}'.format(i))
            t.append(F.relu(task_separate_layer(share_layer_output)))

        for i in range(self.num_tasks):
            predict = getattr(self, 'predict_{}'.format(i))
            out.append(predict(t[i]))

        return out

class MTLClassification_3p1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, tower_h1, tower_h2,num_tasks) -> None:
        super().__init__()

        self.share_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, tower_h1),
            nn.ReLU(),

            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
        )
        self.num_tasks = num_tasks
        # for i in range(self.num_tasks):
        #     setattr(self, 'task_separate_layer_{}'.format(i), torch.nn.Linear(tower_h1, tower_h2))

        for i in range(self.num_tasks):
            setattr(self, 'predict_{}'.format(i), torch.nn.Linear(tower_h2, num_classes))

    def forward(self, x):
        out = []
        share_layer_output = (self.share_layer(x))

        # for i in range(self.num_tasks):
        #     task_separate_layer = getattr(self, 'task_separate_layer_{}'.format(i))
        #     t.append(F.relu(task_separate_layer(share_layer_output)))

        for i in range(self.num_tasks):
            predict = getattr(self, 'predict_{}'.format(i))
            out.append(predict(share_layer_output))

        return out
        
