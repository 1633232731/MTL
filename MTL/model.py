import torch

from torch.nn.modules.loss import MSELoss,CrossEntropyLoss

import torch.nn.functional as F




class GradNormLossTrain(torch.nn.Module):

    def __init__(self, model,mode,n_tasks):
        """

        :param model:
        :param mode: 0是回顾,1是分类
        """

        # initialize the module using super() constructor
        super(GradNormLossTrain, self).__init__()
        # assign the architectures
        self.model = model
        # assign the weights for each task
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        # loss function
        if mode == 0:
            self.loss_function = MSELoss()
        else:
            self.loss_function = CrossEntropyLoss()

        self.mode = mode
        self.n_tasks = n_tasks
    
    def forward(self, x, target, index):
        n_tasks = self.n_tasks
        mode = self.mode
        ys = self.model(x)
        task_loss = []
        for i in range(n_tasks):
            if index == i:
                if mode == 0:
                    # 回归
                    task_loss.append(self.mse_loss(ys[index].squeeze(dim=1), ts))
                else:
                    # 分类
                    task_loss.append(self.loss_function(ys[index][:,1], target))
            else:
                if mode == 0:
                    # 回归
                    task_loss.append(self.mse_loss(torch.zeros_like(ys[index].squeeze(dim=1)), ts))
                else:
                    # 分类
                    task_loss.append(self.loss_function(torch.zeros_like(ys[index][:,1]), target))
        task_loss = torch.stack(task_loss)
        return task_loss


    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()



class GradNormLossModel(torch.nn.Module):

    def __init__(self, n_tasks,mode,input_size,hidden_size,tower_h1,tower_h2,num_classes):
        """

        :param n_tasks: 任务数
        :param mode: 0是回顾,1是分类
        :param input_size:
        :param hidden_size:
        :param tower_h1:
        :param tower_h2:
        :param num_classes: 分类的数量
        """
        # initialize the module using super() constructor
        super(GradNormLossModel, self).__init__()
        
        # number of tasks to solve
        self.n_tasks = n_tasks
        # fully connected layers
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, tower_h1)
        self.l3 = torch.nn.Linear(tower_h1, tower_h2)
        # branches for each task
        if mode == 0:
            for i in range(self.n_tasks):
                setattr(self, 'task_{}'.format(i), torch.nn.Linear(tower_h2, 1))
        else:
            for i in range(self.n_tasks):
                setattr(self, 'task_{}'.format(i), torch.nn.Linear(50, num_classes))

    
    def forward(self, x):
        # forward pass through the common fully connected layers
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        # h = F.relu(self.l4(h))

        # forward pass through each output layer
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(h))

        # return torch.stack(outs, dim=1)
        return outs


    def get_last_shared_layer(self):
        return self.l3

        