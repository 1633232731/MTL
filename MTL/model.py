import torch

from torch.nn.modules.loss import MSELoss

import torch.nn.functional as F









class RegressionGradNormLossTrain(torch.nn.Module):

    def __init__(self, model):
        # initialize the module using super() constructor
        super(RegressionGradNormLossTrain, self).__init__()
        # assign the architectures
        self.model = model
        # assign the weights for each task
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        # loss function
        self.mse_loss = MSELoss()

    
    def forward(self, x, ts,index):
        n_tasks = 2
        ys = self.model(x)
        task_loss = []
        for i in range(n_tasks):
            if index == i:
                # 回归
                # task_loss.append( self.mse_loss(ys[index].squeeze(dim=1), ts) )
                # 分类
                task_loss.append( self.mse_loss(ys[index][:,1], ts) )
            else:
                # 回归
                # task_loss.append( self.mse_loss(torch.zeros_like(ys[index].squeeze(dim=1)), ts) )
                # 分类
                task_loss.append( self.mse_loss(torch.zeros_like(ys[index][:,1]), ts) )
        task_loss = torch.stack(task_loss)
        return task_loss


    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()



class RegressionGradNormLossModel(torch.nn.Module):

    def __init__(self, n_tasks):
        # initialize the module using super() constructor
        super(RegressionGradNormLossModel, self).__init__()
        
        # number of tasks to solve
        self.n_tasks = n_tasks
        # fully connected layers
        self.l1 = torch.nn.Linear(1024, 500)
        self.l2 = torch.nn.Linear(500, 200)
        self.l3 = torch.nn.Linear(200, 50)
        # branches for each task
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), torch.nn.Linear(50, 2))

    
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

        