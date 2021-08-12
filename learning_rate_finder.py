import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn
import math
from tqdm import tqdm, trange


class LearningRateFinder:

    def __init__(self, model, criterion, optimizer, device):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()

    def fit(self, data_loader, steps=100, min_lr=1e-7, max_lr=1, constant_increment=False):

        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0

        epochs = math.ceil(steps / len(data_loader))

        progressbar = trange(epochs, desc='Progress')

        for epoch in progressbar:
            batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
                              leave=False)

            for i, (x, y) in batch_iter:

                x, y = x.to(self.device), y.to(self.device)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y.long())
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()

                steps_counter += 1

                if steps_counter > steps:
                    break

                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def plot(self, smoothing=True, clipping=True, smoothing_factor=0.1):

        loss_data = pd.Series(list(self.loss_history.values()))
        lr_list = list(self.loss_history.keys())

        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(pd.Series(
                [1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]))  # bias correction

        if clipping:
            loss_data = loss_data[10:-5]
            lr_list = lr_list[10:-5]

        plt.plot(lr_list, loss_data)
        plt.xscale('log')
        plt.title('Loss vs Learning rate')
        plt.xlabel('Learning rate (log scale)')
        plt.ylabel('Loss (exponential moving average)')
        plt.show()

    def reset(self):
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print('Model and optimizer in initial state.')

