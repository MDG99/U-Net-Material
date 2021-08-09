from tqdm import tqdm, trange
import numpy as np
import torch


class Trainer:

    def __init__(self, model, criterion, optimizer, tr_dataloader, val_dataloader=None,
                 lr_scheduler=None, epochs=100, epoch=0, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = tr_dataloader
        self.validation_DataLoader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):
        progressbar = trange(self.epochs, desc='Progess')

        for i in progressbar:
            """Contador de épocas"""
            self.epoch += 1

            """Bloque de entrenamiento"""
            self._train()

            """Bloque de validación"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Bloque Learning rate scheduler"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    self.lr_scheduler.batch()

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(input)
            loss = self.criterion(out, target)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(f'Training: (loss: {loss_value:.4f})')

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', totla=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f}')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()
