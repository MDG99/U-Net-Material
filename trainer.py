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

        for e in range(self.epochs):
            """Contador de épocas"""
            self.epoch += 1

            """Bloque de entrenamiento"""
            self._train()

            """Bloque de validación"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Bloque Learning rate scheduler"""
            if self.lr_scheduler is not None: #TODO: REVISAR EL FUNCIONAMIENTO DEL LEARNING SCHEDULER
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[e]) #todo: revisar e
                else:
                    self.lr_scheduler.batch()

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        self.model.train()
        train_losses = []

        for x, y in self.training_DataLoader:
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(input)
            loss = self.criterion(out, target.long())
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()
            print(f'Training: (loss: {loss_value:.4f})')

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

    def _validate(self):
        self.model.eval()
        valid_losses = []

        for x, y in self.validation_DataLoader:
            input, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target.long())
                loss_value = loss.item()
                valid_losses.append(loss_value)

                print(f'Validation: (loss {loss_value:.4f}')

        self.validation_loss.append(np.mean(valid_losses))

