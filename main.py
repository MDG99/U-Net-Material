import torch
import numpy as np
from model import UNet
from trainer import Trainer

#Dispositivo
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Modelo
model = UNet(in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same')
#Criterion
criterion = torch.nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion = criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=2,
                  epoch=0)

#Start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()