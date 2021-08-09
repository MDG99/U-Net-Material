import torch
import numpy as np
import pathlib
from model import UNet
from trainer import Trainer
from data import get_dataloaders

# Dispositivo
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Dataloaders
dataloader_training, dataloader_validation = get_dataloaders()

# Modelo
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same').to(device)
# Criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Trainer
tr = Trainer(model=model,
             device=device,
             criterion=criterion,
             optimizer=optimizer,
             tr_dataloader=dataloader_training,
             val_dataloader=dataloader_validation,
             lr_scheduler=None,
             epochs=2,
             epoch=0)

# Start training
training_losses, validation_losses, lr_rates = tr.run_trainer()

model_name = 'material_model.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

