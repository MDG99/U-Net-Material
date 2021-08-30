import torch
import pathlib
from model import UNet
from trainer import Trainer
from data import get_dataloaders
from datetime import datetime


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
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

tr = Trainer(model=model,
             device=device,
             criterion=criterion,
             optimizer=optimizer,
             tr_dataloader=dataloader_training,
             val_dataloader=dataloader_validation,
             lr_scheduler=None,
             epochs=50,
             epoch=0)

# Start training
training_losses, validation_losses, lr_rates = tr.run_trainer()

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M")
model_name = f'material_model{dt_string}.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)



