import torch
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from model import UNet
from trainer import Trainer
from data import get_dataloaders
from learning_rate_finder import LearningRateFinder
from visualize import plot_training
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

option = input("Entrenar[E] o Encontrar el lr[L]")

if option == 'E' or option == 'e':
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

    fig = plot_training(training_losses=training_losses,
                  validation_losses=validation_losses,
                  learning_rate=lr_rates,
                  gaussian=False,
                  sigma=1,
                  figsize=(10, 4))

    plt.show()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M")
    model_name = f'material_model{dt_string}.pt'
    torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

elif option == 'L' or option == 'l':
    # learning rate finder
    lrf = LearningRateFinder(criterion=criterion, device=device, model=model, optimizer=optimizer)
    lrf.fit(dataloader_training, steps=1000)
    lrf.plot()




