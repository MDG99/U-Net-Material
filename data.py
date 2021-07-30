import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torchvision import *
import transforms as tr
from PIL import Image
import napari


class CustomDataSet(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.target = targets
        self.transform = transform

        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        my_input = self.inputs[index]
        my_target = self.target[index]
        i = Image.open(my_input)
        t = Image.open(my_target)

        # Aplicando las transformaciones
        if self.transform is not None:
            i, t = self.transform(i, t)

        i, t = np.float32(i), np.float32(t)

        return i, t


names = os.listdir('Data/input')

inputs = []
targets = []

for n in names:
    inputs.append(os.path.join('Data/input/', n))
    targets.append(os.path.join('Data/target', n))

#Verificamos que cargaron bien las imágenes
#print(targets)

#Transformaciones
transforms = tr.SegmentationCompose([
    #tr.SegmentationRandomRotation((-90, 90))
    tr.SegmentationResize((500, 500)),
    tr.SegmentationHorizontalFlip(0.5),
    tr.SegmentationVerticalFlip(0.5)
])

#Dataset y Dataloader
training_dataset = CustomDataSet(inputs=inputs, targets=targets, transform=transforms)
training_dataloader = data.DataLoader(dataset=training_dataset, batch_size=5, shuffle=True)

# Cargando imágenes
x, y = next(iter(training_dataloader))

#Aseguramos que los tensores tengan la forma esperada
#print(f'x = shape{x.shape}; type: {x.dtype}')
#print(f'x = min: {x.min()}; max: {x.max()}')
#print(f'y = shape {y.shape}; class: {y.unique()}; type: {y.dtype}')

#######################################################################################################################
# Mostrando imágenes
r = random.randint(0, len(x) - 1)
print(r)

#######################################################################################################################
# Procesamiento Matplotlib
# sampleimage = x[r].squeeze().permute(1,2,0)
# samplemask = y[r].squeeze().permute(1,2,0)

# plt.subplot(1, 2, 1)
# plt.imshow(x[r].squeeze().permute(1,2,0))
# plt.title('Input')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(y[r].squeeze().permute(1,2,0))
# plt.title('Target')
# plt.axis('off')
# plt.show()
#

#######################################################################################################################
# Procesamiento Napari
sampleimage = x[r] / 255.0
sampleimage = sampleimage.numpy()
samplemask = y[r] / 255.0
samplemask = samplemask.numpy()

viewer = napari.Viewer()
viewer.add_image(sampleimage, scale=(1, 1), name='Imagen')
viewer.add_image(samplemask, scale=(1, 1), opacity=0.25, name='Material')
#napari.run()
