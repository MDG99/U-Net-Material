import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torchvision import *
import transforms as tr
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def get_dataloaders():
    names = os.listdir('Data/input')

    inputs = []
    targets = []

    for n in names:
        inputs.append(os.path.join('Data/input/', n))
        targets.append(os.path.join('Data/targets/', n))

    train_size = 0.80
    random_seed = 40

    # Dividimos los datos de entrenamiento y validación
    train_inputs, val_inputs = train_test_split(inputs,
                                                random_state=random_seed,
                                                train_size=train_size,
                                                shuffle=True)

    train_targets, val_targets = train_test_split(targets,
                                                  random_state=random_seed,
                                                  train_size=train_size,
                                                  shuffle=True)

    # Transformaciones
    transforms = tr.SegmentationCompose([
        tr.SegmentationRandomCrop((500, 500)),
        tr.SegmentationResize((512, 512)),
        tr.SegmentationRandomRotation((-90, 90)),
        tr.SegmentationHorizontalFlip(0.5),
        tr.SegmentationVerticalFlip(0.5)
    ])

    # Dataset y Dataloader
    training_dataset = CustomDataSet(inputs=train_inputs, targets=train_targets, transform=transforms)
    validation_dataset = CustomDataSet(inputs=val_inputs, targets=val_targets, transform=transforms)

    #print(np.shape(training_dataset))
    #print(np.shape(validation_dataset))

    training_dataloader = data.DataLoader(dataset=training_dataset, batch_size=4, shuffle=True)
    validation_dataloader = data.DataLoader(dataset=validation_dataset, batch_size=4, shuffle=True)

    return training_dataloader, validation_dataloader


def visualize():
    # Cargando imágenes
    training_dataloader, _ = get_dataloaders()
    x, y = next(iter(training_dataloader))

    # Mostrando imágenes
    r = random.randint(0, len(x) - 1)
    print(r)

    ####################################################################################################################
    # Procesamiento Matplotlib
    sampleimage = x[r].squeeze().permute(1, 2, 0)

    plt.subplot(1, 2, 1)
    plt.imshow(sampleimage)
    plt.title('Input')
    plt.axis('off')
    #
    plt.subplot(1, 2, 2)
    plt.imshow(y[r], cmap='gray', vmin=0, vmax=1)
    plt.title('Target')
    plt.axis('off')
    plt.show()

    ####################################################################################################################
    # Procesamiento Napari
    # import napari
    # sampleimage = x[r] / 255.0
    # sampleimage = sampleimage.numpy()
    # samplemask = y[r] / 255.0
    # samplemask = samplemask.numpy()

    # viewer = napari.Viewer()
    # viewer.add_image(sampleimage, scale=(1, 1), name='Imagen')
    # viewer.add_image(samplemask, scale=(1, 1), opacity=0.25, name='Material')
    # napari.run()


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
        t = ImageOps.grayscale(t)

        # Aplicando las transformaciones
        if self.transform is not None:
            i, t = self.transform(i, t)

        t_aux = t.load()

        a, b = np.shape(t)
        for r in range(a):
            for c in range(b):
                if t_aux[r, c] == 61:
                    t_aux[r, c] = 0
                else:
                    t_aux[r, c] = 1

        i, t = np.float32(i), np.float32(t)

        i = i/255.0

        #i = np.moveaxis(i, -1, 0)
        i = i.reshape((i.shape[2], i.shape[1], i.shape[0]))
        t = t.reshape((t.shape[1], t.shape[0]))

        return i, t

