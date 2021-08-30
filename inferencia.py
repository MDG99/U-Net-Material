import pathlib
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from PIL import ImageOps, Image
from skimage.io import imread
from skimage.transform import resize


def prediccion(img, model, device):
    model.eval()
    # Preprocesando la imagen y mandando al dispositivo
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)

    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    # Salida: softmax más postprocesamiento
    out_softmax = torch.softmax(out, dim=1)
    result = torch.argmax(out_softmax, dim=1)  # perform argmax to generate 1 channel
    result = result.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]
    return result


# Seleccionando los datos para el testing
names = os.listdir('Data/test/input')

input_names = []
target_names = []

for n in names:
    input_names.append(os.path.join('Data/test/input/', n))
    target_names.append(os.path.join('Data/test/targets/', n))

# Leemos las imágenes
images = [imread(img_name) for img_name in input_names]
targets = []

for aux in range(len(target_names)):
    t1 = Image.open(target_names[aux])
    t2 = ImageOps.grayscale(t1)
    t_aux = t2.load()

    a, b = np.shape(t2)

    for r in range(a):
        for c in range(b):
            if t_aux[r, c] == 61:
                t_aux[r, c] = 0
            else:
                t_aux[r, c] = 1

    t2 = t2.resize((256, 256), Image.NEAREST)
    targets.append(t2)

# Aplicamos el redimensionamiento de las imágenes
images = [resize(img, (256, 256, 3)) for img in images]

# Seleccionando el dispositivo
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Inicializando el modelo
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same').to(device)

# Cargando el modelo
model_name = 'material_model110820212055.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name)
model.load_state_dict(model_weights)

outputs = [prediccion(image, model, device) for image in images]
idx = 0

sampleimage = images[idx]
samplemask = targets[idx]
sampleprediction = outputs[idx]

intersection = np.logical_and(samplemask, sampleprediction)
union = np.logical_or(samplemask, sampleprediction)
iou_score = np.sum(intersection) / np.sum(union) * 100

print(f"Intersection over Union Score: {iou_score:4f}%")

plt.subplot(1, 3, 1)
plt.imshow(sampleimage)
plt.title('Input')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(samplemask, cmap='gray', vmin=0, vmax=1)
plt.title('Target')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sampleprediction, cmap='gray', vmin=0, vmax=1)
plt.title(f"Intersection over Union Score: {iou_score:4f}%")
plt.axis('off')

plt.show()
