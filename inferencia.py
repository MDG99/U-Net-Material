import pathlib

import skimage.io
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from PIL import ImageOps, Image
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_otsu


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


def lectura_datos(path):
    names = os.listdir(path + 'input/')

    input_names = []
    target_names = []

    for n in names:
        input_names.append(os.path.join(path + 'input/', n))
        target_names.append(os.path.join(path + 'targets/', n))

    # Leemos las imágenes
    images = [imread(img_name) for img_name in input_names]
    #images = [resize(img, (256, 256, 3)) for img in images]
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

        #t2 = t2.resize((256, 256), Image.NEAREST)

        #t3=np.array(t2, dtype=float)
        #skimage.io.imsave("target.JPEG", t3)
        targets.append(t2)

    return images, targets, input_names

#######################################################################################################################
# Lectura de imágenes
#######################################################################################################################

test_path = 'Data/test/'
images, targets, names = lectura_datos(test_path)

#######################################################################################################################
# SVM
#######################################################################################################################
test_path = 'Data/'
images_SVM, targets_SVM, _ = lectura_datos(test_path)
grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)
model = GridSearchCV(svc, grid)
#model.fit(images_SVM, targets_SVM)


#######################################################################################################################
# U-NET
#######################################################################################################################
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
model_name = 'material_model040920211954.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name, map_location=torch.device('cpu'))
model.load_state_dict(model_weights)

outputs = [prediccion(image, model, device) for image in images]

#######################################################################################################################
# Seleccionamos una imagen cualquiera para visualizar
#######################################################################################################################
idx = 5

sampleimage = images[idx]
samplemask = targets[idx]
sampleprediction = outputs[idx]

#######################################################################################################################
# Otsu
#######################################################################################################################
th = threshold_otsu(rgb2gray(images[idx]))
otsu = rgb2gray(sampleimage) > th

for i in range(np.shape(otsu)[0]):
    for j in range(np.shape(otsu)[1]):
        if otsu[i, j] == 0:
            otsu[i, j] = 1
        else:
            otsu[i, j] = 0

# IoU imagen entrenada
intersection = np.logical_and(samplemask, sampleprediction)
union = np.logical_or(samplemask, sampleprediction)
iou_score = np.sum(intersection) / np.sum(union) * 100

# IoU Otsu
intersection_otsu = np.logical_and(samplemask, otsu)
union_otsu = np.logical_or(samplemask, otsu)
iou_otsu = np.sum(intersection_otsu) / np.sum(union_otsu) * 100

print(f"Intersection over Union Model: {iou_score:4f}%")
print(f"Intersection over Union Otsu: {iou_otsu:4f}%")

plt.subplot(1, 4, 1)
plt.imshow(sampleimage)
plt.title(f'{names[idx]}')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(samplemask, cmap='gray', vmin=0, vmax=1)
plt.title('Target')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(sampleprediction, cmap='gray', vmin=0, vmax=1)
plt.title(f"IoU modelo: {iou_score:4f}%")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(otsu, cmap=plt.cm.gray)
plt.title(f"Otsu IoU: {iou_otsu}")
plt.axis('off')

plt.show()
