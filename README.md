# U-Net-Material
 Este repositorio fue programado tomando como referencia el tutorial, a diferencia del programa original, aquí se 
 pretende trabajar con un dataset de imágenes de 2 dimensiones correspondientes a material antes y después de pruebas de adherencia. 
 
 ## Dataset
 El dataset se encuentra en la carpeta [Data](Data), ahí mismo se encuentran tres subdirectorios: input y target para
 de donde se sacar las imágenes para el entrenameinto y validación; y test, el cuál contiene a su vez dos carpetas más 
 con imágenes y sus correspondientes targets que servirán para la etapa de inferencia (imágenes nunca antes vistas).

El archivo [data.py](data.py) cuenta con 2 funciones: *get_dataloaders* y *visualize*; y una clase: *Customdataset*. La 
función *get_dataloaders* se encarga de leer los datos en la carpeta de *Data*, dividirla a razón 80:20 (trainnig y 
validation), aplicar las transformaciones, obtener los datasets con ayuda de la clase y finalmente obtener el dataloader.
Por otro lado la función visualize sirve únicamente para ver algunas imágenes del dataloader de entrenamiento, se mantiene
comentado el procesameinto para usar la librería napari y se mantiene activo el uso de matplotlib.
 
 Las transformaciones aplicadas para construir el dataloader de entrenamiento se encuentran en transforms.py, para ello 
 utilicé parte del código mostrado en la [documentación oficial de torch](https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#CenterCrop) y lo adapté de tal manera que fuera posible aplicar 
 las mismas transformaciones de caracter aleatorio a 2 imágenes. Las transformaciones disponibles son:
 - Random Crop
 - Resize
 - To Tensor
 - Compose
 - Random Vertical Flip
 - Random Horizontal Flip
 - Random Rotation.

## Modelo
El modelo se encuentra construido en [modelo.py](modelo.py) y se centra en la construcción del U-Net en forma de bloques
de subida y bajada.

**Bloque de bajada:**

Convolución 1 + Activación 1 + Normalización 1 (En caso de aplicar)
Convolución 2 + Activación 2 + Normalización 2 (En caso de aplicar)  
Pooling (*Maxpooling2D*)

**Bloque de subida:**

Sobremuestreo (*Transposed2D*) y tratamiento (crop, conv, act, norm y merge)  
Convolución 1 + Activación 1 + Normalización 1 (En caso de aplicar)  
Convolución 2 + Activación 2 + Normalización 2 (En caso de aplicar) 

Finalmente se realiza una convolución final de (1x1), siguiendo de esa manera la arquitectura de la U-Net.

Entre otras cosas es posible modificar las funciones de activación (ReLU, Leaky y Elu), la normalización (batch, instace
 y group) y la manera de realizar el sobremuestreo (Transposed2D y UpSample). 

#### Revisión del modelo
El modelo se puede verificar con el archivo [model_check.py](model_check.py), mediante la librería [torchsummary](https://github.com/sksq96/pytorch-summary),
la cual hace un resumen de lo que sucede en el modelo, así mismo hay dos funciones complementarias que permiten conocer 
la manera en la que van cambiando las dimesiones de las imágenes a lo largo del modelo y otra función para conocer la 
profundidad dado un tamaño de imagen de entrada.

## Entrenamiento
El entrenamiento se empieza desde el [main.py](main.py), los pasos que sigue este archivo son: define el dispositivo, 
obtiene los dataloaders, define el modelo, define el criterio (CrossEntropy) y el optimizador (SGD), realiza el 
entrenamiento y guarda el modelo.
 
Por otro lado, el archivo [trainer.py](trainer.py) contiene toda la lógica para realizar el entrenamiento, lo que retorna
es un promedio del error de entrenamiento y validación después de cada época.
 
## Inferencia
El documento [inferencia.py](inferencia.py) se encarga de probar el modelo ya entrenado en imágenes nunca antes vistas 
(testing dataset), por lo que el programa empieza leyendo este dataset y dándole un preprocesameinto a las imágenes 
(hace un redimensionamiento y el target lo pasa a blanco y negro), inicializa el modelo a usar, carga el archivo de los 
 pesos *.pt* y a cada imagen del testing dataset lo mete a la función *predicción*, la cual realiza un preprocesameinto, 
 evaluación del modelo y un postprocesamiento, otorgando a la salida el target predicho por el modelo.
 Finalmente, se procede a calcular el *intersection over union* entre el target original y el predicho, así como mostrar 
 los resultados usando matplotlib.  
  
 
## Modelos guardados
Los archivos *.pt* resultantes del entrenamiento son generados en el fichero *main.py*  y pueden ser consultados [aquí](https://www.dropbox.com/sh/biej6gyhhk1id7t/AABVJjWshBgU4jGDcP6zc4oQa?dl=0)
 
**Nota:** el nombre del archivo *.pt* es generado tomando la hora y fecha del sistema.

  ## Tareas
  -[X] Modificar el archivo Readme y limpiar el código.
  -[ ] Comparara contra un algortimo de binarización y binarización de Otsu.
  -[ ] Revisar el preprocesamiento y posprocesamiento.
  -[ ] Comparar con otras variantes de U-Net.
  -[ ] Hacer que el *resize* sea dinámico.
  
   ## Referencias
   
   [U-Net Tutorial Parte 1](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55)
   
   [U-Net Tutorial Parte 2](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862)
   
   [U-Net Tutorial Parte 3](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234)
   
   [U-Net Tutorial Parte 4](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-inference-4-4-e52b074ddf6f)
    
  
  