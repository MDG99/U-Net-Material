from model import UNet
from torchsummary import summary


def compute_max_depth(shape, max_depth=10, print_out=True):
    shapes = []
    shapes.append(shape)
    for level in range(1, max_depth):
        if shape % 2 ** level == 0 and shape / 2 ** level > 1:
            shapes.append(shape / 2 ** level)
            if print_out:
                print(f'Level {level}: {shape / 2 ** level}')
        else:
            if print_out:
                print(f'Max-level: {level - 1}')
            break

    return shapes


def compute_possible_shapes(low, high, depth):
    possible_shapes = {}
    for shape in range(low, high + 1):
        shapes = compute_max_depth(shape,
                                   max_depth=depth,
                                   print_out=False)
        if len(shapes) == depth:
            possible_shapes[shape] = shapes

    print(f"Possible shapes: {possible_shapes}")
    return possible_shapes


shape = 1920
low = 128
high = 512
depth = 10

# Describe las dimensiones de las imágenes en un modelo
out = compute_max_depth(shape, print_out=True, max_depth=depth)

#Enlista una serie de tamaños posibles dado un nivel de profundidad
possible_shapes = compute_possible_shapes(low, high, depth)

model = UNet(in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same')

summary = summary(model, (1, 512, 512))

