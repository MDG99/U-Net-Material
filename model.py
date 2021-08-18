from torch import nn
import torch


@torch.jit.script
def autocrop(encoder_layer, decoder_layer):
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[:, :, ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                                                ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)]
    return encoder_layer, decoder_layer


def get_activation_function(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def get_normalization(normalization, num_channels):
    if normalization == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def get_up_layer(in_channels, out_channels, up_mode, kernel_size=2, stride=2):
    if up_mode == 'transposed':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer1, layer2):
        out = torch.cat((layer1, layer2), 1)
        return out


class DownBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 pooling=True,
                 activation='relu',
                 normalization=None,
                 conv_mode='same'):
        super().__init__()

        self.input_ch = input_channels
        self.output_ch = output_channels
        self.pooling = pooling
        self.activation = activation
        self.normalization = normalization

        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0

        #Capas de convolución
        self.conv1 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=3, stride=1, padding=self.padding,
                               bias=True)
        self.conv2 = nn.Conv2d(self.output_ch, self.output_ch, kernel_size=3, stride=1, padding=self.padding,
                               bias=True)

        #Pooling
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #Función de activación
        self.act1 = get_activation_function(self.activation)
        self.act2 = get_activation_function(self.activation)

        #Normalización
        if self.normalization:
            self.norm1 = get_normalization(self.normalization, self.output_ch)
            self.norm2 = get_normalization(self.normalization, self.output_ch)

    def forward(self, x):

        #Convolución-Activación-Normalización 1
        x = self.conv1(x)
        x = self.act1(x)
        if self.normalization:
            x = self.norm1(x)

        #Convolución-Activación-Normalización 2
        x = self.conv2(x)
        x = self.act2(x)
        if self.normalization:
            x = self.norm2(x)

        # Se guardan las salidas antes del pooling
        befoore_pooling = x

        if self.pooling:
            x = self.pool(x)

        return x, befoore_pooling


class UpBlock(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 activation='relu',
                 normalization=None,
                 conv_mode='same',
                 up_mode='transposed'):
        super().__init__()

        self.input_ch = input_channels
        self.output_ch = output_channels
        self.activation = activation
        self.normalization = normalization
        self.up_mode = up_mode

        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0

        #Sobremuestreo (up-sampling)
        self.up = get_up_layer(self.input_ch, self.output_ch, kernel_size=2, stride=2, up_mode=self.up_mode)

        #Capas convolucionales
        self.conv0 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(2 * self.output_ch, self.output_ch, kernel_size=3, stride=1, padding=self.padding, bias=True)
        self.conv2 = nn.Conv2d(self.output_ch, self.output_ch, kernel_size=3, stride=1, padding=self.padding, bias=True)

        #Función de activación
        self.act0 = get_activation_function(self.activation)
        self.act1 = get_activation_function(self.activation)
        self.act2 = get_activation_function(self.activation)

        #Normalización
        if self.normalization:
            self.norm0 = get_normalization(self.normalization, self.output_ch)
            self.norm1 = get_normalization(self.normalization, self.output_ch)
            self.norm2 = get_normalization(self.normalization, self.output_ch)

        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        #Se reducen la dimensión del canal con una capa de convolución
        up_layer = self.up(decoder_layer)
        cropped_encoder, dec_layer = autocrop(encoder_layer, up_layer)

        if self.up_mode != 'transposed':
            up_layer = self.conv0(up_layer)

        up_layer = self.act0(up_layer) #Todo: revisar

        if self.normalization:
            up_layer = self.norm0(up_layer)

        merged_layer = self.concat(up_layer, cropped_encoder)

        #Convolución-Activación-Normalización 1
        x = self.conv1(merged_layer)
        x = self.act1(x)
        if self.normalization:
            x = self.norm1(x)

        #Convolución-Activación-Normalización 2
        x = self.conv2(x)
        x = self.act2(x)
        if self.normalization:
            x = self.norm2(x)

        return x


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, n_blocks=4, start_filters=32, activation='relu',
                 normalization='batch', conv_mode='same', up_mode='transposed'):
        super().__init__()

        self.input_ch = in_channels
        self.out_ch = out_channels
        self.blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        for i in range(self.blocks):
            num_filters_in = self.input_ch if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.blocks - 1 else False

            down_block = DownBlock(input_channels=num_filters_in,
                                   output_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode)

            self.down_blocks.append(down_block)

        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(input_channels=num_filters_in,
                               output_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               conv_mode=self.conv_mode,
                               up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        #Convolucion final
        self.conv_final = nn.Conv2d(num_filters_out, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x):
        encoder_output = []

        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
