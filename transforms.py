
import random
import numbers
import torch.nn
import torchvision.transforms

import numpy as np
import torchvision.transforms.functional as F

from collections.abc import Sequence
from torchvision import *
from torch import Tensor
from torchvision.transforms.transforms import _setup_size
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int


def custom_transform(input, mask):
    input, mask = F.to_pil_image(input), F.to_pil_image(mask)

    w, h = input.size

    # Random horizontal flipping
    if random.random() > 0.5:
        input = F.hflip(input)
        mask = F.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        input = F.vflip(input)
        mask = F.vflip(mask)

    if random.random() > 0.3:
        parametros = torchvision.transforms.RandomAffine(180).get_params(degrees=[-90, 90], translate=[0, 0],
                                                                         scale_ranges=[1, 1], shears=[0, 0],
                                                                         img_size=[w, h])
        input, mask = F.affine(input, *parametros), F.affine(mask, *parametros)

    input = np.array(input)
    mask = np.array(mask)

    input, mask = F.to_tensor(input), F.to_tensor(mask)

    return input, mask


class SegmentationRandomCrop(torch.nn.Module):

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", background=0):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.background = background

    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.background, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.background, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.background, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(mask, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class SegmentationResize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set True for
            ``InterpolationMode.BILINEAR`` only mode.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    """

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = F._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, mask):
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias), \
               F.resize(mask, self.size, self.interpolation, self.max_size, self.antialias)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)


class SegmentationToTensor:
    def call(self, img, mask):
        return F.to_tensor(img), F.to_tensor(mask)

    def repr(self):
        return self.__class__.__name__ + '()'


class SegmentationCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class SegmentationVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(mask)
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SegmentationHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(mask)
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SegmentationRandomRotation(torch.nn.Module):
    def __init__(
            self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = degrees

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def get_params(self, a, b):
        angle = np.random.uniform(a, b)
        #print(f'El angulo de rotacion es: {angle}')
        return angle

    def forward(self, img, mask):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees[0], self.degrees[1])

        mask_fill = [62, 39, 173]

        return F.rotate(img, angle, self.resample, self.expand, self.center, fill),\
               F.rotate(mask, angle, self.resample, self.expand, self.center, mask_fill)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

