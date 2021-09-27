# %%
import os
import torch
from torchvision import transforms as T
from typing import Iterable
import numpy as np
import random
from typing import Sequence
from PIL import Image, ImageOps, ImageFilter
# %%

__all__ = ['RandomFlip', 'GaussianBlur',
           'PadResize', 'AllToTensor', 'ComposedTransform', 'Normalize']


class ComposedTransform:
    def __init__(self, transformers: Iterable) -> None:
        self.transformers = transformers

    def __call__(self, img, mask):
        for t in self.transformers:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transformers:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class Normalize:
    def __init__(self, mean: Sequence, std: Sequence) -> None:
        self.Nornaliztion = T.Normalize(mean, std)

    def __call__(self, img, mask):
        img = self.Nornaliztion(img)
        return img, mask


class AllToTensor:
    def __init__(self) -> None:
        self.to_tensor = T.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.tensor(np.array(mask)).to(torch.long)
        return img, mask


class RandomFlip:
    def __init__(self, probablity: float) -> None:
        self.probality = probablity

    def __call__(self, img, mask):
        if random.random() < self.probality:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class GaussianBlur:
    def __init__(self, probality: float) -> None:
        self.probality = probality

    def __call__(self, img, mask):
        if random.random() < self.probality:
            radius = random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            # mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
        return img, mask


class PadResize:
    """
        初始化参数：期望的宽和高(w, h)
    """

    def __init__(self, size: Iterable) -> None:
        self.expected_size = size

    def __call__(self, img, mask):
        ow, oh = img.size
        ew, eh = self.expected_size
        scale = min(ew/ow, eh/oh)
        nw = int(ow * scale + 0.5)
        nh = int(oh * scale + 0.5)
        img = img.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', self.expected_size, (0, 0, 0))
        new_image.paste(img, ((ew-nw)//2, (eh-nh)//2))
        mask = mask.resize((nw, nh), Image.NEAREST)
        new_mask = Image.new('L', self.expected_size, 255)
        new_mask.paste(mask, ((ew-nw)//2, (eh-nh)//2))
        return new_image, new_mask
        Image.new()
