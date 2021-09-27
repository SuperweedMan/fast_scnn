# %%
from PIL import Image, ImageOps, ImageFilter
import torch.utils.data as tdata
from torchvision import transforms as T
import numpy as np
import torch
import random
from pathlib import Path
import os
import sys
sys.path.append('..')
from config import Config
from typing import List, Tuple, Dict, Iterable

# %%
class CityScapesSeg(tdata.Dataset):
    """
    参数：
    """
    def __init__(self, root=Path(Config['data_root_path']), split='train', transform=None) -> None:
        super().__init__()
        self.class_num = 19
        self.root = root
        self.split = split
        self.transform = transform
        self.img_paths, self.mask_paths = self._get_raw_data_paths(self.split)
        assert(len(self.img_paths) == len(self.mask_paths))
        if len(self.img_paths) == 0:
            raise RuntimeError("Found 0 images in subfolders of: {}".format(self.root))

    def __getitem__(self, index):
        if isinstance(index, slice):
            self.img_paths = self.img_paths[index]
            self.mask_paths = self.mask_paths[index]
            return self
        img = Image.open(self.img_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('L')
        if self.transform is not None:
            return self.transform(img, mask)
        return img, mask 

    def __len__(self):
        return len(self.img_paths)

    def _get_path_pairs(self, mask_folder, img_folder):
        mask_paths = []
        img_paths = []
        for root, dirs, files in os.walk(img_folder):
            # print('number of imgs: {}'.format(files)[:])
            for file_name in files:
                if Path(file_name).suffix == '.png':
                    img_path= root / Path(file_name)
                    mask_folder_name = Path(img_path.parent.stem)
                    mask_file_name = file_name.replace('leftImg8bit', 'gtFine_labelTrainIds')
                    mask_path = mask_folder / mask_folder_name / Path(mask_file_name)
                    if os.path.isfile(img_path) and os.path.isfile(mask_path):
                        img_paths.append(img_path)
                        mask_paths.append(mask_path)
        return img_paths, mask_paths
        # print(img_paths[0], mask_paths[0])


    def _get_raw_data_paths(self, split='train'):
        _types = ['train', 'val']
        if split in _types:
            img_folder = self.root / \
                Path('leftImg8bit_trainvaltest') / \
                Path('leftImg8bit') / Path(split)
            mask_folder = self.root / Path('gtFine') / Path(split)
            return self._get_path_pairs(mask_folder, img_folder)
        else:
            raise Exception(
                'data split must in {}, input is {}'.format(_types, split))


# %%
if __name__ == '__main__':
    from img_transformer import *
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    transformer = ComposedTransform([
        RandomFlip(0.5),
        GaussianBlur(0.5),
        PadResize([1980, 1080]),
        AllToTensor()
    ])
    city_dataset = CityScapesSeg(transform=transformer)[:1]
    dl = tdata.DataLoader(city_dataset)
    to_PIL = T.ToPILImage()
    for img, mask in dl:
        img = to_PIL(make_grid(img))
        mask = to_PIL(make_grid(mask))
    plt.show(img)
    plt.show(mask)