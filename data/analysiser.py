# %%
from PIL import Image
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset
import sys
# %%


class DataAny:
    def __init__(self, ds: dataset, batch:int = 200) -> None:
        self.ds = ds
        self.max_classes = 256
        self.batch = batch

    def draw_pixel_histogram(self):
        # if not hasattr(self.ds, 'num_classes'):
        #     raise RuntimeError("Dataset do not have attribut \"num_classes\"")
        # np.zeros((self.ds.num_classes))
        value = np.zeros(self.max_classes, dtype=np.int64)  # 一共可以分256个种类
        print('---------------------------------------\n' +
              'Len of dataset: {}\n'.format(len(self.ds)) + 
              '---------------------------------------')

        for i in range(int(len(self.ds) / (self.batch)) + 1):
            imgs = list(zip(*(list(self.ds[i * self.batch : (i+1)*self.batch]))))[0]
            imgs = [np.array(x) for x in imgs]
            imgs = np.array(imgs)
            imgs = imgs.flatten()
            print("idx: {}".format(i*self.batch), end='\r')
            sys.stdout.flush()
            for idx in range(self.max_classes):
                value[idx] += int(np.sum(imgs == idx))
        value = value[np.where(value > 0)]
        return value

    


# %%
if __name__ == '__main__':
    import sys
    sys.path.append('C:\\Users\\wyz\\codes\\FastSCNN')
    from cityscapes import CityScapesSeg
    ds = CityScapesSeg()
    ay = DataAny(ds[:3])
