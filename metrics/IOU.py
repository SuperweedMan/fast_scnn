#%%
import numpy as np


class IOU:
    def __init__(self) -> None:
        pass

    def generate_iou(self, confusion_matrix:np.ndarray):
        IOUS = []
        for i in range(confusion_matrix.shape[0]):  # 每一行代表一种类别
            # TP/(TP+FP+FN)
            IOU = confusion_matrix[i, i] / (np.sum(confusion_matrix[i,:])+np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]+ 0.0000001)
            IOUS.append(IOU)
        MIOU = sum(IOUS) / len(IOUS)
        return IOUS, MIOU

#%%
if __name__ == '__main__':
    import sys
    sys.path.append('C:\\Users\\wyz\\codes\\FastSCNN')
    from data.cityscapes import CityScapesSeg
    from metrics.confusion_matrix import ConfusionMatrix
    ds = CityScapesSeg()
    cm = ConfusionMatrix()
    imgs, masks = zip(*list(ds[0:10]))
    masks = [np.array(x) for x in masks]
    CM, axis_labels = cm.get_confusion_matrix(np.array(masks), np.array(masks))

    iou = IOU()
    ious, miou = iou.generate_iou(CM)