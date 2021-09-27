# %%
from types import LambdaType
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from sklearn import metrics



class ConfusionMatrix:
    def __init__(self) -> None:
        pass

    def get_confusion_matrix(self, predicts: np.ndarray, ground_truth: np.ndarray, labels=None, normalize=None):
        if (predicts.shape != ground_truth.shape):
            raise RuntimeError(
                'Shape of Predictions must be same of GroundTruth')
        cm = metrics.confusion_matrix(ground_truth.flatten(
        ), predicts.flatten(), labels=np.unique(ground_truth), normalize=normalize)
        axis_label = np.unique(ground_truth)
        if labels is not None:
            diff = list(set(labels).difference(set(axis_label)))
            diff.sort()
            labels = np.array(labels)
            if diff is not None:
                for label in diff:
                    idx = np.where(labels == label)[0]
                    cm = np.insert(cm, idx, 0, axis=0)
                    cm = np.insert(cm, idx, 0, axis=1)
        return cm, labels
    
    def save_fig(self, path, cm, cm_labels, labels:Dict, *arg, **kvarg):
        fig, ax = plt.subplots(1,1)
        ax0 = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(range(len(cm_labels)))
        ticklabels = [labels[k] for k in cm_labels]
        ax.set_xticklabels(ticklabels, rotation=90)
        ax.set_yticks(range(len(cm_labels)))
        ax.set_yticklabels(ticklabels)
        fig.colorbar(ax0, ax=ax)
        fig.savefig(path, **kvarg)
        


# %%
if __name__ == '__main__':
    import sys
    sys.path.append('C:\\Users\\wyz\\codes\\FastSCNN')
    from data.cityscapes import CityScapesSeg
    ds = CityScapesSeg()
    cm = ConfusionMatrix()
    imgs, masks = zip(*list(ds[0:2]))
    masks = [np.array(x) for x in masks]
    labels = np.unique(masks)
    labels = np.append(labels, 50)
    labels = np.append(labels, 17)
    labels = np.sort(labels)

    CM, axis_labels = cm.get_confusion_matrix(np.array(masks), np.array(masks), labels=labels, normalize='true')
    # CM, axis_labels = cm.get_confusion_matrix(np.array(masks), np.array(masks))
    fig, ax = plt.subplots(1, 1)
    # ax = ax.flatten()

    ax0 = ax.imshow(CM, cmap=plt.cm.Blues)
    # plt.xticks(range(len(labels)), labels, rotation=90)
    # plt.yticks(range(len(labels)), labels)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    # ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    # ax.set_xlabel(labels)
    fig.colorbar(ax0, ax=ax)
    plt.show()
    # plt.imshow(mask)
    # plt.show()
