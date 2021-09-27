#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Feature_fusion_module import FeatureFusionModule
from .Global_feature_extractor import GlobalFeatureExtractor
from .Learning_to_downsample import LearningToDownsample
from .Classifier import Classifer

#%%
class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)  # shape(b, 64, 128, 128)
        x = self.global_feature_extractor(higher_res_features)  # shape(b, 128, 32，32)
        x = self.feature_fusion(higher_res_features, x)   # shape(b, 128, 128, 128)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

if __name__ == '__main__':
    net = FastSCNN(10)
    x = torch.ones(2, 3, 1024, 1024)
    y = net(x)