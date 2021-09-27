#%%
import torch

checkpoint = torch.load('C:\\Users\\wyz\\codes\\FastSCNN\\checkpoint\\train_data\\ckpt_epoch_18')

#%%
import matplotlib.pyplot as plt

loss = checkpoint['losses']
miou = checkpoint['mious']
plt.plot(range(len(loss)), loss)
plt.plot(range(len(miou)), miou)
plt.show()