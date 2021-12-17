from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
from torchvision import transforms as T, utils
import datahandler as d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


## IMPORTANT!! in order to pass tensors to matplotlib we need shape = (H,W,C)


ds= d.SegDataset("./data1", "Images", "Masks")

bright = [0]
contrasts = [(2.8,3)]

for b in bright:
    for c in contrasts:
        t = T.Compose([d.Resize((400, 400), (400, 400)),  # Now its a ndarray of shape (3,400,400)
                       d.ToTensor(),  # Now its a tensor of shape (3,400,400)
                       d.ToPILIMAGE(),  # Change to PIL image in order to use RandomHorizontalFlip
                       d.ColorJitter(brightness=b, contrast=c),
                       d.PIL_to_numpy()])
        sample = t(ds[3])
        image = sample['image']
        cv2.imshow(f'bright={b}, contrast={c}', image.transpose(1, 2, 0))
        cv2.waitKey()



"""
sample = trans(ds[6])
image, mask = sample['image'], sample['mask']
image = image.numpy().transpose(1,2,0)
mask = np.reshape(mask.numpy(), (400,400)).transpose(0,1)
plt.hist(mask)
plt.show()
flat_mask = mask.reshape(1,-1).T
clusters = KMeans(n_clusters=5, random_state=0, max_iter=500).fit(flat_mask)
plt.imshow(image)
plt.show()
plt.imshow(mask)
plt.show()
"""
img = ds[0]['mask']
print(img)
