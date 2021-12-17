import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import model
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans


def return_nearest_cluster(point, clusters):
    dists = np.abs(clusters - point)
    return clusters[np.argmin(dists)].item()

### CHECK THATS WORKING
def quantization(mask):
    out = np.zeros(shape=(400,400), dtype=float)
    flat_mask = mask.reshape(1, -1).T
    clusters = KMeans(n_clusters=5, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    for i in range(400):
        for j in range(400):
            out[i][j] = return_nearest_cluster(mask[i][j], clusters)
    return out


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


model = model.getModel()
# Load the trained model
# model.load_state_dict(torch.load("./exp_dir/weights.pt", map_location=torch.device('cpu')))
# Set the model to evaluate mode
model.eval()


def quantizationa(img, thresholds=(0.5, 0.7, 0.9, 1)):
    out = np.zeros(shape=(400, 400), dtype=float)
    for i in range(400):
        for j in range(400):
            coord = img[i][j]
            if coord <= thresholds[0]:
                out[i][j] = 0
            elif thresholds[0] < coord <= thresholds[1]:
                out[i][j] = 0.3
            elif thresholds[1] < coord <= thresholds[2]:
                out[i][j] = 0.6
            else:
                out[i][j] = 1
    return out


def img_to_mask(img_pt):
    img = cv2.imread(img_pt)
    img = cv2.resize(img, (400, 400)).transpose(2, 0, 1).reshape(1, 3, 400, 400)
    with torch.no_grad():
        mask = model(torch.from_numpy(img).type(torch.FloatTensor) / 255)
        quantized_mask = quantization(mask)
    return img, mask, quantized_mask


def create_dir_masks(in_pt, out_pt):
    import os
    imgs_and_masks = []
    for root, dirs, files in os.walk(in_pt):
        for name in files:
            img_pt = os.path.join(root, name)
            masks = img_to_mask(img_pt)
            imgs_and_masks.append(mask)

    return masks


def show_many_imgs(x, labels_list, amount, out_path=''):
    f, axarr = plt.subplots(amount, amount) # here it got stuck
    print("dddd2")
    plot_ndxs = [(i, j) for i in range(amount) for j in range(amount)]
    for i in range(amount * amount):
        print("dddd2")
        e = axarr[plot_ndxs[i][0], plot_ndxs[i][1]].imshow(x[i])
        f.colorbar(e, ax=axarr[plot_ndxs[i]], shrink=0.7)
        # if len(labels_list) != 0:
        axarr[plot_ndxs[i][0], plot_ndxs[i][1]].set_title(labels_list[i])
    if len(out_path) > 0:
        print("dddd2")
        plt.savefig(out_path)
        print("dddd3")
    else:
        plt.show()


def create_masks(data_dir, out_dir):
    results = []
    import datahandler
    dataloaders = datahandler.get_dataloader_sep_folder(
        data_dir, batch_size=1)
    k=0
    for phase in ['Train', 'Test']:
        model.eval()   # Set model to evaluate mode
        # Iterate over data.
        for sample in tqdm(iter(dataloaders[phase])):
            inputs = sample['image']
            mask = sample['mask']
            outputs = model(inputs)
            output = outputs['out'].cpu().detach().numpy()[0][0]
            quantized_mask = quantization(output)
            sample_results = [np.array(inputs[0]).transpose(1, 2, 0), mask[0][0], output, quantized_mask]
            results.append(sample_results)
            #show_many_imgs(sample_results[0], ["img", "real mask", "output", "output_mask"], 2, str(k) + ".png")
            k += 1
    import pickle
    pickle.dump(results, open("seg_results.p", "wb"))


# create_masks('./separated_data', 'results')
# create_dir_masks('separated_data\\Train\\Images', 'separated_data\\Train\\Predictions')

ino = 11
# Read  a sample image and mask from the data-set
img = cv2.imread('data/Train/Images/011.PNG')
# img = cv2.imread(f'./data/Images/{ino:03d}.PNG')
img = cv2.resize(img, (400, 400)).transpose(2, 0, 1).reshape(1, 3, 400, 400)
mask = cv2.imread('data/Train/Masks/011_label.tif')
plt.imshow(mask)
plt.show()
# mask = cv2.imread(f'./data/Masks/{ino:03d}_label.tif')
with torch.no_grad():
    a = model(torch.from_numpy(img).type(torch.FloatTensor) / 255)

plt.figure(figsize=(10, 10));
plt.subplot(131);
plt.imshow(img[0, ...].transpose(1, 2, 0));
plt.title('Image')
plt.axis('off');
plt.subplot(132);

plt.imshow(mask);
plt.title('Ground Truth')
plt.axis('off');
plt.subplot(133);
out = quantization(a.cpu().detach().numpy()[0][0])
# out = quantization(a['out'].cpu().detach().numpy()[0][0])
# plt.imshow(a['out'].cpu().detach().numpy()[0][0]);
plt.imshow(out)
plt.title('Segmentation Output')
plt.axis('off');
plt.show()
