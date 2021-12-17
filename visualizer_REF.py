

#  Writen by David Nukrai, April 2020, Tel Aviv University

from __future__ import print_function, division

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

from matplotlib import image as img_saver
from pathlib import Path

from PIL import Image
import copy

PATCH_SIZE = 120
STEP_SIZE = int(PATCH_SIZE * 0.125)  # 0.25
checkpoint = 'saved_ws_4classes_sec.pt'


data_directory_path = "C:\\Users\\david565\\Desktop\\clouds_seg\\patches_maker\\data" if str(device) == "cpu" else "data"

val_transformer = transforms.Compose([
    transforms.Resize(120),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def show_imgs(imgs_list, labels_list):
    """

    :param imgs_list:
    :param labels_list:
    :return:
    """
    f, axarr = plt.subplots(2, int(len(imgs_list) / 2))

    for i in range(len(imgs_list)):
        if ((i % 2) == 0):
            axarr[i % 2, int(i / 2)].imshow(imgs_list[i], 'gray')
        else:
            axarr[i % 2, int(i / 2)].imshow(imgs_list[i])
        if len(labels_list) != 0:
            axarr[i % 2, int(i / 2)].set_title(labels_list[i])

    plt.show()


def init_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint, map_location=device)

    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4 if FOUR_CLASSES else (3 if THREE_CLASSES else 5))

    model_ft.load_state_dict(state_dict)
    model_ft.eval()
    return model_ft


def patch_preidction(model_ft, input_patch):
    outputs = model_ft(input_patch)
    _, preds = torch.max(outputs, 1)
    return int(preds.data)


def get_patch_from_path(path_to_patch):
    path_to_patch = Path(path_to_patch)
    img = Image.open(path_to_patch).convert('RGB')
    img = np.array(img)
    image_tensor = val_transformer(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    return input


def get_img_from_path(path_to_patch):
    path_to_patch = str(Path(path_to_patch))
    img = Image.open(path_to_patch).convert('RGB')
    img = np.array(img)
    return img


def get_img_from_path____a(path_to_patch):
    path_to_patch = str(Path(path_to_patch))
    img = plt.imread(path_to_patch)
    img_3d = np.ones((img.shape[0], img.shape[1], 3))
    img_3d[:, :, 0] = img[:]
    img_3d[:, :, 1] = img[:]
    img_3d[:, :, 2] = img[:]
    return img_3d


def get_coor_from_name(name):
    x, y = str(name)[:-4].split(' ')
    x = x.split('(')[-1][:-1]
    y = y[:-1]
    x = int(int(x) / STEP_SIZE)
    y = int(int(y) / STEP_SIZE)
    return (x, y)


def create_seg_from_ptchs(model, dir_path):
    seg = np.zeros((40, 40))
    imgs_list = [os.path.join(dir_path, o) for o in os.listdir(dir_path)
                 if (not os.path.isdir(os.path.join(dir_path, o))) and (
                             o[-3:] == 'jpg' or o[-3:] == 'png' or o[-3:] == 'PNG' or o[-3:] == 'npz')]
    for img_path in imgs_list:
        x, y = get_coor_from_name(img_path)
        ptch = get_patch_from_path(img_path)

        lbl = patch_preidction(model, ptch)
        seg[x][y] = lbl


def calc_seg(img_path, model):
    print("started seg for ", img_path)
    img = get_img_from_path(img_path)

    w, h = img.shape[0], img.shape[1]
    seg = np.zeros((int((w - PATCH_SIZE) / STEP_SIZE) + 1, int((h - PATCH_SIZE) / STEP_SIZE) + 1))

    for cur_x in range(0, w - PATCH_SIZE, STEP_SIZE):
        for cur_y in range(0, h - PATCH_SIZE, STEP_SIZE):
            cur_patch = img[cur_x:cur_x + PATCH_SIZE, cur_y:cur_y + PATCH_SIZE, :]
            temp_filepth = "tempfile.png"
            img_saver.imsave(temp_filepth, cur_patch, cmap='gray')
            path_to_patch = Path(temp_filepth)
            cur_patch = Image.open(path_to_patch).convert('RGB')

            cur_patch = val_transformer(cur_patch).float()
            cur_patch = cur_patch.unsqueeze_(0)
            cur_patch = Variable(cur_patch)

            lbl = patch_preidction(model, cur_patch)
            seg[int(cur_x / STEP_SIZE)][int(cur_y / STEP_SIZE)] = lbl

    return img, seg


def seg_for_seq(dir_path):
    my_model = init_model()
    imgs_list = [os.path.join(dir_path, o) for o in os.listdir(dir_path)
                 if (not os.path.isdir(os.path.join(dir_path, o))) and (
                         o[-3:] == 'jpg' or o[-3:] == 'png' or o[-3:] == 'PNG' or o[-3:] == 'npz')]
    seg_list = []
    seg_names_list = []
    for img_path in imgs_list:
        img, seg = calc_seg(img_path, my_model)
        seg_list.append(img)
        seg_list.append(seg)
        seg_names_list.append(str(img_path)[-40:])
        seg_names_list.append('mask')
    show_imgs(seg_list, seg_names_list)
    exit(0)


if __name__ == "__main__":
    seg_for_seq('clouds_patches//actual_seg//2016_10//15_3seq')
