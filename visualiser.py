import torch, copy
import matplotlib.pyplot as plt
import matplotlib
import cv2, os
import pandas as pd
import model
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import accuracy_score
import macros, metrics, helper, datahandler, model
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if str(device) != "cpu":
    matplotlib.use('Agg')  # trick to work in gpu server without connection to display for matplotlib - it prevents error as could not connect to any x display

def init_model(w_pth):
    my_model = model.getModel(using_unet=macros.using_unet, outputchannels=((4 if (not macros.unify_classes_first_and_third) else 3) if macros.cross_entropy_loss else 1))
    # Load the trained model
    weights_filepath = os.path.join(w_pth, 'weights.pt')
    my_model.load_state_dict(torch.load(weights_filepath, map_location=torch.device(device)))
    my_model.eval()  # Set model to evaluate mode
    return my_model


def get_img_from_path(path_to_patch):
    path_to_patch = str(Path(path_to_patch))
    img = cv2.imread(path_to_patch, 0 if macros.one_ch_in else 1)
    img = np.array(img)
    return img

PATCH_SIZE = 128
STEP_SIZE = 128
def calc_seg(img_path, model):
    print("started seg for ", img_path)
    img = get_img_from_path(img_path)
    w, h = img.shape[0], img.shape[1]
    mask = np.zeros((0, h))
    for cur_x in range(0, w, STEP_SIZE):
        cur_raw = np.zeros((PATCH_SIZE, 0))
        orig_x = 0
        if cur_x >= (w - PATCH_SIZE):  # margin
            orig_x = cur_x
            cur_x = w - PATCH_SIZE - 1
        for cur_y in range(0, h, STEP_SIZE):
            orig_y = 0
            if cur_y >= (h - PATCH_SIZE): # margin
                orig_y = cur_y
                cur_y = h - PATCH_SIZE - 1
            cur_patch = img[cur_x:cur_x + PATCH_SIZE, cur_y:cur_y + PATCH_SIZE]
            cur_patch = torch.from_numpy(cur_patch.reshape((1,1,) + cur_patch.shape))
            cur_patch = (cur_patch.type(torch.FloatTensor) / macros.IMG_MAX_VAL) - (0.5 if macros.norm_with_average_sub else 0)

            cur_patch = model(cur_patch)
            cur_patch = cur_patch.argmax(dim=1).detach().numpy().reshape(PATCH_SIZE, PATCH_SIZE)
            if STEP_SIZE != PATCH_SIZE:  # focus ion the center with 42
                assert STEP_SIZE == 42
                cur_patch = cur_patch[STEP_SIZE:2*STEP_SIZE, STEP_SIZE:2*STEP_SIZE]
            if orig_y > 0:  # margin
                cur_patch = cur_patch[:, orig_y - cur_y -1:]
            cur_raw = np.concatenate((cur_raw, cur_patch), axis=1)
        if orig_x > 0:  # margin
            cur_raw = np.array(cur_raw)[orig_x - cur_x -1:,:]
        cur_raw = np.array(cur_raw)
        mask = np.concatenate((mask, cur_raw), axis=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_DILATE, kernel, iterations=3)

    # plt.imshow(opening, 'gray')
    # plt.show()
    # plt.imshow(mask, 'gray')
    # plt.show()
    return img, opening


def show_three_imgs(x, labels_list=['image', 'mask', 'prediction'], out_path=''):

    f, axarr = plt.subplots(1, 3)
    f.set_size_inches(15, 5)
    plot_ndxs = [i for i in range(3)]
    for i in range(3):
        axarr[i].axis('off')
        if i == 0:
            axarr[i].imshow(x[i], cmap='gray')
        else:
            axarr[i].imshow(x[i])
        # f.colorbar(e, ax=axarr[plot_ndxs[i]], shrink=0.7)

        axarr[plot_ndxs[i]].set_title(labels_list[i])
    if len(str(out_path)) > 0:
        plt.savefig(out_path, dpi=200)
    else:
        plt.show()
    plt.close(fig=f)


def seg_for_seq(in_dir_path, gt_path, out_dir_path, w_pth):
    my_model = init_model(w_pth)
    imgs_list = sorted([os.path.join(in_dir_path, o) for o in os.listdir(in_dir_path)
                 if (not os.path.isdir(os.path.join(in_dir_path, o))) and (
                         o[-3:] == 'jpg' or o[-3:] == 'png' or o[-3:] == 'PNG' or o[-3:] == 'npz')])
    gts_list = sorted([os.path.join(gt_path, o) for o in os.listdir(gt_path)
                 if (not os.path.isdir(os.path.join(gt_path, o))) and (
                         o[-3:] == 'jpg' or o[-3:] == 'png' or o[-3:] == 'PNG' or o[-3:] == 'npz')])

    # colors of original labels
    cloud_map = np.array([[0., 0., 0.],
                          [0.75, 0.15, 0.1],
                          ([0.8, 0.8, 0.0] if macros.unify_classes_first_and_third else [0.1, 0.7, 0.2]),
                          [0.8, 0.8, 0.]])
    mapping = {
        0: 0.0,
        38: 1.0,
        75: 2.0,
        113: 3.0
    } if (not macros.unify_classes_first_and_third) else {
        0: 0.0,
        38: 1.0,
        75: 0.0,
        113: 2.0
    }

    seg_list = []
    seg_names_list = []
    acc = 0.0
    j=0.0
    for i, img_path in enumerate(imgs_list):
        gt = get_img_from_path(gts_list[i])
        # show_three_imgs([gt, gt, gt], out_path=os.path.join(w, "tripels", str(img_path)[-24:]))
        img, seg = calc_seg(img_path, my_model)

        seg = np.array([cloud_map[p] for p in seg.reshape(-1)]).reshape(seg.shape[0], seg.shape[1], 3)
        # gt = np.array([cloud_map[p] for p in gt.reshape(-1)]).reshape(gt.shape[0], gt.shape[1], 3)
        for k in mapping:
            gt[gt == k] = mapping[k]
        gt = np.array([cloud_map[p] for p in gt.reshape(-1)]).reshape(gt.shape[0], gt.shape[1], 3)

        # matplotlib.image.imsave(os.path.join(output_dir, str(img_path)[-24:]), seg)
        # show_three_imgs([img[:seg.shape[0], :seg.shape[1]], gt[:seg.shape[0], :seg.shape[1]], seg], out_path=w_pth  + os.path.join(str(in_dir_path)[-12:], str(img_path)[-24:]))
        show_three_imgs([img[:seg.shape[0], :seg.shape[1]], gt[:seg.shape[0], :seg.shape[1]], seg],
                        out_path=os.path.join(w_pth, str(img_path)[-24:]))
        # show_three_imgs([img[:seg.shape[0], :seg.shape[1]], gt[:seg.shape[0], :seg.shape[1]], seg])
        acc += np.array(seg == gt[STEP_SIZE:seg.shape[0]+STEP_SIZE, STEP_SIZE:seg.shape[1]+STEP_SIZE]).mean()
        j+=1
    acc = acc/j
    print(f'acc={acc} for path={in_dir_path}')

if __name__ == "__main__":
    for w in  ["C:\\Users\david565\Desktop\MSC\CNN\dlcourse\\finalProj\\testproj\\bbb\\gpu_results\\new_code\\latest\\exp_dir_2021_10_09_02_28_22", "C:\\Users\david565\Desktop\MSC\CNN\dlcourse\\finalProj\\testproj\\bbb\\gpu_results\\new_code\\latest\\exp_dir_2021_10_09_02_28_54"]:
        seg_for_seq(Path("C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_valid_img"), Path("C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_valid_msk"), "valid_mask", w)
        seg_for_seq(Path("C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_test_img"),  Path("C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_test_msk"), "test_mask", w)
        seg_for_seq(Path("C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_train_img"),  Path("C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_train_msk"), "train_mask", w)
