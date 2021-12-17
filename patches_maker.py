import numpy as np
import os, cv2
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib import image as img_saver


### args
arg_paths = {
    'train_data': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_train_img', 'data\\Train\\Images'),
    'train_mask': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_train_msk', 'data\\Train\\Masks'),
    'test_data': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_test_img', 'data\\Test\\Images'),
    'test_mask': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_test_msk', 'data\\Test\\Masks'),
    'valid_data': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_valid_img', 'data\\Valid\\Images'),
    'valid_mask': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_valid_msk', 'data\\Valid\\Masks')
}

out_size = 128
drop_confusing_patches_rate = 0.97 * 0  # use zero to for disable
hist_stat = 0
hist_stat = np.zeros(4)

only_measure_statistics = True
# img = np.ones((30, 30))
W, H = out_size, out_size
W_jump, H_jump = int(out_size/2), int(out_size/2)
img_w, img_h = 1256, 1213

def see_mask():
    in_path = "C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_test_img\\2016_09_05_10_00_vis.png"
    img = plt.imread(in_path)
    img = np.array(img)
    print("aaaa", img.shape)
    plt.imshow(img)
    plt.show()
# see_mask()


def imag_to_patches(im):
    tiles = [im[y:y+H, x:x+W] for x in range(0, img_w, W_jump) if x + W < img_w for y in range(0, img_h, H_jump) if y + H < img_h]
    tiles_with_dominante_class = []
    channels_amount = np.array(tiles[0]).shape[-1]
    global hist_stat
    print(f'tiles amount = {len(tiles)}')
    for t in tiles:
        h = np.histogram(np.array(t), bins=4, range=(0.0, 114.0))[0]
        hist_stat = hist_stat + h
        if drop_confusing_patches_rate > 0:
            h = np.histogram(np.array(t), bins=4)[0]
            print(h)
            if h.max() >= (channels_amount * out_size * out_size * drop_confusing_patches_rate):
                tiles_with_dominante_class.append(t)
        if t.shape != (H, W, 3) and t.shape != (H, W):
            print(f'error shape={t.shape}')
            exit(1)
    print(len(tiles_with_dominante_class))

    if drop_confusing_patches_rate > 0:
        return tiles_with_dominante_class
    print(hist_stat / np.sum(hist_stat))
    return tiles


def file_to_many(in_path, name, out_dir_path):
    img = plt.imread(in_path)
    # img = cv2.imread(in_path, flags=(0 if True and only_measure_statistics else 1))
    img = np.array(img)
    patches = imag_to_patches(img)
    if only_measure_statistics:
        return
    for i, cur_patch in enumerate(patches):
        print(i)
        path = Path(out_dir_path) / f'{name}_ptch_{i}.png'
        print(cur_patch.shape)
        img_saver.imsave(Path(path),
                         cur_patch, cmap='gray')


def dir_to_dir(in_path, out_dir_path):
    in_imgs_paths_list = [os.path.join(in_path, o) for o in os.listdir(in_path)
                          if (not os.path.isdir(os.path.join(in_path, o))) and (
                                      o[-3:] == 'jpg' or o[-3:] == 'png' or o[-3:] == 'PNG' or o[-3:] == 'npz')]
    for file_path in in_imgs_paths_list:
        file_to_many(file_path, os.path.basename(file_path)[:-4], out_dir_path)


def handle_all_data():
    for key in arg_paths.keys():
        dir_to_dir(arg_paths[key][0], arg_paths[key][1])

handle_all_data()