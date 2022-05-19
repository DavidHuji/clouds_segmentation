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

arg_paths = {
    'train_data': ('C:\\Users\\david565\\Desktop\\original_IR4input\\train', 'dataIR\\Train\\Images'),
    'train_mask': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data\\vis_train_msk', 'dataIR\\Train\\Masks'),
    'test_data': ('C:\\Users\\david565\\Desktop\\original_IR4input\\test', 'dataIR\\Test\\Images'),
    'test_mask': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data_IR\\vis_test_msk', 'dataIR\\Test\\Masks'),
    'valid_data': ('C:\\Users\\david565\\Desktop\\original_IR4input\\valid', 'dataIR\\Valid\\Images'),
    'valid_mask': ('C:\\Users\\david565\\Desktop\\clouds_seg\\data_IR\\vis_valid_msk', 'dataIR\\Valid\\Masks')
}

# the following paths are for the new IR images was done in 24/03 (new mac)
arg_paths = {
    'train_data': ('/Users/danu/Desktop/michal/new_data_for_ir_full_images/Train/Images', '/Users/danu/Desktop/michal/new_data_for_ir_patches/Train/Images'),
    'train_mask': ('/Users/danu/Desktop/michal/new_data_for_ir_full_images/Train/Masks', '/Users/danu/Desktop/michal/new_data_for_ir_patches/Train/Masks'),
    'test_data': ('/Users/danu/Desktop/michal/new_data_for_ir_full_images/Test/Images', '/Users/danu/Desktop/michal/new_data_for_ir_patches/Test/Images'),
    'test_mask': ('/Users/danu/Desktop/michal/new_data_for_ir_full_images/Test/Masks', '/Users/danu/Desktop/michal/new_data_for_ir_patches/Test/Masks'),
}

# the following paths are for the new masks of five classes (1.5)
arg_paths = {
    'train_mask': ('/Users/danu/Desktop/michal/new_masks_of_5_classes/full_iamge_masks', '/Users/danu/Desktop/michal/new_masks_of_5_classes/masks_patches')
}

# the following paths are for the new masks of five classes (17.5)
arg_paths = {
    'train_data': ('/Users/danu/Desktop/michal/5classesFinal17_5/Train_img', '/Users/danu/Desktop/michal/final5classesPatches/Train/Images'),
    'train_mask': ('/Users/danu/Desktop/michal/5classesFinal17_5/Train_msk', '/Users/danu/Desktop/michal/final5classesPatches/Train/Masks'),
    'test_data': ('/Users/danu/Desktop/michal/5classesFinal17_5/Test_img', '/Users/danu/Desktop/michal/final5classesPatches/Test/Images'),
    'test_mask': ('/Users/danu/Desktop/michal/5classesFinal17_5/Test_msk', '/Users/danu/Desktop/michal/final5classesPatches/Test/Masks'),
}

out_size = 128
drop_confusing_patches_rate = 0.97 * 0  # use zero to for disable
hist_stat = np.zeros(5)

only_measure_statistics = False
# img = np.ones((30, 30))
W, H = out_size, out_size
W_jump, H_jump = int(out_size/2), int(out_size/2)
img_w, img_h = 1256, 1213

def see_mask():
    in_path = "/Users/danu/Desktop/michal/new_masks_of_5_classes/full_iamge_masks/2018_08_09_00_00_IR108_truth.png"
    img = plt.imread(in_path)
    img = np.array(img)
    print("aaaa", img.shape)
    plt.imshow(img)
    plt.show()

see_mask()


def imag_to_patches(im):
    if len(im.shape) == 3:
        im = im[:,:,0:3]  # prevent cases of 4 channels
    tiles = [im[y:y+H, x:x+W] for x in range(0, img_w, W_jump) if x + W < img_w for y in range(0, img_h, H_jump) if y + H < img_h]
    tiles_with_dominante_class = []
    channels_amount = np.array(tiles[0]).shape[-1]
    global hist_stat
    print(f'tiles amount = {len(tiles)}')
    for t in tiles:
        h = np.histogram(np.array(t), bins=hist_stat.shape[0])[0]
        hist_stat = hist_stat + h
        if drop_confusing_patches_rate > 0:
            h = np.histogram(np.array(t), bins=hist_stat.shape[0])[0]
            print(h)
            if h.max() >= (channels_amount * out_size * out_size * drop_confusing_patches_rate):
                tiles_with_dominante_class.append(t)
        if t.shape != (H, W, 3) and t.shape != (H, W):
            print(f'error shape={t.shape}')
            exit(1)
    print(len(tiles_with_dominante_class))

    if drop_confusing_patches_rate > 0:
        return tiles_with_dominante_class
    # print(hist_stat / np.sum(hist_stat))
    return tiles, hist_stat


def file_to_many(in_path, name, out_dir_path):
    img = plt.imread(in_path)
    # img = cv2.imread(in_path, flags=(0 if True and only_measure_statistics else 1))
    img = np.array(img)
    patches, hist_stat = imag_to_patches(img)

    # dirty hack to prevent name issues
    # if name[-5:] == 'IR108':
    #     name = name[:-6]
    for i, cur_patch in enumerate(patches):
        print(i)
        path = Path(out_dir_path) / f'{name}_ptch_{i}.png'
        print(cur_patch.shape)
        img_saver.imsave(path,
                         cur_patch, cmap='gray')
    return hist_stat


def dir_to_dir(in_path, out_dir_path):
    if not os.path.isdir(out_dir_path):
        os.mkdir(out_dir_path)
    in_imgs_paths_list = [os.path.join(in_path, o) for o in os.listdir(in_path)
                          if (not os.path.isdir(os.path.join(in_path, o))) and (
                                      o[-3:] == 'jpg' or o[-3:] == 'png' or o[-3:] == 'PNG' or o[-3:] == 'npz')]
    for file_path in in_imgs_paths_list:
        file_to_many(file_path, os.path.basename(file_path)[:-4], out_dir_path)
    print('stats of ',in_path , hist_stat / np.sum(hist_stat))


def handle_all_data():
    global hist_stat
    for key in arg_paths.keys():
        hist_stat = np.zeros(5)
        dir_to_dir(arg_paths[key][0], arg_paths[key][1])


if __name__ == '__main__':
    handle_all_data()