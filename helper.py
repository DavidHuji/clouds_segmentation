from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2


def save_imgs(imgs_list, labels_list, out_path):
    f, axarr = plt.subplots(2, int(len(imgs_list) / 2))
    plot_ndxs = [(i, j) for i in range(2) for j in range(int(len(imgs_list) / 2))]
    for i in range(len(imgs_list)):
        #              e = axarr[i % 2, int(i/2)].imshow(imgs_list[i], 'gray')
        # print("\n\nshape: ", imgs_list.shape, axarr.shape )
        e = axarr[i % 2, int(i / 2)].imshow(imgs_list[i])
        f.colorbar(e, ax=axarr[plot_ndxs[i]], shrink=0.7)
        if len(labels_list) != 0:
            axarr[i % 2, int(i / 2)].set_title('Coeffs_amount=' + str(labels_list[i]))
    if len(out_path) > 0:
        plt.savefig(out_path)
    else:
        plt.show()


def show_ten_by_ten(x, labels_list, out_path=''):
    f, axarr = plt.subplots(10, 10)
    plot_ndxs = [(i, j) for i in range(10) for j in range(10)]
    for i in range(100):
        e = axarr[plot_ndxs[i][0], plot_ndxs[i][1]].imshow(x[:, min(i, x.shape[1] - 1)].reshape(14, 20))
        f.colorbar(e, ax=axarr[plot_ndxs[i]], shrink=0.7)
        axarr[plot_ndxs[i][0], plot_ndxs[i][1]].set_title('ndx=' + str(labels_list[min(i, len(labels_list) - 1)]))
    if len(out_path) > 0:
        plt.savefig(out_path)
    else:
        plt.show()


def show_many_imgs(x, labels_list, amount, out_path=''):
    f, axarr = plt.subplots(amount, amount)
    plot_ndxs = [(i, j) for i in range(amount) for j in range(amount)]
    for i in range(amount * amount):
        e = axarr[plot_ndxs[i][0], plot_ndxs[i][1]].imshow(x[i])
        f.colorbar(e, ax=axarr[plot_ndxs[i]], shrink=0.7)
        # if len(labels_list) != 0:
        axarr[plot_ndxs[i][0], plot_ndxs[i][1]].set_title(labels_list[i])
    if len(out_path) > 0:
        plt.savefig(out_path)
    else:
        plt.show()


def show_img(in_img, ds=False):
    import cv2
    if ds:
        in_img = cv2.resize(in_img, (20, 16), interpolation=cv2.INTER_AREA)
    fig, axs = plt.subplots(1, 1)
    e = axs.imshow(in_img)
    fig.colorbar(e, ax=axs, shrink=0.5)
    plt.show()


def pickle_to_png_files(pickle_pt):
    import pickle
    list_of_results = pickle.load(open(pickle_pt, "rb"))
    ndx = 0
    for res in list_of_results:
        show_many_imgs(res, ['Image', 'Ground truth', 'Mask', 'Quantized Mask'], 2, f'results\img_{ndx}')
        ndx += 1


# Receives np image and returns its negative
def IR_to_negative(ir_image):
    def to_negative(x):
        return 255-x

    s = ir_image.shape
    negative = np.array(list(map(to_negative, ir_image.reshape(-1)))).reshape(s)
    return negative

# Receives path to a folder containing IR images, creates negative of each image and saves it to output dir
def folder_content_to_negative(in_dir, out_dir):
    images = glob.glob(in_dir+"/*")
    i=0
    for image in images:
        negative = IR_to_negative(cv2.imread(image))
        cv2.imwrite(out_dir+"/negative_"+str(i)+".png", negative)
        i += 1



if __name__ == '__main__':
    pickle_to_png_files('seg_results.p')
