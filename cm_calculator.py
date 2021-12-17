import torch, copy
import matplotlib.pyplot as plt
import cv2, os
import pandas as pd
import model
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import accuracy_score
import macros, metrics, helper


def return_nearest_cluster(point, clusters):
    dists = np.abs(clusters - point)
    return clusters[np.argmin(dists)].item()


def quantization(mask, num_classes):
    out = np.zeros(shape=(400, 400), dtype=float)
    flat_mask = mask.reshape(1, -1).T
    clusters = KMeans(n_clusters=num_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    for i in range(400):
        for j in range(400):
            out[i][j] = return_nearest_cluster(mask[i][j], clusters)
    return out


def indexed_cluster_map(mask, clusters):
    out = np.zeros(shape=mask.shape, dtype=np.uint8)
    height, width = mask.shape[0], mask.shape[1]
    for h in range(height):
        for w in range(width):
            dists = np.abs(clusters - mask[h][w])
            out[h][w] = np.argmin(dists)
    return out


def accuracy_per_sample(quantized_output, true_mask, num_classes):
    flat_mask = true_mask.reshape(1, -1).T
    kmeans = KMeans(n_clusters=num_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    clusters = sorted([kmeans[i].item() for i in range(len(kmeans))])
    index_mask = indexed_cluster_map(true_mask, clusters)
    index_output = indexed_cluster_map(quantized_output, clusters)
    return accuracy_score(index_mask.flatten(), index_output.flatten())


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def quantization_fix_thresholds(img, thresholds=(0.5, 0.7, 0.9, 1)):
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


def img_to_mask(img_pt, num_classes):
    img = cv2.imread(img_pt)
    img = cv2.resize(img, (400, 400)).transpose(2, 0, 1).reshape(1, 3, 400, 400)
    with torch.no_grad():
        mask = model(torch.from_numpy(img).type(torch.FloatTensor) / 255)
        quantized_mask = quantization(mask, num_classes)
    return img, mask, quantized_mask


def create_dir_masks(in_pt, num_classes):
    import os
    imgs_and_masks = []
    for root, dirs, files in os.walk(in_pt):
        for name in files:
            img_pt = os.path.join(root, name)
            mask = img_to_mask(img_pt)
            imgs_and_masks.append(mask)
    return imgs_and_masks


def show_many_imgs(x, labels_list, amount, out_path=''):
    """
    Doesn't work in Nova
    Args:
        x:
        labels_list:
        amount:
        out_path:

    Returns:

    """
    f, axarr = plt.subplots(amount, amount)  # here it got stuck
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


def create_hist(i):
    rng = np.random.RandomState(10)  # deterministic random data
    a = np.hstack((rng.normal(size=1000),
                   rng.normal(loc=5, scale=2, size=1000)))
    _ = plt.hist(i.reshape(-1), bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    # Text(0.5, 1.0, "Histogram with 'auto' bins")
    plt.show()


def create_masks(data_dir, num_classes, weights_filename):
    results = []

    import datahandler, model

    ####
    other_than_five_classes = True if num_classes != 5 else False
    ####

    dataloaders = datahandler.get_dataloader_sep_folder(
        data_dir, batch_size=(macros.btch_size if (macros.overfit_data or str(device) == "cpu") else macros.btch_size), other_than_5_classes=other_than_five_classes, num_classes=num_classes, with_aug=False)

    model = model.getModel(using_unet=macros.using_unet, outputchannels=((4 if (not macros.unify_classes_first_and_third) else 3) if macros.cross_entropy_loss else 1))
    # Load the trained model
    weights_filepath = os.path.join(weights_filename, 'weights.pt')
    model.load_state_dict(torch.load(weights_filepath, map_location=torch.device('cpu')))
    # model.eval()  # Set model to evaluate mode
    model.train()  # Set model to evaluate mode Use batch noorm also for prediction
    imgs_to_show = []
    final_results = {}
    for phase in ['Valid', 'Test', 'Train']:  # Test Train or Valid
        i, total_accuracy, cm_counter = 0, 0, 0
        ground_truth, predictions = [], []
        cm = np.zeros((num_classes, num_classes))
        # Iterate over data.
        for sample in tqdm(iter(dataloaders[phase])):
#            if i > 1000:
#                break
            inputs = sample['image']
            # mask = cv2.resize(np.array(sample['mask']), (400, 400), cv2.INTER_AREA)
            # inputs = cv2.resize(sample['image'], (400, 400), cv2.INTER_AREA)
            mask = sample['mask']
            # classes_amount = len(np.unique(mask))

            output = model(inputs)
            for single_image_idx in range(output.shape[0]):
                i += 1
                single_out = output[single_image_idx].unsqueeze(0)
                single_mask = mask[single_image_idx].unsqueeze(0)
                acc = metrics.cust_accuracy(single_out, single_mask)
                total_accuracy = (total_accuracy + acc)

                ground_truth.append(single_mask.view(-1))
                predictions.append(single_out.argmax(dim=1).view(-1))

                if i % 2 == 0:
                    new_cm = metrics.calc_confusion_matrix(predictions, ground_truth)
                    print('\n\n\n', new_cm, '\n')
                    ground_truth.clear()
                    predictions.clear()
                    if cm.shape == new_cm.shape:
                        cm_counter += 1
                        cm = (cm + new_cm)
                        print(cm / cm_counter)
                        print(cm_counter)
                    print("ACC  -  ", total_accuracy / i)
                # iou = metrics.my_f1_score(output, mask)

                SHOW = False
                if SHOW:
                    imgs_to_show.append(copy.deepcopy(inputs[single_image_idx].numpy().reshape(128, 128)))
                    imgs_to_show.append(copy.deepcopy(single_mask.numpy().reshape(128, 128)))
                    imgs_to_show.append(copy.deepcopy(single_out.argmax(dim=1).numpy().reshape(128, 128)))

                    if len(imgs_to_show) == 9:
                        helper.show_many_imgs(imgs_to_show, [str(i) for i in range(9)], 3)
                        imgs_to_show.clear()

                # sample_results = [np.array(inputs[0]).transpose(1, 2, 0), single_mask[0][0], single_out.argmax(dim=1)]
                # results.append(sample_results)


        # cm = metrics.calc_confusion_matrix(predictions, ground_truth)
        cm = cm / cm_counter
        final_results[phase] = (cm, total_accuracy / i)
        print("\n\nconfusion matrix:\n", cm)
        #####
    print(f'For architechture {weights_filename}: {final_results}')
    # pickle.dump(results, open("seg_results.p", "wb"))
    return final_results


if __name__ == '__main__':
    # 0.758 accuracy
    # create_masks("C:\\Users\\david565\\Desktop\\clouds_seg\\patches_maker\\data", 4, "C:\\Users\david565\Desktop\MSC\CNN\dlcourse\\finalProj\\testproj\\bbb\\gpu_results\\focaloss_michalUneet20E")
    test_big_images = False
    create_masks(("C:\\Users\\david565\Desktop\clouds_seg\data\data" if  test_big_images else "C:\\Users\\david565\\Desktop\\clouds_seg\\patches_maker\\data" if not macros.overfit_data else "C:\\Users\\david565\\Desktop\\clouds_seg\\patches_maker\\overfit_data"), 3,
                 "C:\\Users\david565\Desktop\MSC\CNN\dlcourse\\finalProj\\testproj\\bbb\\gpu_results\\new_code\\exp_dir_2021_10_03_06_27_48GREATORESULTS-LATEST-NO-SINGLECLASS")
