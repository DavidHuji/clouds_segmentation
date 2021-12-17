import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import model
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from torchvision import transforms as T
import datahandler as d
import pickle

##################################################################################################
##################################################################################################
################################                                    ##############################
################################               inputs               ##############################
################################                                    ##############################
##################################################################################################
##################################################################################################

data_pt = 'C:\\Users\\david565\\Downloads\\2016_10'  # path to a directory that contains all of the images
weights_path = ".//weights//3class_75isopen_trainall_resnet_withmetrics_withnegatives.pt"  # model's weights
excel_out_path = 'Example4_2016_10.xlsx'  # Path to excel file to save the final results
good_name_suffix = ['00_00_IFR', '02_00_IFR', '04_00_IFR', '06_00_IFR', '08_00_IFR', '10_00_VIS', '12_00_VIS', '14_00_VIS', '16_00_VIS', '18_00_VIS', '20_00_IFR', '22_00_IFR']

model = model.getModel()
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()


def return_nearest_cluster(point, clusters):
    dists = np.abs(clusters - point)
    return clusters[np.argmin(dists)].item()


def quantization(mask, n_classes=3):
    out = np.zeros(shape=(400,400), dtype=float)
    flat_mask = mask.reshape(1, -1).T
    clusters = KMeans(n_clusters=n_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    for i in range(400):
        for j in range(400):
            out[i][j] = return_nearest_cluster(mask[i][j], clusters)
    return out


transformer = T.Compose([d.Resize((400, 400), (400, 400)),  # Now its a ndarray of shape (3,400,400)
                   d.ToTensor()  # Now its a tensor of shape (3,400,400)
                   ])


def calc_mask_percentage(mask):
    mask_values = np.unique(mask)
    mask_values.sort()
    percentages = []
    total = mask.shape[0] * mask.shape[1]
    for v in mask_values:
        amount = np.sum(mask == v)
        percentages.append(amount / total)
    return percentages


def write_results_to_xl(results_dict):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(excel_out_path)
    worksheet = workbook.add_worksheet()

    # Start from the first cell.
    # Rows and columns are zero indexed.
    row = 0
    # iterating through content list
    for name in results_dict.keys():
        column = 0
        worksheet.write(row, column, name)
        for val in results_dict[name]:
            column += 1
            worksheet.write(row, column, val)

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    workbook.close()


def name_to_skip(img_time_suffix):
    return img_time_suffix not in good_name_suffix


def create_percentages_for_dir(data_pt):
    pathlist = Path(data_pt).glob('*')
    results_dict = {}
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        if name_to_skip(path_in_str[-13:-4]):
            continue

        img = cv2.imread(path_in_str)
        img = cv2.resize(img, (400, 400)).transpose(2, 0, 1).reshape(1, 3, 400, 400)
        if 'IFR' in path_in_str[-24:-4]:
            img = 255 - img
        img = torch.from_numpy(img).type(torch.FloatTensor) / 255
        with torch.no_grad():
            output = model(img)

        output = output['out'].cpu().detach().numpy()[0][0]
        output = quantization(output)

        percentages = calc_mask_percentage(output)
        results_dict[path_in_str] = percentages
        print(f'process image: {path_in_str},    |||   Percentages = {percentages}')
        # plt.imshow(output)
        # plt.title(str(percentages))
        # plt.show()
    write_results_to_xl(results_dict)
    return results_dict


if __name__ == '__main__':
    res_dict = create_percentages_for_dir(data_pt)
    pickle.dump(res_dict, open("long_period_results.p", "wb"))
