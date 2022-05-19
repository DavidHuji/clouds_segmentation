import torch.optim as optim  # trial
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from model import getModel
from trainer import train_model
import datahandler
import argparse
import os
import torch
from pathlib import Path
from datetime import datetime
import macros, metrics, focal_loss
# torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_directory_path = "/Users/danu/Desktop/michal/overfit_small_data" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/new_data_for_ir_patches"

if macros.overfit_data:
    data_directory_path = "C:\\Users\\david565\\Desktop\\clouds_seg\\patches_maker\\overfit_data" if str(device) == "cpu" else "overfit_data"

if macros.five_classes:
    data_directory_path = "/Users/danu/Desktop/michal/new_masks_of_5_classes/fake_5classes_data_just_for_code_testing" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/final5classesPatches"

exp_directory_path = "exp_dir_" + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_directory", default=data_directory_path, help='Specify the dataset directory path')
parser.add_argument(
    "--exp_directory", default=exp_directory_path, help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=macros.epochs, type=int)
parser.add_argument("--batchsize", default=(1 if (macros.overfit_data or str(device) == "cpu") else macros.btch_size), type=int)

######
parser.add_argument("--num_classes", default=5, type=int)
parser.add_argument("--using_unet", default=macros.using_unet, type=int)
parser.add_argument("--train_all", default=macros.train_all, type=int)
######

args = parser.parse_args()

######
num_classes = 5 if macros.five_classes else (4 if (not macros.unify_classes_first_and_third) else 3) if macros.cross_entropy_loss else args.num_classes
train_all = (args.train_all==1)

other_than_five_classes = True if num_classes != 5 else False
######



bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize

# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = getModel(using_unet=macros.using_unet, train_all=train_all, outputchannels=(((4 if (not macros.unify_classes_first_and_third) else 3) if not macros.five_classes else 5) if macros.cross_entropy_loss else 1))
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)

lr=1e-4

# save configuration in text file
configuration = {
    'epochs': epochs,
    'augmentations': macros.augmentations,
    'train_all': macros.train_all,
    'cross_entropy_loss': macros.cross_entropy_loss,
    'focal_loss': macros.focal_loss,
    'using_unet': macros.using_unet,
    'using_michals_unet': macros.using_michals_unet,
    'overfit_data': macros.overfit_data,
    'weighted_loss': macros.weighted_loss,
    'one_ch_in': macros.one_ch_in,
    'norm_with_average_sub': macros.norm_with_average_sub,
    'unify_classes_first_and_third': macros.unify_classes_first_and_third,
    'use_only_single_class': macros.use_only_single_class,
    'use_gradient_accumulation': macros.use_gradient_accumulation,
    'batch_size': batchsize,
    'LR': lr
}
with open(os.path.join(bpath, 'metaInfo.txt'), 'w') as f:
    f.write(str(configuration).replace(',', '\n'))


# Specify the loss function
if macros.cross_entropy_loss:
    if macros.weighted_loss:
        if num_classes ==4:
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0/0.45, 1.0/0.25, 1.0/0.15, 1.0/0.17]).to(device))
        else:
            print(f'device={device}')
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0/(0.45 + 0.15), 1.0/0.25, 1.0/0.17]).to(device)) # 3.0, 8.0, 12.0
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if macros.focal_loss:
        criterion = focal_loss.FocalLoss(num_class=4 if (not macros.unify_classes_first_and_third) else 3)
else:
    criterion = torch.nn.MSELoss(reduction='mean')

if macros.five_classes:
    criterion = torch.nn.CrossEntropyLoss()
    # [0.28162515 0.23309405 0.19862842 0.15451344 0.13213895] is the statistics of the 5 classes (17.05)
    if macros.weighted_loss:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0 / 0.28, 1.0 / 0.23, 1.0 / 0.19, 1.0 / 0.15, 1.0 / 0.13]).to(device))
# Specify the optimizer with a lower learning rate
print(f'lr={lr}')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Specify the evalutation metrics
# metrics = {'f1_score': f1_score, 'IoU': metrics.IoU, 'accuracy' : metrics.cust_accuracy}
metrics = {'accuracy': metrics.cust_accuracy}  # , 'IoU': metrics.IoU}


print("Begin training process with the following properties:")
print(f'Number of epochs = {epochs}\nBatchsize = {batchsize}\nNumber of classes = {num_classes}\nUsing unet: {macros.using_unet}\nTrain the whole network: {train_all}')

# Create the dataloader
dataloaders = datahandler.get_dataloader_sep_folder(
    data_dir, batch_size=batchsize, other_than_5_classes=other_than_five_classes, num_classes=num_classes, with_aug=macros.augmentations)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, device=device, num_epochs=epochs, using_unet=macros.using_unet)


# Save the trained model
torch.save(model.state_dict(), os.path.join(bpath, 'weights.pt'))

#  calc confusion matrixes
import cm_calculator
final_results = cm_calculator.create_masks(data_dir, num_classes, bpath)
with open(os.path.join(bpath, 'metaInfo.txt'), 'a+') as f:
    f.write('\n\n' + str(final_results))

# calc visualisations
from visualiser import seg_for_seq

path_to_trained_weights = bpath
if macros.five_classes:
    path_to_gt_masks_test = "/Users/danu/Desktop/michal/5classesFinal17_5/Test_msk" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/5classesFinal17_5/Test_msk"
    path_to_gt_masks_train = "/Users/danu/Desktop/michal/5classesFinal17_5/Train_msk" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/5classesFinal17_5/Train_msk"
    path_to_images_test = "/Users/danu/Desktop/michal/5classesFinal17_5/Test_img" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/5classesFinal17_5/Test_img"
    path_to_images_train = "/Users/danu/Desktop/michal/5classesFinal17_5/Train_img" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/5classesFinal17_5/Train_img"
else:
    path_to_gt_masks_test = "/Users/danu/Desktop/michal/data/Test/Masks" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/new_data_for_ir_full_images/Test/Masks"
    path_to_gt_masks_train = "/Users/danu/Desktop/michal/data/Train/Masks" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/new_data_for_ir_full_images/Train/Masks"
    path_to_images_test = "/Users/danu/Desktop/michal/data/Test/Images" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/new_data_for_ir_full_images/Test/Images"
    path_to_images_train = "/Users/danu/Desktop/michal/data/Train/Images" if str(device) == "cpu" else "/home/gamir/DER-Roei/davidn/michal/new_data_for_ir_full_images/Train/Images"
seg_for_seq(Path(path_to_images_test), Path(path_to_gt_masks_test), "test_mask", path_to_trained_weights)
seg_for_seq(Path(path_to_images_train), Path(path_to_gt_masks_train), "train_mask", path_to_trained_weights)
