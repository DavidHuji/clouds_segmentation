from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
from torchvision import transforms, utils
import random
import macros


class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None,
                 imagecolormode='grayscale' if macros.one_ch_in else 'rgb', maskcolormode='grayscale'):
                 # imagecolormode='rgb', maskcolormode='grayscale'):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            imagecolormode: 'rgb' or 'grayscale'
            maskcolormode: 'rgb' or 'grayscale'
        """
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert (imagecolormode in ['rgb', 'grayscale'])
        assert (maskcolormode in ['rgb', 'grayscale'])

        self.imagecolorflag = self.color_dict[imagecolormode]  # 1 if images are rgb and 0 if grayscale
        self.maskcolorflag = self.color_dict[maskcolormode]  # 1 if masks are rgb and 0 if grayscale
        self.root_dir = root_dir  # "./data" in our case
        self.transform = transform
        if not fraction:  # if fraction==0 then don't split to train & test
            self.image_names = sorted(
                ### List of elements of the form ['./data\\Images\\001.jpg', './data\\Images\\002.jpg',....]
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                ### List of elements of the form ['./data\\Masks\\001_label.PNG', './data\\Masks\\002_label.PNG'....]
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))

        else:  # Here we split to train and test
            assert (subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]  # Same as image_list but shuffled
                self.mask_list = self.mask_list[indices]  # Same as mask_list but shuffled

            if subset == 'Train':  # Slice the first fraction of images amd masks as training set
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:  # Use the rest as test set
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction))):]

        if macros.use_only_single_class:
            print("use only patches with single class")
            self.mapping = {
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
            self.good_indexes = set()
            self.bad_indexes = set()  # with_more_than_single_class
            for i in range(0, len(self.image_names)):
                msk_name = self.mask_names[i]
                if self.maskcolorflag:
                    mask = cv2.imread(msk_name, self.maskcolorflag).transpose(2, 0, 1)
                else:
                    mask = cv2.imread(msk_name, self.maskcolorflag)

                if self.is_there_more_than_one_class(mask):
                    self.bad_indexes.add(i)
                else:
                    self.good_indexes.add(i)


            self.good_indexes = list(self.good_indexes)
            print(f'\namount of single class patches={len(self.good_indexes)}, percentage={len(self.good_indexes) / len(self.image_names)}\n')

    def is_there_more_than_one_class(self, mask):
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        h = np.histogram(np.array(mask))[0]
        return h.max() < (128 * 128 * 0.97)

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):  # Returns a dictionary of image and mask as numpy arrays according to color type
        if macros.use_only_single_class:
            if idx in self.bad_indexes:
                idx = random.choice(self.good_indexes)
        # print(f'index={idx}')
        img_name = self.image_names[idx]
        msk_name = self.mask_names[idx]
        assert img_name.split('/')[-1][:16] == msk_name.split('/')[-1][:16]  # make sure the images names and the masks names are aligned

        if self.maskcolorflag:
            mask = cv2.imread(msk_name, self.maskcolorflag).transpose(2, 0, 1)
        else:
            mask = cv2.imread(msk_name, self.maskcolorflag)

        if self.imagecolorflag:
            image = cv2.imread(
                # cv2.imread returns a numpy array in the form (H,W,C) so we reshape it to our (C,H,W) form
                img_name, self.imagecolorflag).transpose(2, 0, 1)
        else:
            image = cv2.imread(img_name, self.imagecolorflag)
        sample = {'image': np.array(image), 'mask': np.array(mask)}


        if self.transform:  # Performs transform on the image if given any. The transformers are suitable for our dict structure
            sample = self.transform(sample)

        return sample


# Define few transformations for the Segmentation Dataloader

class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize):
        self.imageresize = imageresize
        self.maskresize = imageresize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:  # if rgb, change to cv2 structure from (C,H,W)-->(H,W,C)
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
        image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)  # Change back to our (H,W,C) structure
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = np.array(sample['image']), sample['mask']
        if len(mask.shape) == 2:
            mask = mask.reshape((1,) + mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,) + image.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class ToPILIMAGE(object):
    """Convert ndarrays in sample to PIL."""

    def __call__(self, sample):
        to_pil = transforms.ToPILImage()
        image, mask = sample['image'], sample['mask']
        # print(f'ToPILIMAGE: After converting to tensor,'
        #      f'and before converting to PIL shape= = {image.shape} and type = {type(image)}')
        if len(mask.shape) == 2:
            mask = mask.reshape((1,) + mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,) + image.shape)

        return {'image': to_pil(image),
                'mask': to_pil(mask)}


class PIL_to_numpy(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(np.array(image).shape) == 3:
            image = np.array(image).transpose(2, 0, 1)
        # print(f'PIL_to_numpy: Now we convert from PIL to numpy, shape after converting = {image.shape} and type={type(image)}')
        mask = np.array(mask)
        # print(f'PIL_to_numpy: Now we convert from PIL to numpy, mask shape after converting = {mask.shape} and type={type(mask)}')
        return {'image': image,
                'mask': mask}


class RandomHorizontalFlip(object):
    """Randomly flip the sample horizontally"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        prob = random.uniform(0, 1)
        if prob <= 0.5:
            flip = transforms.RandomHorizontalFlip(p=1)
            image, mask = flip(image), flip(mask)
            return {'image': image, 'mask': mask}
        else:
            return {'image': image, 'mask': mask}


class RandomHVerticalFlip(object):
    """Randomly flip the sample horizontally"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        prob = random.uniform(0, 1)
        if prob <= 0.5:
            flip = transforms.RandomVerticalFlip(p=1)
            image, mask = flip(image), flip(mask)
            return {'image': image, 'mask': mask}
        else:
            return {'image': image, 'mask': mask}


class Normalize(object):
    '''Normalize mask'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': (image.type(torch.FloatTensor) / macros.IMG_MAX_VAL) - (0.5 if macros.norm_with_average_sub else 0),
                'mask': mask.type(torch.FloatTensor) / (1 if macros.cross_entropy_loss else macros.MSK_MAX_VAL)}


class ColorJitter(object):
    def __init__(self, contrast, brightness=0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        jitter = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast)
        return {'image': jitter(image), 'mask': mask}


class mask_to_n_class(object):
    def __init__(self, num_classes):
        self.n = num_classes
        self.mapping = {
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

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return {'image': image, 'mask': mask}


def get_transformer_for_test(num_classes):
    return transforms.Compose([mask_to_n_class(num_classes),
                                            ToTensor(),
                                            Normalize()])

def get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=4,
                              other_than_5_classes=False, num_classes=5, with_aug=False):
    """
        Create Train and Test dataloaders from two separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Mask1
        ---------MaskN
        --Test
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Mask1
        ---------MaskN
    """
    if with_aug:
        data_transforms = {
            'Train': transforms.Compose([mask_to_n_class(num_classes),
                                         ToTensor(),
                                         ToPILIMAGE(),
                                         RandomHorizontalFlip(),
                                         RandomHVerticalFlip(),
                                         PIL_to_numpy(),
                                         ToTensor(),
                                         Normalize()]),
            'Valid': get_transformer_for_test(num_classes),
            'Test': get_transformer_for_test(num_classes)
        }
    else:
        data_transforms = {
            'Train': get_transformer_for_test(num_classes),
            'Test': get_transformer_for_test(num_classes),
            'Valid': get_transformer_for_test(num_classes)
        }

    image_datasets = {x: SegDataset(root_dir=os.path.join(data_dir, x),
                                    transform=data_transforms[x], maskFolder=maskFolder, imageFolder=imageFolder)
                      for x in ['Train', 'Test', 'Valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True)
                   for x in ['Train', 'Test', 'Valid']}
    # shuffle=True, num_workers=(4 if device != "cpu" else 0), pin_memory=(device != "cpu"))
    return dataloaders
