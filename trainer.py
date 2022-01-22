import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import macros
from helper import show_img
import metrics

def return_nearest_cluster(point, clusters):
    dists = np.abs(clusters - point)
    return clusters[np.argmin(dists)].item()


def quantization(mask, num_classes):
    out = np.zeros(shape=mask.shape, dtype=float)
    flat_mask = mask.reshape(1, -1).T
    clusters = KMeans(n_clusters=num_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    for i in range(mask.shape[0]):
        out[i] = return_nearest_cluster(mask[i], clusters)
    return out


def indexed_cluster_map(mask, clusters):
    out = np.zeros(shape=mask.shape, dtype=np.uint8)
    length = mask.shape[0]
    for i in range(length):
        dists = np.abs(clusters - mask[i])
        out[i] = np.argmin(dists)
    return out


def accuracy_per_sample(quantized_output, true_mask, num_classes):
    flat_mask = true_mask.reshape(1, -1).T
    kmeans = KMeans(n_clusters=num_classes, random_state=0, max_iter=500).fit(flat_mask).cluster_centers_
    clusters = sorted([kmeans[i].item() for i in range(len(kmeans))])
    index_mask = indexed_cluster_map(true_mask, clusters)
    index_output = indexed_cluster_map(quantized_output, clusters)
    return accuracy_score(index_mask.flatten(), index_output.flatten())


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, device, num_epochs=3, using_unet=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_acc = 0
    # Use gpu if available
    print(f'device={device}')
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
                 [f'Train_{m}' for m in metrics.keys()] + \
                 [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    model.train()
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:          
            if phase == 'Train':
                model.train()  # Set model to training mode
                
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            progress_in_epoch = 0
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].float().to(device)
                masks = sample['mask'].long().to(device)

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    if using_unet:
                        outputs = {'out': outputs}

                    loss = criterion(outputs['out'], (masks.squeeze(1) if macros.cross_entropy_loss else masks))

                    for name, metric in metrics.items():
                        batchsummary[f'{phase}_{name}'].append(float(metric(outputs['out'], masks).item()))

                    if phase == 'Train':
                        loss.backward()
                        if macros.use_gradient_accumulation > 1:
                            if progress_in_epoch % macros.use_gradient_accumulation == 0:
                                optimizer.step()
                        else:
                            optimizer.step()
                    optimizer.zero_grad()
                progress_in_epoch += 1
                epoch_loss = loss.item()
                batchsummary[f'{phase}_loss'].append(epoch_loss)

            batchsummary['epoch'] = epoch

        for field in fieldnames[1:]:
            batchsummary[field] = np.mean(np.array(batchsummary[field]))
        epoch_loss = batchsummary['Test_loss']
        epoch_acc = batchsummary['Test_accuracy']
        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model          
            if phase == 'Test' and epoch_acc > best_acc:  # can also change to: epoch_loss < best_loss:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Test Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
