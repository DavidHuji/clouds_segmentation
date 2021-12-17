import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


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


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=3, using_unet=True):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
                 [f'Train_{m}' for m in metrics.keys()] + \
                 [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

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
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device).float()
                masks = sample['mask'].to(device).float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    if using_unet:
                        outputs = {'out': outputs}

                    loss = criterion(outputs['out'], masks)

                    y_pred = outputs[
                        'out'].data.cpu().numpy().ravel()  # Move to cpu, convert to numpy and flatten to a long vector
                    y_true = masks.data.cpu().numpy().ravel()  # Move to cpu, convert to numpy and flatten to a long vector
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))


                        elif name == 'accuracy':
                            batchsummary[f'{phase}_{name}'].append(
                                accuracy_per_sample(quantization(y_pred, 3), y_true, 3))

                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(
                phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model