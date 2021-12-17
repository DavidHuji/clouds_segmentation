import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np

def calc_confusion_matrix(input, targs):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    # n = targs.shape[0]
    # input = input.argmax(dim=1).view(-1)
    # targs = targs.view(-1)
    input, targs = np.vstack(input).reshape(-1), np.vstack(targs).reshape(-1)
    cm = confusion_matrix(targs, input)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def cust_accuracy(input, targs):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    acc = (input==targs).float().mean()
    return acc


def my_f1_score(input, targs):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(-1)
    targs = targs.view(-1)
    f1 = f1_score(targs, input, average='micro')
    # acc = (input==targs).float().mean()
    return f1



def IoU(preds, targs, eps=1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value
             for multi-class image segmentation
    """
    num_classes = preds.shape[1]

    # Single class segmentation?
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(preds)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)

    # Multi-class segmentation
    else:
        # Convert target to one-hot encoding
        # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
        true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]

        # Permute [B,H,W,C] to [B,C,H,W]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        # Take softmax along class dimension; all class probs add to 1 (per pixel)
        probas = torch.nn.functional.softmax(preds, dim=1)

    true_1_hot = true_1_hot.type(preds.type())
    # true_1_hot = true_1_hot.float()

    # Sum probabilities by class and across batch images
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)  # [class0,class1,class2,...]
    cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()  # find mean of class IoU values
    return iou.item()