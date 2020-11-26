import os
import scipy
from scipy import ndimage
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from math import ceil
from PIL import Image as PILImage
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def evaluate_main(save_path, model, loader, num_classes, type = 'val'):
    """Create the model and start the evaluation process."""

    ignore_label = 255
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    model.eval()
    model.cuda()

    confusion_matrix = np.zeros((num_classes,num_classes))
    palette = get_palette(256)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for index, batch in enumerate(loader):
        if index % 100 == 0:
            print('%d processd'%(index))
        if type == 'val':
            image, label, size, name = batch
        elif type == 'test':
            image, size, name = batch
        size = size[0].numpy()
        with torch.no_grad():
            image = image.data
            output = model(image.cuda())
            if isinstance(output, list):
                output = output[0]
            interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
            output = interp(output).cpu().data[0].numpy().transpose(1,2,0)

        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        if type == 'test': seg_pred = id2trainId(seg_pred, id_to_trainid, reverse=True)

        output_im = PILImage.fromarray(seg_pred)
        output_im.putpalette(palette)
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        output_im.save(save_path+'/'+name[0]+'.png')

        if type == 'val':
            seg_gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, num_classes)

    if type == 'val':
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        return mean_IU, IU_array

