# -*- coding: utf-8 -*-
""" Set of utilities """
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from typing import List
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import recall_score, precision_score, accuracy_score

from constants import COARSE_TO_LABEL, VERB_TO_LABEL, NOUN_TO_LABEL, FINE_TO_LABEL, CLEAN_FINE, COLS
        
class ValueMeter(object):
    def __init__(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n):
        self.sum += value * n
        self.total += n

    def value(self):
        return self.sum / self.total
    

def calculate_metrics(pd_labels, gt_labels, class_names):

    result = OrderedDict()

    result["per_class_P"] = OrderedDict()
    result["per_class_R"] = OrderedDict()

    result["Acc"] = accuracy_score(gt_labels, pd_labels)
    per_class_precision = precision_score(gt_labels, pd_labels, average=None, zero_division=0)
    per_class_recall = recall_score(gt_labels, pd_labels, average=None, zero_division=0)

    print(per_class_precision)
    print(per_class_recall)
    print(result["Acc"])

    for idx, class_name in enumerate(class_names):
        result["per_class_P"][class_name] = per_class_precision[idx]
        result["per_class_R"][class_name] = per_class_recall[idx]

    return result


class BalancedSoftmax(nn.Module):
    """Implement the Balanced Soft proposed in paper "Balanced Meta-Softmax for Long-Tailed Visual
    Recognition" (NeurIPS 2020) https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification"""

    def __init__(
        self,
        num_samples_per_class: List[int],
        reduction: str = "mean",
    ):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(
            reduction = reduction,
        )

        self.num_samples_per_class = torch.FloatTensor(num_samples_per_class)

    def forward(self, input, target):
        spc = self.num_samples_per_class.to(input.device)
        spc = spc.unsqueeze(0).expand(input.shape[0], -1)
        input = input + spc.log()

        return self.criterion(input, target)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xx = x
    x = x.reshape((-1, x.shape[-1]))
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    res = e_x / e_x.sum(axis=1).reshape(-1, 1)
    return res.reshape(xx.shape)



def filelist_to_df(base:str, files:List[str]):
    dataframe = []
    for file in files:
        df = pd.read_csv(base + file, header=None, names=COLS)
        df['toy_name'] = file.split('-')[2].split('_')[0]
        df['video'] = file.replace('.csv', '')
        dataframe.append(df)
    df = pd.concat(dataframe)
    df = transform_row(df, clean=False)
    return df

def transform_row(df, clean):
    """
    For filelist types, need to convert to label constants
    """
    if not clean: 
        df['fine'] = df.apply(lambda row: CLEAN_FINE[row.label if pd.isna(row.remark) else row.remark], axis=1)
    df['fine'] = df.apply(lambda row: FINE_TO_LABEL[row.fine], axis=1)
    df['coarse'] = df.apply(lambda row: COARSE_TO_LABEL[row.label], axis=1)
    df['verb'] = df.apply(lambda row: VERB_TO_LABEL[row.verb], axis=1)
    df['this'] = df.apply(lambda row: NOUN_TO_LABEL[row.this], axis=1)
    df['that'] = df.apply(lambda row: NOUN_TO_LABEL[row.that], axis=1)
    return df