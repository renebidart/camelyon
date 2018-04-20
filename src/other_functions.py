import os
import sys
import glob
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torchvision.models as models
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

def bagging_ensemble_acc(models_arch, models_name, PATH, train_folder, csv_fname, weights=None):
    """ Test list of pre-trained fastai models using csv format data

    train_folder - folder with all the images (all in one directory, not label-folder format)
    csv_fname - location of csv where data is stored
    PATH - path to dataset
    model_arch - list of models
    models_name - list of names of models weights

    Model parameters (models_name) must be in same order as models_arch
    Weights is used for final ensemble accuracy
    """

    if weights is None:
        weights = [1/len(models_arch)]*len(models_arch)
    all_preds = []
    all_y = []
    int_acc_list = []
    
    for idx, arch in enumerate(models_arch):
        if arch in [inceptionresnet_2, inception_4]:
            sz = 299
        else:
            sz = 224

        tfms = tfms_from_model(arch, sz, aug_tfms=None, max_zoom=1)
        
        # do data from path, because this is the same split, just doesn't req
        # get the validation indices based on this
        data = ImageClassifierData.from_csv(PATH, train_folder, csv_fname, tfms=tfms, 
                                            val_idxs=val_idxs, bs=64)
        
        model_loc = os.path.join(PATH, 'models', models_name[idx])
        learn = ConvLearner.pretrained(arch, data, precompute=False)
        print(learn.model)
        learn.load(model_loc)
        # this returns log probs, but don't need to do anything because just taking the argmax
        preds, y = learn.predict_with_targs(is_test=False)
        print(models_name[idx], ':   ', accuracy_np(preds, y))
        all_preds.append(preds)
        all_y.append(y)

        curr_preds = np.array(all_preds)
        curr_probs = np.average(np.exp(curr_preds), axis=0)
        int_acc = accuracy_np(curr_probs, y)
        int_acc_list.append(int_acc)
        print(models_name[idx], 'int acc:   ', int_acc_list)

    all_preds = np.array(all_preds)
    probs = np.average(np.exp(all_preds), axis=0, weights=weights)
    print(accuracy_np(probs, all_y))
    return int_acc

def bagging_ensemble_acc_dir(models_arch, models_name, models_path, PATH, weights=None):
    """ Test list of pre-trained fastai models using the keras-style paths 

    models_path - standard fastai format directory for models
    PATH - path to dataset
    model_arch - list of models
    models_name - list of names of models weights
    Model parameters (models_name) must be in same order as models_arch
    Weights is used for final ensemble accuracy
    """
    if weights is None:
        weights = [1/len(models_arch)]*len(models_arch)
    all_preds = []
    all_y = []
    int_acc_list = []
    
    for idx, arch in enumerate(models_arch):
        if arch in [inceptionresnet_2, inception_4]:
            sz = 299
        else:
            sz = 224

        tfms = tfms_from_model(arch, sz, aug_tfms=None, max_zoom=1)
        
        # do data from path, because this is the same split, just doesn't req
        # get the validation indices based on this
        data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=64)

        model_loc = os.path.join(models_path, 'models', models_name[idx])
        learn = ConvLearner.pretrained(arch, data, precompute=False)
        learn.load(model_loc)
        # this returns log probs, but don't need to do anything because just taking the argmax
        preds, y = learn.predict_with_targs(is_test=False)
        print(models_name[idx], ':   ', accuracy_np(preds, y))
        all_preds.append(preds)
        all_y.append(y)

        curr_preds = np.array(all_preds)
        curr_probs = np.average(np.exp(curr_preds), axis=0)
        int_acc = accuracy_np(curr_probs, y)
        int_acc_list.append(int_acc)
        print(models_name[idx], 'int acc:   ', int_acc)

    all_preds = np.array(all_preds)
    probs = np.average(np.exp(all_preds), axis=0, weights=weights)
    print(accuracy_np(probs, all_y))
    return int_acc_list


def make_validation_mask(data_df, normal_valid, tumor_valid):
    img_names = data_df['file_name'].tolist()
    valid_idxs = [is_validation(name, normal_valid, tumor_valid) for name in img_names]
    return valid_idxs


def get_best_ensemble_acc(models, models_name, PATH):
    perf = []
    all_preds = []
    
    for idx, arch in enumerate(models):
        if arch in [inceptionresnet_2, inception_4]:
            sz = 299
        else:
            sz = 224

        tfms = tfms_from_model(resnet50, sz, aug_tfms=transforms_top_down, max_zoom=1)
        data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=64)

        model_loc = os.path.join(PATH, 'models', models_name[idx])
        learn = ConvLearner.pretrained(arch, data, precompute=False)
        learn.load(model_loc)
        preds, y = learn.predict_with_targs(is_test=False)
        print(models_name[idx], ':   ', accuracy_np(preds, y))
        perf.append(accuracy_np(preds, y))
        all_preds.append(preds)
        
    # Now try to find the optimal weighting. Use exponential to test.
    all_preds = np.array(all_preds)

    for power in range(0, 21, 3):
        _perf = [1/(1-x) for x in perf]
        _perf = np.power(_perf, power)
        weights = _perf/(len(_perf)*np.mean(_perf))
        print(weights)
        probs = np.average(np.exp(all_preds), axis=0, weights=weights)
        print('Power:', power, 'Accuracy: ', accuracy_np(probs, y))
    return perf