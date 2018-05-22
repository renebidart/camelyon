import os
import sys
import glob
import shutil
import random
import pickle
import numpy as np
from PIL import Image
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.models import resnet18, resnet34, resnet50, resnet101, vgg16
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms.functional as F

import pretrainedmodels.utils as utils
from sklearn.metrics import confusion_matrix




def make_batch_gen(PATH, batch_size, num_workers, valid_name='valid', test_name=None, size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        valid_name: transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    if test_name!=None:
        data_transforms[test_name] = data_transforms[valid_name]
        
    image_datasets = {x: datasets.ImageFolder(os.path.join(PATH, x),
                                              data_transforms[x])
                      for x in list(data_transforms.keys())}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in list(data_transforms.keys())}

    dataset_sizes = {x: len(image_datasets[x]) for x in list(data_transforms.keys())}
    return dataloaders, dataset_sizes


def make_batch_gen_pretrained(PATH, batch_size, num_workers, valid_name='valid', size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(20)
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        valid_name: transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(PATH, x),
                                              data_transforms[x])
                      for x in ['train', valid_name]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', valid_name]}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', valid_name]}
    return dataloaders, dataset_sizes


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    use_gpu = True
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                # for nets that have multiple outputs such as inception
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o,labels) for o in outputs)) # output is (outputs, aux_outputs)
                else:
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    if isinstance(outputs, tuple):
                        _, preds = torch.max(outputs[0].data, 1)
                    else:
                        _, preds = torch.max(outputs.data, 1)
                    loss.backward()
                    optimizer.step()
                else:
                    if isinstance(outputs, tuple):
                        _, preds = torch.max(outputs[0].data, 1)
                    else:
                        _, preds = torch.max(outputs.data, 1)
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            # stop those memory leaks
            del loss, outputs 

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return best_acc, model


def eval_model(model, dataloader, dataset_size, criterion):
    model.train(False)  # Set model to evaluate mode
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # forward
        outputs = model(inputs)

        # for nets that have multiple outputs such as inception
        if isinstance(outputs, tuple):
            loss = sum((criterion(o,labels) for o in outputs))
        else:
            loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data[0] * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    del loss, outputs 
    
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc



def get_preds(model, dataloader, dataset_size, criterion):
    """Return label, prediction, prediction (rounded)"""
    model.train(False)  # Set model to evaluate mode
    model.eval()

    all_labels, all_preds = [], []

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        all_labels.extend(labels.data.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        del outputs 

    return all_labels, all_preds

def get_preds_fusion(model, model_list, dataloader, dataset_size):
    model.train(False)  # Set model to evaluate mode
    model.eval()

    all_labels, all_preds = [], []

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        ######### Get model outputs
        features = []
        for model_tmp in model_list:
            output = model_tmp(inputs)
            features.append(output)
        cat_features = torch.cat(features, 1)
        ###########
        
        # forward
        outputs = model(cat_features)
        _, preds = torch.max(outputs.data, 1)
    
        all_labels.extend(labels.data.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        del outputs 
    
    return all_labels, all_preds



def get_metrics(all_labels, all_preds):
    """https://stackoverflow.com/questions/31324218"""
    metrics = {}
    cm = confusion_matrix(all_labels, all_preds)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    metrics['TPR'] = TP/(TP+FN)
    # Specificity or true negative rate
    metrics['TNR'] = TN/(TN+FP) 
    # Precision or positive predictive value
    metrics['PPV'] = TP/(TP+FP)
    # Negative predictive value
    metrics['NPV'] = TN/(TN+FN)
    # Fall out or false positive rate
    metrics['FPR'] = FP/(FP+TN)
    # False negative rate
    metrics['FNR'] = FN/(TP+FN)
    # False discovery rate
    metrics['FDR'] = FP/(TP+FP)
    # Overall accuracy
    metrics['ACC'] = (TP+TN)/(TP+FP+FN+TN)
    return metrics

def get_metrics_bin(all_labels, all_preds):
    metrics = {}
    cm = confusion_matrix(all_labels, all_preds)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    # Sensitivity, hit rate, recall, or true positive rate
    metrics['TPR'] = TP/(TP+FN)
    # Specificity or true negative rate
    metrics['TNR'] = TN/(TN+FP) 
    # Precision or positive predictive value
    metrics['PPV'] = TP/(TP+FP)
    # Negative predictive value
    metrics['NPV'] = TN/(TN+FN)
    # Fall out or false positive rate
    metrics['FPR'] = FP/(FP+TN)
    # False negative rate
    metrics['FNR'] = FN/(TP+FN)
    # False discovery rate
    metrics['FDR'] = FP/(TP+FP)
    # Overall accuracy
    metrics['ACC'] = (TP+TN)/(TP+FP+FN+TN)
    return metrics


############################  FUSION STUFF  ###########################

class WeightedSum(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.fc1 = nn.Linear(num_input, 2)

    def forward(self, x):
        out = self.fc1(x)
        return out

def train_fusion_model(model, model_list, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
        
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                    
                ######### Get model outputs
                features = []
                for model_tmp in model_list:
                    output = model_tmp(inputs)
                    features.append(output)
                cat_features = torch.cat(features, 1)
                    
                ###########
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(cat_features)

                # for nets that have multiple outputs such as inception
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o,labels) for o in outputs))
                else:
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    _, preds = torch.max(outputs.data, 1)
                    loss.backward()
                    optimizer.step()
                else:
                    _, preds = torch.max(outputs.data, 1)

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('saving model with acc ', epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_fusion_model(model, model_list, dataloader, dataset_size, criterion):
    model.train(False)  # Set model to evaluate mode
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        ######### Get model outputs
        features = []
        for model_tmp in model_list:
            output = model_tmp(inputs)
            features.append(output)
        cat_features = torch.cat(features, 1)
        ###########
        
        # forward
        outputs = model(cat_features)
        
        # for nets that have multiple outputs such as inception
        if isinstance(outputs, tuple):
            loss = sum((criterion(o,labels) for o in outputs))
        else:
            loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data[0] * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    del loss, outputs 
    
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
