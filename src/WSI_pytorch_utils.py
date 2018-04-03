import os
import sys
import glob
import time
import random
import numpy as np
import copy
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from WSI_utils import*


class WSIDataset(Dataset):
    """Sample from the slides indicated by the wsi. 
    
    Switch turning the imgs to batches into the Dataset rather than the dataloader.
    
    Standard pytorch dataloader wants to return one img at a time, 
    so instead set batch_size=1 and return all the imgs at once.
    Set the length to 100 000

    Must check if having one batch from the same slide and of one class is a problem
    
    """
    SEED = 101
    random.seed(SEED)

    def __init__(self, data_loc, normal_nums, tumor_nums, batch_size, length=100000, transforms=None):
        """nums is a list of """
        
        all_data = glob.glob(data_loc+'/**/*.tif', recursive=True)  
        self.normal_locs = [loc for loc in all_data if any(str(x) in loc for x in normal_nums) and 'normal' in loc.lower()]
        self.tumor_locs = [loc for loc in all_data if any(str(x) in loc for x in tumor_nums) and 'tumor' in loc.lower() and 'mask' not in loc.lower()]
#         self.tumor_mask_locs = [loc for loc in all_data if any(str(x) in loc for x in tumor_nums) and 'mask' in loc.lower()]
        self.all_locs = self.normal_locs + self.tumor_locs

        self.batch_size = batch_size
        self.length = length
        self.transforms = transforms

    def __len__(self):
        # we tell pytorch we are using a batch size of 1, so need to scale down the length
        return int(self.length/self.batch_size)

    def __getitem__(self, index):
        """Easiest way is to return half of each batch as tumor and non-tumor.
                
        We don't care about a sampler method, or the indices. 
        At each call of __getitem__ we randomly select 2 WSIs. There is no iterating over the dataset.
        """
        
        num_tiles = int(self.batch_size/2)
        
        tumor_loc = random.choice(self.tumor_locs)        
        tumor_wsi = WSI(tumor_loc)
        tumor_imgs = tumor_wsi.sample_batch_tumor_region(num_tiles, tile_size=224)
        
        normal_loc = random.choice(self.all_locs)
        normal_wsi = WSI(normal_loc)
        normal_imgs = normal_wsi.sample_batch_normal_region(num_tiles, tile_size=224)

        batch_imgs = tumor_imgs+normal_imgs
        labels = [1]*num_tiles + [0]*num_tiles 
        
        
        if self.transforms is not None:
            for idx, img in enumerate(batch_imgs):
                batch_imgs[idx] = self.transforms(batch_imgs[idx])
        
        return torch.stack(batch_imgs), labels


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5, use_gpu=True):
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
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                labels = torch.stack(labels, 0)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            
                inputs, labels = torch.squeeze(inputs), torch.squeeze(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                # for nets that have multiple outputs such as inception
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o,labels) for o in outputs))
                else:
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
#                     _, preds = torch.max(outputs[0].data, 1)
                    _, preds = torch.max(outputs.data, 1)
                    loss.backward()
                    optimizer.step()
                else:
                    _, preds = torch.max(outputs.data, 1)

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                del loss, outputs # Don't know why we need to do this, but some kind of memory leak

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model