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

# import pretrainedmodels.utils as utils
from sklearn.metrics import confusion_matrix




def make_batch_gen(PATH, batch_size, num_workers, valid_name='valid', test_name=None, size=224, return_locs=False):
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

    if return_locs: 
        image_datasets = {x: datasets.ImageFolderBad(os.path.join(PATH, x),
                                          data_transforms[x])
                          for x in list(data_transforms.keys())}
    else: 
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



def get_preds(model, dataloader, dataset_size):
    """Return label, prediction (rounded)"""
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


def get_preds_locs(model, dataloader, dataset_size):
    """Terrible"""
    model.train(False)  # Set model to evaluate mode
    model.eval()

    all_labels, all_preds, all_locs = [], []


    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels, locs = data

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        all_labels.extend(labels.data.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_locs.extend(locs)
        del outputs 

    return all_labels, all_preds, all_locs


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



###########.    THERE MUST BE A BETTER WAY THAN COPY PASTING THE WHOLE THING TO CHANGE 1 LINE


import torch.utils.data as data
from PIL import Image
import os
import os.path

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolderBad(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderBad(DatasetFolderBad):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderBad, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
