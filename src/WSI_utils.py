import os
import sys
import os.path
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage import color
from skimage import filters
from skimage.morphology import disk
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models as models
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler


class WSI(object):
    """
    Not sure how to do this. Should have the mask associated with this, or returned?
    I will leave these associated with it because there should only be one for each instance:
    wsi, mask, mask level, 

    Where to use self. vs. not using it in general?
    """
    def __init__(self, wsi_path):
        self.wsi_path = wsi_path
        self.wsi = OpenSlide(wsi_path)
        self.wsi_name = self.wsi_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]

    def generate_mask(self, mask_level=5, disk_radius=False):
        """Generate mask indicating tissue regions for a WSI. Otsu's method on saturation, optionally smoothing
        
        Args:
            mask_level: level to downsample to create the mask
            disk_radius: set this to an int (e.g. 10) if want this smoothing method
        """

        self.mask_level = mask_level
        img = self.wsi.read_region(location=(0, 0), level=mask_level, size=self.wsi.level_dimensions[mask_level]).convert('HSV')
        # otsu's on the saturation:
        img_saturation = np.asarray(img)[:, :, 1]
        threshold = filters.threshold_otsu(img_saturation)
        high_saturation = (img_saturation > threshold)
        # otsu's on the hue:
        img_hue = np.asarray(img)[:, :, 0]
        threshold = filters.threshold_otsu(img_hue)
        high_hue = (img_hue > threshold)

        mask = high_saturation | high_hue
        
        # optionally use the disk method (sensible radius was 10)
        if disk_radius!=False:
            disk_object = disk(disk_radius)
            self.mask = closing(mask, disk_object) # ? correct ? never used it
            self.mask = opening(mask, disk_object)
        else: 
            self.mask = mask

    def est_total_tiles(self, tile_size = 224):
        """ Estimate the number of tiles in a mask. Not exact method, just by dividing mask area by size of tile
        """
        num_patches = np.sum(self.mask)
        total_pixels = num_patches*self.wsi.level_downsamples[self.mask_level]**2
        total_tiles = np.round(total_pixels/tile_size**2)
        return total_tiles


    def sample_from_wsi(self, out_dir, num_tiles, tile_size=224, normalize=False, tile_sample_level=0):
        """Sample from a WSI. Ignore any annotation file if it exits"""
        self.mask_level=5 
        
        # create the mask
        self.generate_mask(mask_level=self.mask_level)
        patch_size = np.round(self.mask_level/float(self.wsi.level_downsamples[tile_sample_level]))
        
        # randomly select a part in the mask
        all_indices = np.asarray(np.where(self.mask))

        curr_samples = 0
        while(curr_samples < num_tiles):
            idx = np.random.randint(0, len(all_indices[0]))
            sample_patch_ind = np.array([all_indices[1][idx], all_indices[0][idx]]) # not sure why this is backwards

            # convert to coordinates of level: tile_sample_level
            sample_patch_ind = np.round(sample_patch_ind*
                self.wsi.level_downsamples[self.mask_level]/float(self.wsi.level_downsamples[tile_sample_level]))

            # random point inside this patch for center of new image (sampled image could extend outside patch or mask)
            location = (np.random.randint(sample_patch_ind[0]-tile_size/2, sample_patch_ind[0]+tile_size/2),
                        np.random.randint(sample_patch_ind[1]-tile_size/2, sample_patch_ind[1]+tile_size/2))

            # adjust coordinates toward center by amount of half the number of pixels in the mask level at the downsample ratio
            # The above location is only correct is the number of pixels in mask pixel is the same as 224

            try:
                img = self.wsi.read_region(location=location, level=0, size=(tile_size, tile_size)).convert('RGB')
            except Exception as e: 
                print(e)
                continue # if exception try sampling a new location.
            curr_samples+=1

            if normalize:
                img = img-np.amin(img) # left shift to 0
                img = (img/np.amax(img))*255 # scale to [0, 255]
                img = Image.fromarray(img.astype('uint8'))
            
            out_file = os.path.join(out_dir, self.wsi_name +'_'+ str(curr_samples)+'.png')
            img.save(out_file, 'PNG')


    def sample_from_tumor_region(self, out_dir, num_tiles, tile_size=224, tile_sample_level=0):
        """Sample from the tumor region in a WSI Do this at level 6 """
        self.tumor_mask_level = 5

        base_dir = self.wsi_path.rsplit('TrainingData', 1)[:-1][0]
        annotation_file_name = self.wsi_path.rsplit('/', 1)[-1].replace(".tif", "_Mask.tif").replace("tumor", "Tumor")
        self.annotation_path = os.path.join(base_dir, 'TrainingData', 'Ground_Truth', 'Mask', annotation_file_name)
        
        self.tumor_annotation_wsi = OpenSlide(self.annotation_path)
        self.tumor_mask = self.tumor_annotation_wsi.read_region(location=(0, 0), level=self.tumor_mask_level, 
            size=self.tumor_annotation_wsi.level_dimensions[self.tumor_mask_level]).convert('RGB')
        
        patch_size = np.round(self.tumor_mask_level/float(self.tumor_annotation_wsi.level_downsamples[tile_sample_level]))

        # locations there there is tumor:
        all_indices = np.asarray(np.where(np.asarray(self.tumor_mask)==255))

        curr_samples = 0
        while(curr_samples < num_tiles):
            # randomly select a part in the tumor mask at a higher level (tumor_mask_level)
            idx = np.random.randint(0, len(all_indices[0]))
            sample_patch_ind = np.array([all_indices[1][idx], all_indices[0][idx]]) # backwards becaus of numnpy vs. pil ordering

            # convert to coordinates of level: tile_sample_level
            sample_patch_ind = np.round(sample_patch_ind*self.tumor_annotation_wsi.level_downsamples[self.tumor_mask_level]
                /float(self.tumor_annotation_wsi.level_downsamples[tile_sample_level]))

            # random point inside this patch for center of new image (sampled image could extend outside tumor region)
            location = (np.random.randint(sample_patch_ind[0]-tile_size/2, sample_patch_ind[0]+tile_size/2),
                        np.random.randint(sample_patch_ind[1]-tile_size/2, sample_patch_ind[1]+tile_size/2))

            try:
                img = self.wsi.read_region(location=location, level=0, size=(tile_size, tile_size)).convert('RGB')
            except Exception as e: 
                print(e)
                continue # if exception try sampling a new location.
            curr_samples+=1

            out_file = os.path.join(out_dir, self.wsi_name +'_tumor_'+ str(curr_samples)+'.png')
            img.save(out_file, 'PNG')


    def sample_from_normal_region(self, out_dir, num_tiles, tile_size=224, tile_sample_level=0):
        """Sample from the tumor region in a WSI Do this at level 6 """
        self.mask_level = 5 # use this because it is the lowest resolution that img and annotation are almost same dimension
        self.tumor_mask_level = self.mask_level

        base_dir = self.wsi_path.rsplit('TrainingData', 1)[:-1][0]
        annotation_file_name = self.wsi_path.rsplit('/', 1)[-1].replace(".tif", "_Mask.tif").replace("tumor", "Tumor")
        self.annotation_path = os.path.join(base_dir, 'TrainingData', 'Ground_Truth', 'Mask', annotation_file_name)

        # create the tissue mask
        self.generate_mask(mask_level=self.mask_level)

        patch_size = np.round(self.mask_level/float(self.wsi.level_downsamples[tile_sample_level]))
        
        # generate the tumor mask
        self.tumor_annotation_wsi = OpenSlide(self.annotation_path)
        self.tumor_mask = np.asarray(self.tumor_annotation_wsi.read_region(location=(0, 0), level=self.tumor_mask_level, 
            size=self.tumor_annotation_wsi.level_dimensions[self.tumor_mask_level]).convert('RGB'))[:, :, 0]

        # Combine the masks, getting where no tumor and inside tissue region. First make sure they are the same size:
        annotation_size = self.tumor_annotation_wsi.level_dimensions[self.tumor_mask_level]
        
        pil_mask = Image.fromarray(self.mask.astype('uint8'))
        self.mask = pil_mask.resize((annotation_size), Image.ANTIALIAS)
        all_indices = np.asarray(np.where((self.tumor_mask!=255)& self.mask))
        
        patch_size = np.round(self.tumor_mask_level/float(self.tumor_annotation_wsi.level_downsamples[tile_sample_level]))

        curr_samples = 0
        while(curr_samples < num_tiles):
            # randomly select a part in the tumor mask at a higher level (tumor_mask_level)
            idx = np.random.randint(0, len(all_indices[0]))
            sample_patch_ind = np.array([all_indices[1][idx], all_indices[0][idx]]) # not sure why this is backwards

            # convert to coordinates of level: tile_sample_level
            sample_patch_ind = np.round(sample_patch_ind*self.tumor_annotation_wsi.level_downsamples[self.tumor_mask_level]
                /float(self.tumor_annotation_wsi.level_downsamples[tile_sample_level]))

            # random point inside this patch for center of new image (sampled image could extend outside tumor region)
            location = (np.random.randint(sample_patch_ind[0]-tile_size/2, sample_patch_ind[0]+tile_size/2),
                        np.random.randint(sample_patch_ind[1]-tile_size/2, sample_patch_ind[1]+tile_size/2))

            try:
                img = self.wsi.read_region(location=location, level=0, size=(tile_size, tile_size)).convert('RGB')
            except Exception as e: 
                print(e)
                continue # if exception try sampling a new location.
            curr_samples+=1

            out_file = os.path.join(out_dir, self.wsi_name +'_normal_'+ str(curr_samples)+'.png')
            img.save(out_file, 'PNG')


    def make_tiles_by_class(self, out_dir, num_tiles, tile_class='normal', tile_size=224, tile_sample_level=0):
        """ Sample tiles randomly within the a given image, in the given class. Assume running on GPU
        
        Args:
            out_dir: where to save output
            num_tiles: number of tiles to create
            tile_class: 'normal' or 'tumor'
            tile_size: Must be in terms of pixels at the given level you are sampling

         **** Note PIL uses dim as width, height. Numpy does it as row (height), col (width.). **** 
        """

        # check if there is a tumor in the slide:
        base_dir = self.wsi_path.rsplit('TrainingData', 1)[:-1][0]
        annotation_file_name = self.wsi_path.rsplit('/', 1)[-1].replace(".tif", "_Mask.tif").replace("tumor", "Tumor")
        self.annotation_path = os.path.join(base_dir, 'TrainingData', 'Ground_Truth', 'Mask', annotation_file_name)

        if os.path.isfile(self.annotation_path):
            if(tile_class =='tumor'):
                self.sample_from_tumor_region(out_dir, num_tiles, tile_sample_level=tile_sample_level, tile_size=tile_size)
            elif(tile_class =='normal'):
                self.sample_from_normal_region(out_dir, num_tiles, tile_sample_level=tile_sample_level, tile_size=tile_size)
            else:
                print('Error, invalid tile_class')
                return

        else: # no annotation found, so wsi is normal
            if(tile_class =='normal'):
                self.sample_from_wsi(out_dir, num_tiles, tile_size, normalize=False, tile_sample_level=tile_sample_level)
            else:
                print("This is normal WSI, can't sample tumor from it")
                print(annotation_file_name)
                print(self.annotation_path)
                return


    def make_heatmap_simple(self, model, batch_size=16, tile_sample_level=0, patch_size=299):
        """Generate heatmap. Use mask level of 8, so that one mask pixel is a little smaller than 299x299

        Args:
            model: pre-trained pytorch model
            batch_size: do inference in batches to speed it up
            tile_sample_level: The level we will be taking the samples from to feed to neural net
            patch_size: size of the patch (square) that we are taking, in terms of pixels at tile_sample_level

        Not sure how this is done in the paper, but this will make a heatmap with slight overlap between tiles.
        """

        self.mask_level=8

        # I don't know why some don't have the high magnification factor
        try:
            self.generate_mask(mask_level=self.mask_level)
        except Exception: 
            self.mask_level=7
            print('using mask level: ', self.mask_level)
            self.generate_mask(mask_level=self.mask_level)

        patch_size = int(patch_size)
        tile_sample_level = int(tile_sample_level)
        
        pixel_size = np.round(float(self.wsi.level_downsamples[self.mask_level]/self.wsi.level_downsamples[tile_sample_level]))
        all_indices = np.asarray(np.where(self.mask))

        num_batches = int(np.ceil(len(all_indices[0])/batch_size))

        heatmap = np.zeros((self.mask.shape[0], self.mask.shape[1], 2))

        print('len(all_indices[0])', len(all_indices[0]))
        print('heatmap.shape', heatmap.shape)
        print('num_batches', num_batches)

        for batch in tqdm(range(num_batches)):
            batch_images = []

            # predict for all images in the batch
            for idx in range(batch_size*batch, batch_size*(batch+1)):
                try:
                    sample_patch_ind = np.array([all_indices[1][idx], all_indices[0][idx]])
                except Exception as e: 
                    print(e)
                    continue # hopefully exception is just the last batch

                # convert to coordinates of level: tile_sample_level
                sample_patch_ind = np.round(sample_patch_ind*
                    self.wsi.level_downsamples[self.mask_level]/float(self.wsi.level_downsamples[tile_sample_level]))

                # want center the patch on the pixel on the heatmap. Coordinates are in terms of top left
                location = (int(sample_patch_ind[0] - (patch_size - pixel_size)/2),
                            int(sample_patch_ind[1] - (patch_size - pixel_size)/2))
                try:
                    img = self.wsi.read_region(location=location, level=tile_sample_level, size=(patch_size, patch_size)).convert('RGB')
                except Exception as e: 
                    print(e)
                    print('Not able to read in location: ', location)
                    continue

                img = np.asarray(img)
                img = np.swapaxes(img,0, 2)
                batch_images.append(img)

            batch_images = torch.from_numpy(np.array(batch_images)).type(torch.cuda.FloatTensor)
            batch_images = Variable(batch_images)

            batch_output = model(batch_images)

            # add the predictions in the batch to the heatmap
            for idx, loc in enumerate(all_indices[batch_size*batch:batch_size*(batch+1)]):
                heatmap[loc[0], loc[1], :] = batch_output[idx]

        return heatmap




########## Other basic stuff:

def get_average_tiles(train_folder):
    normal_locs = glob.glob(os.path.join(train_folder, 'Train_Normal/*'))
    tumor_locs = glob.glob(os.path.join(train_folder, 'Train_Tumor/*'))
    all_locs = normal_locs+tumor_locs
    tile_num_list = []

    for loc in all_locs:
        wsi = WSI(loc)
        wsi.generate_mask(mask_level=6)
        num_tiles = wsi.est_total_tiles(tile_size = 224)
        tile_num_list.append(num_tiles)
        
    average_tiles = np.average(np.array(tile_num_list))
    return np.round(average_tiles)


