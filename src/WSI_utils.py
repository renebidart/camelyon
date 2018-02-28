import os
import sys
import numpy as np
from PIL import Image
from skimage import color
from skimage import filters
from skimage.morphology import disk
from openslide import OpenSlide, OpenSlideUnsupportedFormatError


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

    def generate_mask(self, mask_level=6, disk_radius=False):
        """Generate mask indicating tissue regions for a WSI. Otsu's method on saturation, optionally smoothing
        
        Args:
            mask_level: level to downsample to create the mask
            disk_radius: set this to an int (e.g. 10) if want this smoothing method
        """
        self.mask_level = mask_level
        img = self.wsi.read_region(location=(0, 0), level=mask_level, size=self.wsi.level_dimensions[mask_level]).convert('HSV')
        img_saturation = np.asarray(img)[:, :, 1]
        threshold = filters.threshold_otsu(img_saturation)
        high_saturation = (img_saturation > threshold)
        
        # optionally use the disk method (sensible radius was 10)
        if disk_radius!=False:
            disk_object = disk(disk_radius)
            self.mask = closing(high_saturation, disk_object)
            self.mask = opening(mask, disk_object)
        else: 
            self.mask = high_saturation

    def est_total_tiles(self, tile_size = 224):
        """ Estimate the number of tiles in a mask. Not exact method, just by dividing mask area by size of tile
        """
        num_patches = np.sum(self.mask)
        total_pixels = num_patches*self.wsi.level_downsamples[self.mask_level]**2
        total_tiles = np.round(total_pixels/tile_size**2)
        return total_tiles
    

    def make_tiles(self, out_dir, num_tiles, tile_size=224, normalize=True):
        """ Sample tiles randomly within the mask.
        
        Args:
            out_dir: where to save output
            num_tiles: number of tiles to create
        """

        patch_size = np.round(self.mask_level) # size of each pixel in mask in terms of level 0
        curr_samples = 0
        while(curr_samples < num_tiles):
            # randomly select a pixel in the mask to sample from
            all_indices = np.asarray(np.where(self.mask))
            idx = np.random.randint(0, len(all_indices[0]))
            sample_patch_ind = np.array([all_indices[1][idx], all_indices[0][idx]]) # not sure why this is backwards
            # convert to coordinates of level 0
            sample_patch_ind = np.round(sample_patch_ind*self.wsi.level_downsamples[self.mask_level])
            # random point inside this patch for center of new image (sampled image could extend outside patch or mask)
            location = (np.random.randint(sample_patch_ind[0]-tile_size/2, sample_patch_ind[0]+tile_size/2),
                        np.random.randint(sample_patch_ind[1]-tile_size/2, sample_patch_ind[1]+tile_size/2))
            try:
                img = wsi.read_region(location=location, level=0, size=(tile_size, tile_size))
            except:
                continue # if exception try sampling a new location.
            curr_samples+=1

            if normalize:
                img = img-np.amin(img) # left shift to 0
                img = (img/np.amax(img))*255 # scale to [0, 255]
                img = Image.fromarray(img.astype('uint8'))
            
            out_file = os.path.join(out_dir, self.wsi_name +'_'+ str(curr_samples))
            img.save(out_file, 'PNG')


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


