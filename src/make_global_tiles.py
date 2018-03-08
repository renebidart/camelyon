from WSI_utils import*
import argparse

def main(args):
    import os
    import sys
    import glob
    import random
    import numpy as np
    import pandas as pd
    from PIL import Image
    from skimage import color
    from skimage import filters
    from skimage.morphology import disk
    from openslide import OpenSlide, OpenSlideUnsupportedFormatError

    SEED = 101
    average_tiles = 27098

    # create validation set. Randomly sample 20% of tumor and non-tumor training set
    np.random.seed(SEED)
    wsi_locs = glob.glob(os.path.join(args.data_loc, '*'))
    num = len(wsi_locs)
    vaild_idx = np.random.choice(int(num), int(np.round(num*args.valid_frac)))

    # Make the tiles
    for loc in wsi_locs:
        # check if in validation set
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        if wsi_id in vaild_idx:
            ttv = 'valid'
        else:
            ttv = 'train'

        # Get the slide class
        if 'Test' in loc:
            wsi_type = 'unknown'
            ttv = 'test'
        elif 'Normal' in loc:
            wsi_type = 'normal'
        elif 'Tumor' in loc:
            wsi_type = 'tumor'
        else: 
            print('Error')

        # now read in and get the samples:
        wsi = WSI(loc)
        wsi.generate_mask(mask_level=args.mask_level)
        total_tiles = wsi.est_total_tiles(tile_size = args.tile_size)
        num_tiles = np.amin([total_tiles, average_tiles/2])
        num_tiles = int(np.round(num_tiles/args.sample_reduction_factor))
        print('Sampling ', num_tiles, ' from ', loc)
        
        # Make folders for normal, tumor. Save each set of samples from a wsi in a folder within these.
        out_wsi_loc = os.path.join(args.out_loc, ttv, wsi_type, wsi.wsi_name)
        if not os.path.exists(out_wsi_loc):
            os.makedirs(out_wsi_loc)

        # Now make the tiles
        wsi.sample_from_wsi(out_wsi_loc, num_tiles, args.tile_size, normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make tiles from WSI')
    parser.add_argument('--data_loc', type=str)
    parser.add_argument('--out_loc', type=str)
    parser.add_argument('--sample_reduction_factor', type=int, default=1)
    parser.add_argument('--tile_size', type=int, default=224)
    parser.add_argument('--valid_frac', type=int, default=.2)
    parser.add_argument('--mask_level', type=int, default=6)
    args = parser.parse_args()

    main(args)