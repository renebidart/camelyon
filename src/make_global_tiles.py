from WSI_utils import*

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
    mask_level=6

    # create validation set. Randomly sample 20% of tumor and non-tumor training set
    np.random.seed(SEED)
    wsi_locs = glob.glob(os.path.join(data_folder, '/*'))
    num = len(wsi_locs)
    vaild_idx = np.random.choice(num, np.round(num*valid_frac))

    # Make the tiles
    for loc in all_locs[0:1]:
        # Get the slide class
        if 'Test' in loc:
            wsi_type = 'unknown'
            wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
            ttv = 'test'
        elif 'Normal' in loc:
            wsi_type = 'normal'
            wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        elif 'Tumor' in loc:
            wsi_type = 'tumor'
            wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        else: 
            print('Error')

        # check if in validation set
        if wsi_id in vaild__idx:
            ttv = 'valid'
        else:
            ttv = 'train'

        # now read in and get the samples:
        wsi = WSI(loc)
        wsi.generate_mask(mask_level=mask_level)
        total_tiles = wsi.est_total_tiles(tile_size = tile_size)
        num_tiles = np.amin([total_tiles, average_tiles/2])
        
        # Make folders for normal, tumor. Save each set of samples from a wsi in a folder within these.
        out_dir = os.path.join(base_out_dir, ttv, wsi_type, wsi.wsi_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Now make the tiles
        wsi.make_tiles(out_dir, num_tiles, tile_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorithm on Learning Rate Schedule')
    parser.add_argument('--data_loc', type=str)
    parser.add_argument('--out_loc', type=str)
    parser.add_argument('--test_set', type=int, default=False)
    parser.add_argument('--tile_size', type=int, default=224)
    parser.add_argument('--valid_frac', type=int, default=.2)
    args = parser.parse_args()

    main(args)