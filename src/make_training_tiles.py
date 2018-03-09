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
    np.random.seed(SEED)

    # args.data_loc is location of CAMELYON16 directory
    normal_loc = os.path.join(args.data_loc, 'TrainingData', 'Train_Normal')
    tumor_loc = os.path.join(args.data_loc, 'TrainingData', 'Train_Tumor')
    num_normal = len(glob.glob(os.path.join(normal_loc, '*')))
    num_tumor = len(glob.glob(os.path.join(tumor_loc, '*')))
    num_all = num_normal+num_tumor

    # create validation set. Randomly sample args.valid_frac of tumor and non-tumor training set
    normal_wsi_locs = glob.glob(os.path.join(normal_loc, '*'))
    normal_vaild_idx = np.random.choice(num_normal, int(np.round(num_normal*args.valid_frac)))
    tumor_wsi_locs = glob.glob(os.path.join(tumor_loc, '*'))
    tumor_vaild_idx = np.random.choice(num_tumor, int(np.round(num_tumor*args.valid_frac)))

    # sample from all the normal slides (only normal samples)
    normal_tiles_per_wsi = int(np.round((1-args.tumor_frac)*args.num_samples/num_all))
    for idx, loc in enumerate(normal_wsi_locs):
        print(loc)
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        if wsi_id in normal_vaild_idx:
            ttv = 'valid'
        else: 
            ttv = 'train'
        
        wsi = WSI(loc)
        out_loc = os.path.join(args.out_dir, ttv, 'normal')
        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        wsi.make_tiles_by_class(out_loc, num_tiles=normal_tiles_per_wsi, tile_class='normal', 
            tile_size=args.tile_size, tile_sample_level=0)

    # sample from all the tumor slides, both normal and tumor
    normal_tiles_per_wsi = int(np.round((1-args.tumor_frac)*args.num_samples/num_all))
    tumor_tiles_per_wsi = int(np.round(args.tumor_frac*args.num_samples/num_tumor))

    for idx, loc in enumerate(tumor_wsi_locs):
        print(loc)
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        if wsi_id in tumor_vaild_idx:
            ttv = 'valid'
        else: 
            ttv = 'train'

        wsi = WSI(loc)
        out_loc = os.path.join(args.out_dir, ttv, 'normal')
        if not os.path.exists(out_loc):
            os.makedirs(out_loc)
        wsi.make_tiles_by_class(out_loc, num_tiles=normal_tiles_per_wsi, 
            tile_class='normal', tile_size=args.tile_size, tile_sample_level=0)

        out_loc = os.path.join(args.out_dir, ttv, 'tumor')
        if not os.path.exists(out_loc):
            os.makedirs(out_loc)
        wsi.make_tiles_by_class(out_loc, num_tiles=tumor_tiles_per_wsi, 
            tile_class='tumor', tile_size=args.tile_size, tile_sample_level=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make training tiles from WSI by local annotation')
    parser.add_argument('--data_loc', type=str) 
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--tumor_frac', type=float, default=.5)
    parser.add_argument('--tile_size', type=int, default=299)
    parser.add_argument('--valid_frac', type=int, default=.2)
    parser.add_argument('--mask_level', type=int, default=5)
    args = parser.parse_args()

    main(args)