from WSI_utils import*
import argparse

def main(args):
    """ Make a tile-level test set out of the test dataset"""
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
    test_loc = os.path.join(args.data_loc, 'Testset', 'Images')
    test_wsi_locs = glob.glob(os.path.join(test_loc, '*'))

    # Check if there is a tumor or not
    test_mask_loc = os.path.join(args.data_loc, 'Testset', 'Ground_Truth', 'Masks')
    test_mask_wsi_locs = glob.glob(os.path.join(test_mask_loc, '*'))
    test_tumor_idx = [int(loc.rsplit('/', 1)[-1].rsplit('_', 1)[0].rsplit('_', 1)[-1]) for loc in test_mask_wsi_locs]
    num_all = len(test_wsi_locs)
    num_tumor = len(test_tumor_idx)
    print('Num test: ', num_all, 'Num Tumor: ', num_tumor)


    # sample from all the slides (only normal samples)
    normal_tiles_per_wsi = int(np.round((1-args.tumor_frac)*args.num_samples/num_all))

    # sample from all the tumor slides, both normal and tumor
    normal_tumor_tiles_per_wsi = int(np.round((1-args.tumor_frac)*args.num_samples/num_all))
    tumor_tiles_per_wsi = int(np.round(args.tumor_frac*args.num_samples/num_tumor))

    for idx, loc in enumerate(test_wsi_locs):
        print(loc)
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        ttv='test'

        if wsi_id in test_tumor_idx:
            print('tumor wsi')
            wsi = WSI(loc)
            out_loc = os.path.join(args.out_dir, ttv, 'normal')
            if not os.path.exists(out_loc):
                os.makedirs(out_loc)
            wsi.make_test_tiles_by_class(out_loc, num_tiles=normal_tumor_tiles_per_wsi, 
                tile_class='normal', tile_size=args.tile_size, tile_sample_level=0, strict=args.strict)

            out_loc = os.path.join(args.out_dir, ttv, 'tumor')
            if not os.path.exists(out_loc):
                os.makedirs(out_loc)
            wsi.make_test_tiles_by_class(out_loc, num_tiles=tumor_tiles_per_wsi, 
                tile_class='tumor', tile_size=args.tile_size, tile_sample_level=0, strict=args.strict)
        else: 
            wsi = WSI(loc)
            out_loc = os.path.join(args.out_dir, ttv, 'normal')
            if not os.path.exists(out_loc):
                os.makedirs(out_loc)

            wsi.make_test_tiles_by_class(out_loc, num_tiles=normal_tiles_per_wsi, tile_class='normal', 
                tile_size=args.tile_size, tile_sample_level=0, strict=args.strict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make training tiles from WSI by local annotation')
    parser.add_argument('--data_loc', type=str) 
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--tumor_frac', type=float, default=.5)
    parser.add_argument('--tile_size', type=int, default=299)
    parser.add_argument('--valid_frac', type=int, default=.2)
    parser.add_argument('--mask_level', type=int, default=5)
    parser.add_argument('--strict', dest='strict', action='store_true')
    args = parser.parse_args()

    main(args)