import argparse
from WSI_utils import*

def main(args):
    import os
    import os.path
    import sys
    import glob
    import random
    import pickle
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
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

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NUMBER

    # load ttv split
    ttv_split = pickle.load(open( args.ttv_split_loc, "rb" ))

    # define the model 
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # load the model
    model.load_state_dict(torch.load(args.model_loc))
    model.eval()
    model.cuda()

    # get all the WSIs:
    all_wsi_locs = glob.glob(args.data_loc+'/**/*.tif', recursive=True)

    avoid_list = ['mask']
    all_wsi_locs = [loc for loc in all_wsi_locs if not any(x in loc.lower() for x in avoid_list)]

    # for each WSI, make the heatmap
    for loc in tqdm(all_wsi_locs):
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        ttv = 'train'
        if 'normal' in loc.lower():
            slide_class = 'normal'
            if wsi_id in ttv_split['normal_vaild_idx']:
                ttv = 'valid'
        elif 'tumor' in loc.lower():
            slide_class = 'tumor'
            if wsi_id in ttv_split['tumor_vaild_idx']:
                ttv = 'valid'
        elif 'test' in loc.lower():
            slide_class = 'test'
            ttv = 'test'
        else:
            print("--------  Invalid slide class ------")

        outfile = os.path.join(args.out_loc, ttv, slide_class)
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        heatmap_loc = outfile+str(wsi_id)+'heatmap.npy'

        # check if the file was already made:
        if os.path.isfile(heatmap_loc):
            print('Already created: ', loc)
            continue
        else:
            print('Creating heatmap: ', loc)
            wsi = WSI(loc)
            heatmap = wsi.make_heatmap_simple(model, batch_size=args.batch_size, tile_sample_level=0, patch_size=299)
            np.save(heatmap_loc, heatmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make heatmaps')
    parser.add_argument('--data_loc', type=str) 
    parser.add_argument('--out_loc', type=str)
    parser.add_argument('--model_loc', type=str)
    parser.add_argument('--ttv_split_loc', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--GPU_NUMBER', type=str, default='1')

    args = parser.parse_args()

    main(args)