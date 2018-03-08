from WSI_utils import*
import argparse

def main(args):
    import os
    import sys
    import glob
    import random
    import pickle
    import numpy as np
    from PIL import Image

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torchvision.transforms as transforms
    import torch.utils.data
    import torchvision.models as models

    def create_img_features(img_folder, imsize=224, cuda=True):
        resnet50 = models.resnet50(pretrained=True)
        if cuda: 
            resnet50.cuda()
        resnet50_feat = nn.Sequential(*list(resnet50.children())[:-1])
        
        all_images = glob.glob(os.path.join(img_folder, '*'))
        all_features = []

        def load_image_pytorch(image_loc, imsize, cuda):
            """load image, returns cuda tensor"""
            image = np.asarray(Image.open(image_loc))
            image = Image.fromarray(image[:,:,:3]) # throw away the transparency channel if it exists
            loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
            image = loader(image).float()
            image = Variable(image, requires_grad=False)
            image = image.unsqueeze(0) # add batch dim
            if cuda:
                image = image.cuda()
            return image

        for image_loc in all_images:
            image = load_image_pytorch(image_loc, imsize, cuda)
            feat = resnet50_feat(image)
            feat = feat.cpu().data.numpy()
            all_features.append(feat)

        features=np.squeeze(np.array(all_features))
        return features

    # Look through directories to get test/train/valid, the slide class, and all the individual WSIs and their tiles. 
    ttv_dirs = glob.glob(os.path.join(args.data_loc, '*'))
    for ttv_dir in ttv_dirs:
        ttv = ttv_dir.rsplit('/', 1)[-1]
        slide_classes = glob.glob(os.path.join(ttv_dir, '*'))
        for slide_class_dir in slide_classes:
            print(slide_class_dir)
            slide_class = slide_class_dir.rsplit('/', 1)[-1]
            wsi_folders = glob.glob(os.path.join(slide_class_dir, '*'))
            for img_folder in wsi_folders:
                img_name = img_folder.rsplit('/', 1)[-1]
                print(img_folder)
                # create all the features for a given folder
                features = create_img_features(img_folder, imsize=args.imsize, cuda=args.cuda)
                save_dir = os.path.join(args.out_loc, ttv, slide_class)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(os.path.join(save_dir, img_name+'.npy'), features)
                print(features.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract resnet50 features from tiles')
    parser.add_argument('--data_loc', type=str)
    parser.add_argument('--out_loc', type=str)
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--cuda', type=int, default=True)
    args = parser.parse_args()

    main(args)