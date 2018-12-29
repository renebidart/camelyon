# Whole Slide Histopathology Classification

Automated detection and classification of WSIs from the [Camelyon](https://camelyon17.grand-challenge.org) dataset in Pytorch, used for class [project](Ensemble_Methods_for_Histopathology_Classification.pdf).

## Batch Generator
WSIs are large images, so standard deep learning classification methods won't work, because it isn't possible to feed a 10 billion pixel image into a CNN. With WSIs the image is broken up into smaller tiles, and a classifier is used to classify these. This classifier can be applied across the entire image, generating a heatmap of cancer probability. These tiles can either be generated within the pytorch batch generator, or created and saved:


* Code for generating and saving a set of tiles is located in [WSI_utils.py](src/WSI_utils.py) and [make_training_tiles.py](src/make_training_tiles.py).

* Alternatively, batches of tiles can be generated ramdomly during training, as in [WSI_pytorch_utils.py](src/WSI_pytorch_utils.py).

## Tile Level Classification
This focuses on the tile-level classification, with some investigation into ensemble methods for classification. Bagging was shown to be [ineffective](notebooks/final/bagging-final.ipynb), and other ensembles were somewhat [more effective](notebooks/final/ensemble-methods.ipynb).

