# Allergy Estimation
This product aims to find the containing probability of 27 allergens of a food.

## Requirements
- pytorch 1.12.0
- pytorch-lightning 1.6.4
- torchvision 0.13.0
- torchmetrics 0.9.2
- opencv-python 4.5.5.64
- numpy 1.22.3

## Usage
0. Download [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
1. Modify model in `model.py` to desired. (As Food101 dataset is used, make sure to have 101 dimension output.)
2. To train model, modify `train.py` to contain following paths, and run script. Ckpt file will be saved under `logs` directory.
```
data_dir_path = './path/to/data/directory'
ann_dir_path = './path/to/annotation/directory'
class_file_path = './path/to/class.txt/file'
```
3. To start video for inference, modify `app_ui.py` to contain following paths, and run script.
```
weight_file_path = './path/to/weights.csv/file'
ckpt_file_path = './path/to/ckpt/file'
```

## Note
This application uses image classification to find the food class probability of an input image first. Allergen probability is not incorporated in the model (due to no loss improvement in experiments), so the network output is simply dot multiplied with the food-class-to-allergen weight matrix.

In addition, the food-class-to-allergen weight matrix is derived from research impressions on the percentage of a ingredients used for the arrangement of a food class.