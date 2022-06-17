import pytorch_lightning as pl
import torch
import cv2
from src.models import AllergyClassifierModel
from src.system import AllergyClassifier
from src.datasets import FoodDataModule

# from app_ui import hoge

# temp and 
def hoge():
    img_path = "../img/sample_food.jpeg"
    img = cv2.imread(img_path)
    return img

def prediction(img = None):
    dataset = FoodDataModule(
        data_dir='../food-101/images/',
        ann_dir='../food-101/meta/',
        class_file='../food-101/meta/classes.txt',
        batch_size=16
    )
    model = AllergyClassifierModel()
    classifier = AllergyClassifier(model)

    logger = pl.loggers.TensorBoardLogger(
        save_dir='../logs/',
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='../logs/',
        save_weights_only=True,
        save_top_k=1
    )
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        auto_select_gpus=True,
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    ckpt = "../ckpt/epoch=0-step=1263.ckpt"
    if img == None:
        img = hoge()
    
    output = trainer.predict(model = classifier, ckpt_path = ckpt, return_predictions = True)
    # I don't know how to input image

    return output
    # return possible_foods_dict possible_allergen_dict