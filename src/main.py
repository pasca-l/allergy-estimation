import pytorch_lightning as pl

from models import AllergyClassifierModel
from system import AllergyClassifier
from datasets import FoodDataModule


def main():
    dataset = FoodDataModule(
        data_dir='../food-101/images/',
        ann_dir='../food-101/meta/',
        class_file='../food-101/meta/classes.txt',
        weight_file='../food-101/meta/weights.csv',
        batch_size=8
    )
    model = AllergyClassifierModel(
        weight_file='../food-101/meta/weights.csv'
    )
    classifier = AllergyClassifier(model)

    logger = pl.loggers.TensorBoardLogger(
        save_dir='../logs/',
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_weights_only=True,
        monitor='val_loss',
        dirpath='../logs/',
        filename="{epoch:02d}-{val_loss:.2f}"
    )
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        auto_select_gpus=True,
        max_epochs=30,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
