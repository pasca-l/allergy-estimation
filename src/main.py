import pytorch_lightning as pl

from models import AllergyClassifierModel
from system import AllergyClassifier
from datasets import FoodDataModule


def main():
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

    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
