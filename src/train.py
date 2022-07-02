import pytorch_lightning as pl

from models import AllergyClassifierModel
from system import AllergyClassifier
from datasets import FoodDataModule


def main():
    data_dir_path = '../../../datasets/food-101/images/'
    ann_dir_path = '../../../datasets/food-101/meta/'
    class_file_path = '../../../datasets/food-101/meta/classes.txt'

    dataset = FoodDataModule(
        data_dir=data_dir_path,
        ann_dir=ann_dir_path,
        class_file=class_file_path,
        batch_size=8
    )
    model = AllergyClassifierModel()
    classifier = AllergyClassifier(model)

    logger = pl.loggers.TensorBoardLogger(
        save_dir='../logs/'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        dirpath='../logs/',
        filename="imagecls_{epoch:02d}-{val_loss:.2f}"
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10
    )
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        auto_select_gpus=True,
        max_epochs=30,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping]
    )

    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()