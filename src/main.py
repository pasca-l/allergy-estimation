import pytorch_lightning as pl

from models import AllergyClassifierModel
from system import AllergyClassifier
from datasets import FoodDataModule, FoodDataset


def main():
    dataset = FoodDataModule(
                data_dir='../food-101/images/',
                ann_dir='../food-101/meta/')
    model = AllergyClassifierModel()
    classifier = AllergyClassifier(model)

    trainer = pl.Trainer(default_root_dir='../logs/', max_epochs=1)
    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
