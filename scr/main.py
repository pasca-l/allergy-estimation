import pytorch_lightning as pl

from models import AllergyClassifierFinetuneModel
from system import AllergyClassifier
from datasets import FoodDataModule


def main():
    dataset = FoodDataModule()
    model = AllergyClassifierFinetuneModel()
    classifier = AllergyClassifier(model)

    trainer = pl.Trainer()
    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
