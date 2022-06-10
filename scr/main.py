import pytorch_lightning as pl

from models import AllergyClassifierFinetuneModel
from system import AllergyClassifier
from datasets import FoodDataModule


def main():
    # dataset = FoodDataModule(
    #             data_dir='../food-101/images/',
    #             ann_dir='../food-101/meta/')
    # model = AllergyClassifierFinetuneModel()
    # classifier = AllergyClassifier(model)
    #
    # trainer = pl.Trainer(max_epochs=1)
    # trainer.fit(classifier, dataset)

    model = AllergyClassifierFinetuneModel()
    print(model.parameters())


if __name__ == '__main__':
    main()
