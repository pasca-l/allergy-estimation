import pytorch_lightning as pl

from models import AllergyClassifierModel
from system import AllergyClassifier
from datasets import FoodDataModule

# https://pytorch.org/vision/stable/models.html
import torchvision.models as models


def main():
    dataset = FoodDataModule(
                data_dir='../food-101/images/',
                ann_dir='../food-101/meta/')
    model = models.convnext_large(pretrained=True)
    classifier = AllergyClassifier(model)

    # trainer = pl.Trainer(max_epochs=1)
    # trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
