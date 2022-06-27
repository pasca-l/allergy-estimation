import torch

from models import AllergyClassifierModel
from system import AllergyClassifier


def main():
    model = AllergyClassifierModel(
        weight_file='../food-101/meta/weights.csv'
    )
    classifier = AllergyClassifier(model)

    ckpt = torch.load("../logs/epoch=0-step=1263-v1.ckpt")
    classifier.load_state_dict(ckpt['state_dict'])
    classifier.eval()

    # データを入れる部分
    # --------------
    from datasets import FoodDataModule
    dataset = FoodDataModule(
        data_dir='../food-101/images/',
        ann_dir='../food-101/meta/',
        class_file='../food-101/meta/classes.txt',
        weight_file='../data/meta/weights.csv',
        batch_size=16
    )
    dataset.setup()
    img, label = dataset.train_dataloader().__iter__().next()[0][0], dataset.train_dataloader().__iter__().next()[1][0]
    # ----------------

    with torch.no_grad():
        output = classifier(img.unsqueeze(0))
    print(output.argmax(), label)
    print(output)


if __name__ == '__main__':
    main()