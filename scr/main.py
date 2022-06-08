import pytorch_lightning as pl
from system import AllergyClassifier as classifier

def main():
    print(classifier)
    trainer = pl.Trainer()
    # trainer.fit(classifier, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
