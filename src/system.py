import torch
import torch.optim as optim
import torch.nn.functional as nnf
import pytorch_lightning as pl
import torchmetrics


class AllergyClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nnf.cross_entropy(logits, y)

        self.log("train_loss", loss)
        self.train_acc(torch.argmax(logits, dim=1), y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nnf.cross_entropy(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_acc(torch.argmax(logits, dim=1), y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
