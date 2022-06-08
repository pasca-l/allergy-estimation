import torch
import pytorch_lightning as pl
from models import AllergyClassifierModelFineTune as model

class AllergyClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_arch = model.arch
