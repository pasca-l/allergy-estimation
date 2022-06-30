import torch
import pickle

# ckpt1 = torch.load("logs/epoch=0-step=1263-v1.ckpt")
with open("logs/epoch=0-step=1263-v1.pickle", "rb") as f:
    ckpt2 = pickle.load(f)