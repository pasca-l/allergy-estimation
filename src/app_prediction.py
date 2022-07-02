import numpy as np
import torch
from torchvision import transforms

from datasets import FoodImageTransform
from models import AllergyClassifierModel
from system import AllergyClassifier


class Predictor():
    def __init__(self,
        weight_file='../data/meta/weights.csv',
        ckpt_file='../logs/model.ckpt'
    ):
        self.model = AllergyClassifierModel()
        self.classifier = AllergyClassifier(self.model)
        if torch.cuda.is_available():
            self.ckpt = torch.load(ckpt_file)
        else:
            self.ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
        self.classifier.load_state_dict(self.ckpt['state_dict'])
        self.classifier.eval()

        self.transform = FoodImageTransform()
        self.allergen_list = np.loadtxt(weight_file, delimiter=',', 
                                        dtype='unicode')[:1,1:][0]
        self.food_list = np.loadtxt(weight_file, delimiter=',', skiprows=1,
                                    usecols=[0], dtype='unicode')
        self.weights = np.loadtxt(weight_file, delimiter=',', skiprows=1, 
                                  usecols=range(1, 28), dtype='float32')

    def predict(self, img):
        img = self.transform(img)

        with torch.no_grad():
            food_prob = self.classifier(img.unsqueeze(0))

        ordered_food_name = self.food_list[np.argsort(food_prob)]
        ordered_food_prob = np.sort(food_prob)

        allergen_prob = torch.mm(food_prob, torch.tensor(self.weights))
        ordered_allergen_name = self.allergen_list[np.argsort(allergen_prob)]
        ordered_allergen_prob = np.sort(allergen_prob)

        return (ordered_food_name[:,:,-1][0], ordered_allergen_name[:,:,-1],
                ordered_allergen_prob[:,:,-1])
