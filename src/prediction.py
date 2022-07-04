import numpy as np
import torch
import torch.nn.functional as nnf
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
                                        dtype='object')[:1,1:][0]
        self.food_list = np.loadtxt(weight_file, delimiter=',', skiprows=1,
                                    usecols=[0], dtype='object')
        self.weights = np.loadtxt(weight_file, delimiter=',', skiprows=1, 
                                  usecols=range(1, 28), dtype='float32')

    def predict(self, img):
        img = self.transform(img)

        with torch.no_grad():
            food_logits = self.classifier(img.unsqueeze(0))

        ordered_food_name = self.food_list[np.argsort(food_logits)]
        ordered_food_prob = np.sort(nnf.softmax(food_logits))

        allergen_logits = np.dot(food_logits.numpy().copy(), self.weights)
        allergen_prob = nnf.softmax(allergen_logits)
        ordered_allergen_name = self.allergen_list[np.argsort(allergen_prob)]
        ordered_allergen_prob = np.sort(allergen_prob)

        return (
            ordered_food_name[0][::-1],
            ordered_food_prob[0][::-1],
            ordered_allergen_name[0][::-1],
            ordered_allergen_prob[0][::-1]
        )
