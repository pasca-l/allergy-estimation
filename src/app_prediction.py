import sys
# sys.path.append("../src")
import torch
import numpy as np
import pandas as pd
import heapq
import cv2
from PIL import Image
from torchvision import transforms

from models import AllergyClassifierModel
from system import AllergyClassifier

class Predictor():
    def __init__(self) -> None:
        self.weight_file = "../food-101/meta/weights.csv"
        self.model = AllergyClassifierModel(
            weight_file=self.weight_file
        )
        self.classifier = AllergyClassifier(self.model)
        self.ckpt = torch.load("../logs/epoch=0-step=1263-v1.ckpt")
        self.classifier.load_state_dict(self.ckpt['state_dict'])
        self.classifier.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        self.table = pd.read_csv(self.weight_file)
        self.allergen_list = list(self.table.columns)[1:]
        self.food_list = list(self.table["table"])
        self.weights = torch.tensor(np.loadtxt(self.weight_file, delimiter=',', skiprows=1, 
                usecols=range(1, 28), dtype='float32'))


    def index_list(self, l, x):
        return [i for i, _x in enumerate(l) if _x == x]

    def load_sample_img(self):
        img_path = "../img/sample_food.jpeg"
        img = cv2.imread(img_path)
        return img

    def cv2pil(self, img):
        ''' OpenCV型 -> PIL型 '''
        """https://qiita.com/derodero24/items/f22c22b22451609908ee"""
        new_img = img.copy()
        new_img = new_img[:, :, ::-1]
        new_img = Image.fromarray(new_img)
        return new_img

    def top_n_list(self, output, n):
        possible_list = []
        temp = 0
        for value in (heapq.nlargest(n, output)):
            if value == temp:
                break
            else:
                for ind in self.index_list(output, value):
                    possible_list.append([ind, self.allergen_list[ind] ,value])
            temp = value        
        return possible_list


    def predict(self, img, output="a", debug=False):
        if debug:
            img = self.load_sample_img()
        img = self.cv2pil(img)
        img = self.transform(img)
        
        with torch.no_grad():
            if output == "a":
                output_a = self.classifier(img.unsqueeze(0))
            elif output == "f":
                output_f = self.classifier(img.unsqueeze(0))
                output_a = torch.mm(output_f, torch.t(self.weights))
            else:
                print("output should be \"a\" or \"f\"")
                return

        n = 3

        if output == "a":
            output_a = output_a.tolist()[0]
            possible_allergen_list = self.top_n_list(output_a, n)
            print(possible_allergen_list)
            return possible_allergen_list

        elif output == "f":
            output_f = output_f.tolist()[0]
            output_a = output_a.tolist()[0]
            possible_foods_list = self.top_n_list(output_f, n)
            possible_allergen_list = self.top_n_list(output_a, n)
            print(possible_foods_list)
            print(possible_allergen_list)
            return possible_foods_list, possible_allergen_list
